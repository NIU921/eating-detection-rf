"""外层身份五折、内层身份三折随机森林；逐内折/外折保存检查点。"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time
import csv

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from split_meal_datasets import load, sid
from .config import Config
from .events import decode, remove_ignored, score_events, truth_for


DEFAULT_TRAINING = dict(n_estimators=300, n_jobs=-1, inner_folds=3, seed=20261021,
    max_depths=[12,None], min_samples_leaf=[5,20], thresholds=[.35,.45,.55,.65],
    smooth_s=60, start_persistence=3, stop_persistence=6, merge_gap_s=30, min_event_s=60)


def _hash_files(paths):
    h=hashlib.sha256()
    for path in paths:
        h.update(Path(path).name.encode("utf-8"))
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(4*1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def _json(path, value):
    path=Path(path); temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(path)


def load_dataset(dataset):
    dataset=Path(dataset)
    if dataset.is_dir():
        if (dataset/".preparing").exists(): raise ValueError("数据准备未完成，不能开始训练")
        summary=json.loads((dataset/"dataset_summary.json").read_text(encoding="utf-8"))
        X=np.load(dataset/"X.npy",mmap_mode="r",allow_pickle=False)
        with (dataset/"metadata.jsonl").open(encoding="utf-8") as stream:
            metadata=[json.loads(line) for line in stream if line.strip()]
        names=json.loads((dataset/"feature_names.json").read_text(encoding="utf-8"))
        stable_summary={k:v for k,v in summary.items() if k!="elapsed_s"}
        content_hash=_hash_files([dataset/n for n in ("X.npy","metadata.jsonl","feature_names.json")])
        fingerprint=hashlib.sha256((content_hash+json.dumps(stable_summary,sort_keys=True)).encode()).hexdigest()
    else:
        with np.load(dataset,allow_pickle=False) as archive:
            X=archive["X"].astype(np.float32)
            metadata=[json.loads(v) for v in archive["metadata"].tolist()]
            names=archive["feature_names"].tolist()
        summary=dict(config=Config().to_dict(),limited_run=False)
        fingerprint=_hash_files([dataset])
    if X.shape!=(len(metadata),len(names)): raise ValueError("特征矩阵、字段及窗口索引不匹配")
    if not np.isfinite(X).all(): raise ValueError("特征中存在NaN/无穷值")
    return X,metadata,names,summary,fingerprint


def weights(metadata, indices, y):
    counts=Counter((metadata[i]["leakage_group"],int(y[i])) for i in indices)
    w=np.array([1/counts[(metadata[i]["leakage_group"],int(y[i]))] for i in indices],dtype=float)
    for label in (0,1):
        mask=y[indices]==label
        if mask.any(): w[mask]*=.5/w[mask].sum()
    return w*len(indices)


def _model(depth,leaf,seed,settings):
    return RandomForestClassifier(n_estimators=settings["n_estimators"],max_depth=depth,
        min_samples_leaf=leaf,max_features="sqrt",n_jobs=settings["n_jobs"],random_state=seed)


def _decoder(settings, signal_config):
    stride=signal_config["stride_s"]
    return dict(smooth_windows=max(1,round(settings["smooth_s"]/stride)),stride_s=stride,
        start_persistence=settings["start_persistence"],stop_persistence=settings["stop_persistence"],
        merge_gap_s=settings["merge_gap_s"],min_event_s=settings["min_event_s"])


def _event_score(metadata, probability, root, subjects, threshold, decoder):
    predicted=decode(metadata,probability,threshold=threshold,**decoder)
    return score_events(remove_ignored(predicted,root,subjects),truth_for(root,subjects))


def _csv_metrics(output, rows):
    if not rows: return
    with (output/"outer_fold_metrics.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def fit_cv(dataset, root, output, settings=None):
    root,output=Path(root).resolve(),Path(output).resolve()
    settings=dict(DEFAULT_TRAINING,**(settings or {}))
    unknown=set(settings)-set(DEFAULT_TRAINING)
    if unknown: raise ValueError(f"训练配置中存在未知字段: {sorted(unknown)}")
    if settings["n_estimators"]<1 or settings["inner_folds"]<2 or settings["n_jobs"]==0:
        raise ValueError("树数量、内折数或n_jobs无效")
    if not settings["thresholds"] or any(not .2<t<=1 for t in settings["thresholds"]):
        raise ValueError("开始阈值必须位于(0.2,1]，停止阈值低0.20")
    output.mkdir(parents=True,exist_ok=True); (output/"checkpoints").mkdir(exist_ok=True)
    (output/"models").mkdir(exist_ok=True)
    print("TRAIN loading dataset",flush=True)
    X,metadata,names,summary,dataset_hash=load_dataset(dataset)
    if summary.get("limited_run"): raise ValueError("这是limit-files试运行数据，不能用于正式五折训练")
    y=np.asarray([m["label"] for m in metadata],dtype=np.int8)
    groups=np.asarray([m["leakage_group"] for m in metadata]); folds=np.asarray([m["fold"] for m in metadata])
    supervised=np.flatnonzero(y>=0)
    if set(np.unique(y[supervised]))!={0,1}: raise ValueError("必须同时存在正例和负例")
    if set(np.unique(folds))!={1,2,3,4,5}: raise ValueError("五个测试折都必须有有效窗口")
    for group in set(groups):
        if len(set(folds[groups==group]))!=1: raise ValueError(f"身份组跨折: {group}")
    fold_path=root/"惯用手算法设计"/"五折划分"/"subject_folds.csv"
    fold_records=[r for r,_ in load(fold_path,["externalid","fold","leakage_group"])["records"] if r]
    fold_by_subject={sid(r["externalid"]):r for r in fold_records}
    for r in metadata:
        expected=fold_by_subject[r["externalid"]]
        if int(expected["fold"])!=int(r["fold"]) or expected["leakage_group"]!=r["leakage_group"]:
            raise ValueError("数据集分折与当前五折索引不同，请重新准备数据")
    fold_subjects={f:[sid(r["externalid"]) for r in fold_records if int(r["fold"])==f] for f in range(1,6)}
    code_paths=[Path(__file__).parent/n for n in ("train.py","events.py","labels.py")]
    label_paths=[root/"三类筛选结果"/n for n in ("惯用手用餐信息.csv","非惯用手用餐信息.csv")]+[fold_path]
    manifest=dict(dataset_sha256=dataset_hash,settings=settings,signal_config=summary["config"],
        code_sha256=_hash_files(code_paths),labels_sha256=_hash_files(label_paths),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                      sklearn=sklearn.__version__,joblib=joblib.__version__))
    manifest_path=output/"run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8"))!=manifest:
            raise ValueError("此输出目录属于不同数据/配置/代码/环境的实验，请使用新--output目录")
    else: _json(manifest_path,manifest)
    decoder=_decoder(settings,summary["config"]); oof=np.full(len(y),np.nan); fold_rows=[]
    started=time.monotonic(); seed=settings["seed"]
    for fold in range(1,6):
        test=np.flatnonzero(folds==fold); train=np.flatnonzero(folds!=fold)
        if len(set(groups[train]))<settings["inner_folds"]: raise ValueError("训练池身份组不足以内部交叉验证")
        checkpoint=output/"checkpoints"/f"outer_{fold}.npz"
        model_path=output/"models"/f"fold_{fold}.joblib"
        if checkpoint.exists() and model_path.exists():
            with np.load(checkpoint,allow_pickle=False) as cache:
                if not np.array_equal(cache["indices"],test): raise ValueError("外折检查点索引不一致")
                oof[test]=cache["probability"]; score=json.loads(str(cache["score"]))
            fold_rows.append(score); _csv_metrics(output,fold_rows)
            print(f"TRAIN outer={fold}/5 resumed f1={score['f1']:.4f}",flush=True); continue
        print(f"TRAIN outer={fold}/5 train_pool={len(train)} test_windows={len(test)}",flush=True)
        inner=GroupKFold(n_splits=settings["inner_folds"])
        splits=list(inner.split(np.zeros(len(train)),y[train],groups[train])); best=None
        for depth,leaf in itertools.product(settings["max_depths"],settings["min_samples_leaf"]):
            val_idx=[]; val_p=[]
            for k,(itr,iva) in enumerate(splits):
                itr,iva=train[itr],train[iva]; itr=itr[y[itr]>=0]
                if set(np.unique(y[itr]))!={0,1}: raise ValueError("内层训练折缺少正例或负例")
                cache_path=output/"checkpoints"/f"inner_o{fold}_d{depth}_l{leaf}_k{k}.npz"
                if cache_path.exists():
                    with np.load(cache_path,allow_pickle=False) as cache:
                        if not np.array_equal(cache["indices"],iva): raise ValueError("内折检查点索引不一致")
                        probability=cache["probability"]
                else:
                    print(f"TRAIN outer={fold}/5 depth={depth} leaf={leaf} inner={k+1}/{len(splits)} fitting={len(itr)}",flush=True)
                    clf=_model(depth,leaf,seed+fold*100+k,settings)
                    clf.fit(X[itr],y[itr],sample_weight=weights(metadata,itr,y))
                    probability=clf.predict_proba(X[iva])[:,1]
                    temporary=cache_path.with_suffix(".tmp")
                    with temporary.open("wb") as stream: np.savez_compressed(stream,indices=iva,probability=probability)
                    temporary.replace(cache_path)
                val_idx.extend(iva.tolist()); val_p.extend(probability.tolist())
            val_meta=[metadata[i] for i in val_idx]; subjects=sorted({r["externalid"] for r in val_meta})
            for threshold in settings["thresholds"]:
                result=_event_score(val_meta,val_p,root,subjects,threshold,decoder)
                start=result["start_mae_s"] if result["start_mae_s"] is not None else float("inf")
                end=result["end_mae_s"] if result["end_mae_s"] is not None else float("inf")
                key=(result["f1"],-result["fp"],-start-end,-(depth or 999),leaf)
                if best is None or key>best[0]: best=(key,depth,leaf,threshold,result)
        _,depth,leaf,threshold,inner_result=best
        fit_indices=train[y[train]>=0]
        print(f"TRAIN outer={fold}/5 refit depth={depth} leaf={leaf} threshold={threshold}",flush=True)
        clf=_model(depth,leaf,seed+fold,settings)
        clf.fit(X[fit_indices],y[fit_indices],sample_weight=weights(metadata,fit_indices,y))
        probability=clf.predict_proba(X[test])[:,1]; oof[test]=probability
        result=_event_score([metadata[i] for i in test],probability,root,fold_subjects[fold],threshold,decoder)
        result.update(fold=fold,max_depth=depth,min_samples_leaf=leaf,threshold=threshold,
            inner_event_f1=inner_result["f1"],train_windows=len(fit_indices),test_windows=len(test),
            train_leakage_groups=len(set(groups[train])),test_leakage_groups=len(set(groups[test])),
            test_externalids=len(fold_subjects[fold]))
        joblib.dump(dict(model=clf,threshold=threshold,feature_names=names,
            signal_config=summary["config"],decoder=decoder),model_path)
        temporary=checkpoint.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,indices=test,probability=probability,score=np.asarray(json.dumps(result)))
        temporary.replace(checkpoint); fold_rows.append(result); _csv_metrics(output,fold_rows)
        print(f"TRAIN outer={fold}/5 done f1={result['f1']:.4f} elapsed_s={time.monotonic()-started:.0f}",flush=True)
    if not np.isfinite(oof).all(): raise RuntimeError("外层留出概率不完整")
    all_pred,all_truth,raw_predictions={},{},{}
    for row in fold_rows:
        f=row["fold"]; indices=np.flatnonzero(folds==f)
        events=decode([metadata[i] for i in indices],oof[indices],threshold=row["threshold"],**decoder)
        raw_predictions.update(events)
        all_pred.update(remove_ignored(events,root,fold_subjects[f]))
        all_truth.update(truth_for(root,fold_subjects[f]))
    overall=score_events(all_pred,all_truth)
    with (output/"oof_window_predictions.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metadata[0])+["probability"]); writer.writeheader()
        for r,p in zip(metadata,oof): writer.writerow(dict(r,probability=float(p)))
    with (output/"oof_events.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream); writer.writerow(("externalid","beforeTime","afterTime"))
        for subject,events in sorted(raw_predictions.items()):
            for a,b in events: writer.writerow((subject,a,b))
    subject_metrics={s:score_events({s:all_pred.get(s,[])},{s:all_truth[s]}) for s in sorted(all_truth)}
    _json(output/"subject_metrics.json",subject_metrics)
    values=np.array([r["f1"] for r in fold_rows]); summary_f1=dict(mean=float(values.mean()),std=float(values.std(ddof=1)))
    depth=Counter(r["max_depth"] for r in fold_rows).most_common(1)[0][0]
    leaf=int(np.median([r["min_samples_leaf"] for r in fold_rows])); threshold=float(np.median([r["threshold"] for r in fold_rows]))
    final_path=output/"final_model.joblib"
    if not final_path.exists():
        print("TRAIN refitting deployment model on all supervised windows",flush=True)
        final_model=_model(depth,leaf,seed,settings)
        final_model.fit(X[supervised],y[supervised],sample_weight=weights(metadata,supervised,y))
        joblib.dump(dict(model=final_model,threshold=threshold,feature_names=names,
            signal_config=summary["config"],decoder=decoder),final_path)
    predicted_labels=(oof[supervised]>=.5).astype(int)
    precision,recall,f1,_=precision_recall_fscore_support(y[supervised],predicted_labels,average="binary",zero_division=0)
    report=dict(settings=settings,outer_fold_metrics=fold_rows,outer_oof_event_metrics=overall,
        fold_event_f1_mean_std=summary_f1,subject_metrics_file="subject_metrics.json",
        auxiliary_window_metrics_at_05=dict(precision=float(precision),recall=float(recall),f1=float(f1),
            pr_auc=float(average_precision_score(y[supervised],oof[supervised]))),
        windows=len(y),supervised_windows=len(supervised),dataset_summary=summary,
        final_refit=dict(max_depth=depth,min_samples_leaf=leaf,threshold=threshold),
        elapsed_s=round(time.monotonic()-started,2),
        note="事件主指标为IoU严格>0.25的一对一匹配；全标注餐次口径，缺失信号未检出计FN；非惯用手餐次忽略。部署参数由各外层内部选择参数聚合，不使用外层成绩择优。")
    _json(output/"training_report.json",report)
    print(json.dumps(dict(status="complete",outer_oof_event_metrics=overall,fold_event_f1_mean_std=summary_f1),ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",type=Path,nargs="?",default=Path("runs/baseline/dataset"))
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--output",type=Path,default=Path("runs/baseline/training"))
    p.add_argument("--config",type=Path,help="训练与事件解码参数JSON")
    p.add_argument("--jobs",type=int,help="覆盖配置中的n_jobs")
    p.add_argument("--trees",type=int,help="覆盖配置中的n_estimators")
    a=p.parse_args(); settings=json.loads(a.config.read_text(encoding="utf-8-sig")) if a.config else {}
    if a.jobs is not None: settings["n_jobs"]=a.jobs
    if a.trees is not None: settings["n_estimators"]=a.trees
    fit_cv(a.dataset,a.root,a.output,settings)


if __name__=="__main__": main()
