"""复用已完成森林的内外折概率，仅在内折搜索餐次解码规则。"""
import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from split_meal_datasets import load,sid
from .labels import merged,overlap,read_catalog,subtract
from .events import score_events
from .train import load_dataset,_hash_files,_json
from .integrity import validate_labels,valid_probability


def cache_version(root,manifest):
    current=_hash_files([Path(__file__).with_name(n) for n in ('train.py','events.py','labels.py','integrity.py')])
    if manifest['code_sha256']==current: return 'current'
    # 审查过的旧版仅遗漏内折零窗口受试者；其外折概率仍可按原规则独立评价。
    legacy='86df7ec7636d82b2985493ff7a4852349b42c2cdb62198f41b44e7e75a920582'
    snapshot=Path(root)/'runs/code_review_20261022/source_before'
    paths=[snapshot/n for n in ('train.py','events.py','labels.py')]
    if manifest['code_sha256']==legacy and all(p.is_file() for p in paths) and _hash_files(paths)==legacy:
        if all((snapshot/n).read_bytes()==Path(__file__).with_name(n).read_bytes() for n in ('events.py','labels.py')):
            return 'reviewed_legacy_inner_missing_subjects'
    raise ValueError('基线训练或评分代码已改变且未通过兼容性审查，不能直接复用此实验')


class CachedDecoder:
    """与events.decode同规则；复用分段和平滑，避免每个候选重新处理全部窗口。"""
    def __init__(self,metadata,probability,stride_s):
        groups=defaultdict(list); support=defaultdict(list)
        for row,p in zip(metadata,probability):
            a,b=int(row['start_ms']),int(row['end_ms'])
            groups[(row['externalid'],row['source_zip'],row['member'],row['segment'])].append((a,b,float(p)))
            support[row['externalid']].append((a,b))
        self.support={s:merged(v) for s,v in support.items()}; self.traces=[]; self.smoothed={}
        for key,items in groups.items():
            values=np.asarray(sorted(items),dtype=float)
            edges=np.r_[0,np.flatnonzero(np.diff(values[:,0])>stride_s*1500)+1,len(values)]
            for a,b in zip(edges[:-1],edges[1:]): self.traces.append((key[0],values[a:b]))
        self.stride_s=stride_s

    def decode(self,settings):
        n=max(1,round(settings['smooth_s']/self.stride_s))
        if n not in self.smoothed:
            left=(n-1)//2; right=n-left
            # 与原始实现使用相同均值运算，保持阈值附近的数值行为。
            self.smoothed[n]=[np.array([v[max(0,i-left):min(len(v),i+right),2].mean()
                                       for i in range(len(v))]) for _,v in self.traces]
        events=defaultdict(list); onset=settings['threshold']; offset=onset-.2
        start_n=settings['start_persistence']; stop_n=settings['stop_persistence']
        for (subject,values),ps in zip(self.traces,self.smoothed[n]):
            active=False; high=low=0; begin=None
            for i,p in enumerate(ps):
                if not active:
                    high=high+1 if p>=onset else 0
                    if high>=start_n:
                        active=True; begin=int(values[i-start_n+1,0]); low=0
                else:
                    low=low+1 if p<offset else 0
                    if low>=stop_n:
                        end=int(values[i-stop_n+1,1])
                        if end>begin: events[subject].append((begin,end))
                        active=False; high=low=0
            if active and int(values[-1,1])>begin: events[subject].append((begin,int(values[-1,1])))
        for subject,intervals in events.items():
            combined=[]
            for a,b in sorted(intervals):
                end=combined[-1][1] if combined else a
                gap=max(0,a-end)
                can_merge=bool(combined) and gap<=settings['merge_gap_s']*1000 and (
                    gap==0 or overlap(end,a,self.support[subject])>=gap)
                if can_merge: combined[-1]=(combined[-1][0],max(b,combined[-1][1]))
                else: combined.append((a,b))
            events[subject]=[(a,b) for a,b in combined if b-a>=settings['min_event_s']*1000]
        return dict(events)


def candidates(grid):
    keys=['smooth_s','stop_persistence','merge_gap_s','threshold']
    if set(grid)!=set(keys)|{'start_persistence','min_event_s'}: raise ValueError('搜索配置字段不正确')
    if any(not isinstance(grid[k],list) or not grid[k] for k in keys): raise ValueError('候选参数必须是非空列表')
    for settings in (dict(zip(keys,v),start_persistence=grid['start_persistence'],min_event_s=grid['min_event_s'])
                     for v in itertools.product(*(grid[k] for k in keys))):
        if any(type(settings[k]) is not int or settings[k]<1 for k in ('smooth_s','stop_persistence','start_persistence')):
            raise ValueError('平滑时长和连续确认数必须为正整数')
        if not .2<settings['threshold']<=1 or settings['merge_gap_s']<0 or settings['min_event_s']<0:
            raise ValueError('阈值、合并间隔或事件时长无效')
        yield settings


def tune(root,baseline,output,grid):
    root,baseline,output=map(lambda p:Path(p).resolve(),(root,baseline,output))
    if output==baseline or baseline in output.parents: raise ValueError('新输出必须位于原基线目录之外')
    choices=list(candidates(grid)); training=baseline/'training'; started=time.monotonic()
    print('TUNE loading and verifying baseline caches',flush=True)
    _,metadata,_,summary,dataset_hash=load_dataset(baseline/'dataset')
    validate_labels(root,metadata,summary['config'],summary.get('background_policy','all_nonmeal_confirmed'))
    old=json.loads((training/'training_report.json').read_text(encoding='utf-8'))
    old_manifest=json.loads((training/'run_manifest.json').read_text(encoding='utf-8'))
    if old_manifest['dataset_sha256']!=dataset_hash: raise ValueError('基线缓存与当前数据不一致')
    fold_path=root/'惯用手算法设计/五折划分/subject_folds.csv'
    label_paths=[root/'三类筛选结果'/name for name in ('惯用手用餐信息.csv','非惯用手用餐信息.csv')]+[fold_path]
    if old_manifest['labels_sha256']!=_hash_files(label_paths): raise ValueError('标注或五折映射已改变，不能复用概率')
    version=cache_version(root,old_manifest)
    fold_records=[r for r,_ in load(fold_path,['externalid','fold','leakage_group'])['records'] if r]
    fold_subjects={f:[sid(r['externalid']) for r in fold_records if int(r['fold'])==f] for f in range(1,6)}
    mapping={sid(r['externalid']):r for r in fold_records}
    for row in metadata:
        expected=mapping[row['externalid']]
        if row['fold']!=int(expected['fold']) or row['leakage_group']!=expected['leakage_group']:
            raise ValueError('窗口索引与固定身份五折不一致')
    catalog=read_catalog(root)
    truth={s:[(m['start'],m['end']) for m in meals if m['dominant']] for s,meals in catalog.items()}
    ignored={s:[(m['start'],m['end']) for m in meals if not m['dominant']] for s,meals in catalog.items()}
    def score(events,subjects):
        return score_events({s:subtract(events.get(s,[]),ignored.get(s,[])) for s in subjects},
                            {s:truth.get(s,[]) for s in subjects})
    folds=np.array([r['fold'] for r in metadata]); groups=np.array([r['leakage_group'] for r in metadata])
    labels=np.array([r['label'] for r in metadata]); selected={r['fold']:r for r in old['outer_fold_metrics']}
    if set(selected)!={1,2,3,4,5} or summary.get('limited_run'): raise ValueError('需要已完成的正式基线五折')
    paths=[training/'run_manifest.json',training/'training_report.json',training/'final_model.joblib']
    for f,row in selected.items():
        paths.append(training/'checkpoints'/f'outer_{f}.npz')
        paths.extend(training/'checkpoints'/f"inner_o{f}_d{row['max_depth']}_l{row['min_samples_leaf']}_k{k}.npz"
                     for k in range(old['settings']['inner_folds']))
    manifest=dict(grid=grid,baseline=str(baseline),baseline_files_sha256=_hash_files(paths),dataset_sha256=dataset_hash,
        code_sha256=_hash_files([Path(__file__),Path(__file__).with_name('integrity.py')]),
        baseline_cache_version=version,labels_sha256=_hash_files(label_paths))
    output.mkdir(parents=True,exist_ok=True); (output/'checkpoints').mkdir(exist_ok=True)
    if (output/'run_manifest.json').exists():
        if json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))!=manifest:
            raise ValueError('此输出目录属于不同配置或缓存，请换新输出目录')
    else: _json(output/'run_manifest.json',manifest)
    rows=[]; all_raw={}
    for fold in range(1,6):
        checkpoint=output/'checkpoints'/f'fold_{fold}.json'
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text(encoding='utf-8')); rows.append(saved['metrics']); all_raw.update(saved['events'])
            print(f'TUNE outer={fold}/5 resumed',flush=True); continue
        train=np.flatnonzero(folds!=fold); test=np.flatnonzero(folds==fold); forest=selected[fold]
        val_indices=[]; val_probability=[]
        inner=GroupKFold(n_splits=old['settings']['inner_folds'])
        for k,(itr,iva) in enumerate(inner.split(np.zeros(len(train)),labels[train],groups[train])):
            expected=train[iva]
            if set(groups[train[itr]])&set(groups[expected]): raise ValueError('内折身份泄漏')
            path=training/'checkpoints'/f"inner_o{fold}_d{forest['max_depth']}_l{forest['min_samples_leaf']}_k{k}.npz"
            with np.load(path,allow_pickle=False) as cache:
                if not np.array_equal(cache['indices'],expected): raise ValueError('内折缓存索引不符')
                val_indices.extend(expected.tolist()); val_probability.extend(valid_probability(cache['probability'],len(expected)).tolist())
        if len(val_indices)!=len(train) or len(set(val_indices))!=len(train): raise ValueError('内折预测不完整')
        if not np.isfinite(val_probability).all(): raise ValueError('内折概率包含非有限值')
        decoder=CachedDecoder([metadata[i] for i in val_indices],val_probability,summary['config']['stride_s'])
        subjects=[s for f,v in fold_subjects.items() if f!=fold for s in v]; best=None
        with (output/f'inner_search_fold_{fold}.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=None
            for number,settings in enumerate(choices,1):
                metrics=score(decoder.decode(settings),subjects); record=dict(settings,**metrics)
                if writer is None: writer=csv.DictWriter(stream,fieldnames=list(record)); writer.writeheader()
                writer.writerow(record)
                error=sum(metrics[k] if metrics[k] is not None else float('inf') for k in ('start_mae_s','end_mae_s'))
                key=(metrics['f1'],-metrics['fp'],-error)
                if best is None or key>best[0]: best=(key,dict(settings),metrics)
                if number==1 or number%12==0 or number==len(choices):
                    stream.flush(); print(f'TUNE outer={fold}/5 candidates={number}/{len(choices)} best_inner_f1={best[2]["f1"]:.4f}',flush=True)
        _,settings,inner_metrics=best; del decoder
        with np.load(training/'checkpoints'/f'outer_{fold}.npz',allow_pickle=False) as cache:
            if not np.array_equal(cache['indices'],test): raise ValueError('外折缓存索引不符')
            probability=valid_probability(cache['probability'],len(test))
        decoder=CachedDecoder([metadata[i] for i in test],probability,summary['config']['stride_s'])
        events=decoder.decode(settings); metrics=score(events,fold_subjects[fold]); del decoder
        row=dict(fold=fold,**metrics,**settings,inner_event_f1=inner_metrics['f1'],
                 baseline_f1=forest['f1'],max_depth=forest['max_depth'],min_samples_leaf=forest['min_samples_leaf'])
        rows.append(row); all_raw.update(events)
        _json(checkpoint,dict(metrics=row,events=events))
        print(f'TUNE outer={fold}/5 done f1={metrics["f1"]:.4f}',flush=True)
    subjects=[s for values in fold_subjects.values() for s in values]; overall=score(all_raw,subjects)
    with (output/'outer_fold_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output/'oof_events.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(('externalid','beforeTime','afterTime'))
        for s,events in sorted(all_raw.items()):
            for a,b in events: writer.writerow((s,a,b))
    _json(output/'subject_metrics.json',{s:score(all_raw,[s]) for s in subjects})
    deployment={k:float(np.median([r[k] for r in rows])) for k in ('threshold','smooth_s','stop_persistence','merge_gap_s')}
    deployment.update(start_persistence=grid['start_persistence'],min_event_s=grid['min_event_s'])
    for k in ('smooth_s','stop_persistence'): deployment[k]=int(deployment[k])
    final=joblib.load(training/'final_model.joblib'); final['threshold']=deployment['threshold']
    final['decoder']=dict(smooth_windows=max(1,round(deployment['smooth_s']/summary['config']['stride_s'])),
        stride_s=summary['config']['stride_s'],start_persistence=deployment['start_persistence'],
        stop_persistence=deployment['stop_persistence'],merge_gap_s=deployment['merge_gap_s'],min_event_s=deployment['min_event_s'])
    temporary=output/'final_model.joblib.tmp'; joblib.dump(final,temporary); temporary.replace(output/'final_model.joblib')
    values=np.array([r['f1'] for r in rows])
    result=dict(outer_oof_event_metrics=overall,outer_fold_metrics=rows,
        fold_event_f1_mean_std=dict(mean=float(values.mean()),std=float(values.std(ddof=1))),
        baseline_event_metrics=old['outer_oof_event_metrics'],deployment_decoder=deployment,
        elapsed_s=round(time.monotonic()-started,2),
        note='森林参数沿用原内折选择，仅在相应内折概率上重选解码规则；外折概率不参与选择。此为已查看基线外折表现后的开发迭代，最终泛化仍需独立测试验证。')
    _json(output/'training_report.json',result)
    print(json.dumps(dict(status='complete',outer_oof_event_metrics=overall),ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); root=Path(__file__).resolve().parents[1]
    p.add_argument('--root',type=Path,default=root); p.add_argument('--baseline',type=Path,default=root/'runs/local_baseline')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--config',type=Path,required=True)
    a=p.parse_args(); tune(a.root,a.baseline,a.output,json.loads(a.config.read_text(encoding='utf-8-sig')))


if __name__=='__main__': main()
