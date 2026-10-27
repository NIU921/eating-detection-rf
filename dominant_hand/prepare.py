"""从解压目录或ZIP准备窗口，按来源缓存并记录用户明确排除的文件。"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
from contextlib import nullcontext
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from split_meal_datasets import load, sid, zip_name
from .config import Config
from .features import extract, FEATURE_NAMES
from .labels import merged, overlap, read_catalog, read_verified, window_label
from .signal import process
from .sources import SensorSource, affected_meals, read_exclusions, resolve_source


def _rows(path, required):
    return [r for r, _ in load(path, required)["records"] if r]


def _archive(task):
    cache = Path(task["cache"])
    if cache.is_file():
        with np.load(cache,allow_pickle=False) as previous:
            if json.loads(str(previous["audit"]))["status"]=="ok":
                return str(cache), True
    cfg = Config(**task["config"])
    matrix, metadata, audit = [], [], []
    try:
        with nullcontext(SensorSource(Path(task["path"]),task["source_kind"])) as archive:
            members = archive.members()
            if not members:
                raise ValueError("ZIP中没有TXT")
            for member in members:
                with archive.open_text(member) as text:
                    segments, report = process(text, cfg)
                    report["text_encoding"]=text.encoding
                member_windows = 0
                for si, segment in enumerate(segments):
                    t, x, observed = segment["t"], segment["x"], segment["observed"]
                    negatives = task["verified"]
                    if negatives is not None and task["negative_file"]:
                        negatives = list(negatives) + [(int(t[0]), int(t[-1]) + round(1000/cfg.fs))]
                    length, stride = cfg.window_s*cfg.fs, cfg.stride_s*cfg.fs
                    for begin in range(0, len(t)-length+1, stride):
                        end = begin+length
                        coverage = float(observed[begin:end].mean())
                        if coverage < cfg.min_observed:
                            continue
                        start_ms, end_ms = int(t[begin]), int(t[end-1])+round(1000/cfg.fs)
                        label, reason = window_label(start_ms, end_ms, task["meals"], cfg, negatives)
                        matrix.append(extract(x[begin:end], cfg.fs))
                        metadata.append(dict(externalid=task["subject"], leakage_group=task["group"],
                            fold=task["fold"], source_zip=task["source"],source_kind=task["source_kind"],member=member,
                            segment=si, start_ms=start_ms, end_ms=end_ms, label=label,
                            label_source=reason, observed_coverage=coverage,
                            background_source="non_eating_file" if task["negative_file"] else "outside_meals_in_recording"))
                        member_windows += 1
                audit.append(dict(member=member, windows=member_windows, **report))
        status, error = "ok", None
    except (OSError, ValueError, UnicodeError, zipfile.BadZipFile) as exc:
        matrix, metadata = [], []
        status, error = "error", f"{type(exc).__name__}: {exc}"
    report = dict(externalid=task["subject"], source_zip=task["source"],source_kind=task["source_kind"],
                  status=status, error=error, windows=len(metadata), members=audit)
    temporary = cache.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, X=np.asarray(matrix, dtype=np.float32).reshape(-1,len(FEATURE_NAMES)),
            metadata=np.asarray([json.dumps(r,ensure_ascii=False).encode("utf-8") for r in metadata],dtype="S"),
            audit=np.asarray(json.dumps(report, ensure_ascii=False)))
    temporary.replace(cache)
    return str(cache), False


def build(root, output, verified_path=None, config=None, workers=2, cache_dir=None,
          imu_dir=None, limit_files=None, allow_file_errors=False,source_mode="auto"):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.suffix == ".npz":
        raise ValueError("新版--output是数据集目录，例如runs/baseline/dataset，不能以.npz结尾")
    cfg = config or Config()
    imu_dir = Path(imu_dir).resolve() if imu_dir else root/"IMU+PPG"
    if workers < 1:
        raise ValueError("workers必须大于0")
    if not imu_dir.is_dir():
        raise FileNotFoundError(f"原始ZIP目录不存在: {imu_dir}")
    output.mkdir(parents=True, exist_ok=True)
    (output/".preparing").write_text("数据准备尚未完成；完成后由程序自动移除",encoding="utf-8")
    cache_dir = Path(cache_dir).resolve() if cache_dir else output/"cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog(root)
    verified = read_verified(verified_path) if verified_path else None
    fold_file = root/"惯用手算法设计"/"五折划分"/"subject_folds.csv"
    folds = {sid(r["externalid"]): r for r in _rows(fold_file,["externalid","leakage_group","fold"])}
    negative_files = {(sid(r["externalid"]),zip_name(r["sensorData"])) for r in
        _rows(root/"三类筛选结果"/"非进食数据_IMU+PPG.csv",["externalid","sensorData"])}
    source_rows = _rows(root/"IMU+PPG.csv",["externalid","sensorData"])
    selected = sorted({(sid(r["externalid"]),zip_name(r["sensorData"])) for r in source_rows
                       if sid(r["externalid"]) in folds})
    audit = [dict(externalid=s,source_zip=n,status="excluded_unverified_identity")
             for s,n in selected if s=="6C7"]
    selected = [(s,n) for s,n in selected if s!="6C7"]
    exclusions=read_exclusions(root)
    excluded=[dict(externalid=s,source_zip=n,status="excluded_by_user",reason=exclusions[n]["reason"])
              for s,n in selected if n in exclusions]
    audit.extend(excluded)
    selected=[(s,n) for s,n in selected if n not in exclusions]
    sources={(s,n):resolve_source(imu_dir,n,source_mode) for s,n in selected}
    missing = [dict(externalid=s,source_zip=n,status="missing_zip") for s,n in selected
               if sources[s,n] is None]
    if missing:
        (output/"missing_files.json").write_text(json.dumps(missing,ensure_ascii=False,indent=2),encoding="utf-8")
        if not allow_file_errors:
            raise FileNotFoundError(f"缺少{len(missing)}个ZIP，见{output/'missing_files.json'}；请补齐数据")
    audit.extend(missing)
    selected = [(s,n) for s,n in selected if sources[s,n] is not None]
    if limit_files is not None:
        if limit_files < 1: raise ValueError("limit-files必须大于0")
        selected = selected[:limit_files]
    if not selected: raise ValueError("没有找到可处理的ZIP，请检查--root和--imu-dir")
    code_hash = hashlib.sha256()
    for name in ("prepare.py","features.py","signal.py","labels.py","config.py","sources.py"):
        code_hash.update((Path(__file__).parent/name).read_bytes())
    tasks=[]
    for subject,name in selected:
        source=sources[subject,name]; path=source.path; fold=folds[subject]
        provenance=dict(config=cfg.to_dict(),subject=subject,group=fold["leakage_group"],
            fold=int(fold["fold"]),meals=catalog.get(subject,[]),
            verified=None if verified is None else verified.get(subject,[]),
            negative_file=(subject,name) in negative_files,source=name,
            source_state=source.fingerprint(),source_kind=source.kind,code=code_hash.hexdigest())
        key=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
        tasks.append(dict(provenance,path=str(path),cache=str(cache_dir/f"{key}.npz")))
    print(f"PREPARE files={len(tasks)} workers={workers} source={imu_dir}",flush=True)
    started=time.monotonic(); paths=[]
    pool=ProcessPoolExecutor(max_workers=workers) if workers>1 else None
    try:
        results=pool.map(_archive,tasks) if pool else map(_archive,tasks)
        for index,(path,was_cached) in enumerate(results,1):
            paths.append(path)
            with np.load(path,allow_pickle=False) as shard: report=json.loads(str(shard["audit"]))
            audit.append(report)
            print(f"PREPARE {index}/{len(tasks)} cached={was_cached} windows={report['windows']} "
                  f"status={report['status']} elapsed_s={time.monotonic()-started:.0f}",flush=True)
    finally:
        if pool: pool.shutdown(wait=True,cancel_futures=True)
    (output/"prepare_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    errors=[r for r in audit if r["status"]=="error"]
    if errors and not allow_file_errors:
        raise RuntimeError(f"{len(errors)}个ZIP处理失败，见prepare_audit.json；修复源文件后同命令可利用缓存续跑")
    # 来源冲突按排序后的ZIP优先；同片段内有意重叠的滑动窗保留。
    covered=defaultdict(list); descriptors=[]; total=0; duplicates=0
    for path in paths:
        with np.load(path,allow_pickle=False) as shard: rows=[json.loads(v) for v in shard["metadata"].tolist()]
        keep=[]; regions=defaultdict(list)
        for i,r in enumerate(rows): regions[(r["externalid"],r["member"],r["segment"])].append(i)
        for (subject,_,_),indices in regions.items():
            prior=covered[subject]; new_intervals=[]
            for i in indices:
                r=rows[i]
                if overlap(r["start_ms"],r["end_ms"],prior):
                    duplicates+=1; continue
                keep.append(i); new_intervals.append((r["start_ms"],r["end_ms"]))
            covered[subject]=merged(prior+new_intervals)
        descriptors.append((path,np.array(sorted(keep),dtype=np.int64))); total+=len(keep)
    if not total: raise RuntimeError("未生成有效窗口，见prepare_audit.json")
    matrix=np.lib.format.open_memmap(output/"X.npy",mode="w+",dtype=np.float32,shape=(total,len(FEATURE_NAMES)))
    labels=Counter(); counts=defaultdict(Counter); cursor=0
    with (output/"metadata.jsonl").open("w",encoding="utf-8") as stream:
        for path,keep in descriptors:
            if not len(keep): continue
            with np.load(path,allow_pickle=False) as shard:
                matrix[cursor:cursor+len(keep)]=shard["X"][keep]
                for value in shard["metadata"][keep].tolist():
                    r=json.loads(value); stream.write(json.dumps(r,ensure_ascii=False)+"\n")
                    labels[r["label"]]+=1; counts[r["fold"]][r["label"]]+=1
            cursor+=len(keep)
    matrix.flush(); del matrix
    (output/"feature_names.json").write_text(json.dumps(FEATURE_NAMES),encoding="utf-8")
    summary=dict(windows=total,positive=labels[1],negative=labels[0],ignored=labels[-1],
        folds={str(f):dict(positive=c[1],negative=c[0],ignored=c[-1]) for f,c in sorted(counts.items())},
        duplicate_windows_removed=duplicates,source_files=len(tasks),file_errors=len(errors),
        source_counts=dict(Counter(t["source_kind"] for t in tasks)),source_mode=source_mode,
        excluded_sources=excluded,affected_meals=affected_meals(root,exclusions),
        missing_files=len(missing),config=cfg.to_dict(),feature_count=len(FEATURE_NAMES),
        background_policy="all_nonmeal_confirmed" if verified is None else "restricted_coverage",
        confirmation="2026-10-21用户确认非进食文件及含餐文件的全部餐外时间均无进食",
        source_code_sha256=code_hash.hexdigest(),limited_run=limit_files is not None,
        elapsed_s=round(time.monotonic()-started,2))
    (output/"dataset_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/".preparing").unlink()
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--imu-dir",type=Path,help="原始ZIP所在目录；默认root/IMU+PPG")
    p.add_argument("--source-mode",choices=("auto","directory","zip"),default="auto")
    p.add_argument("--output",type=Path,default=Path("runs/baseline/dataset"))
    p.add_argument("--config",type=Path,help="Config字段的JSON文件")
    p.add_argument("--verified-coverage",type=Path)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--cache-dir",type=Path)
    p.add_argument("--limit-files",type=int,help="仅用于冒烟运行，不可用于正式五折成绩")
    p.add_argument("--allow-file-errors",action="store_true",help="显式允许缺失/失败文件，必须报告排除量")
    a=p.parse_args(); cfg=Config(**json.loads(a.config.read_text(encoding="utf-8-sig"))) if a.config else Config()
    build(a.root,a.output,a.verified_coverage,cfg,a.workers,a.cache_dir,a.imu_dir,a.limit_files,a.allow_file_errors,a.source_mode)


if __name__=="__main__": main()
