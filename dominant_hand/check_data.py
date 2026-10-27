"""核对索引及IMU字段表；优先同名解压目录，缺少目录时读取ZIP。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile
from split_meal_datasets import load, sid, zip_name
from .sources import affected_meals, inspect_header, read_exclusions, resolve_source


def check(root,imu_dir,output,source_mode="auto"):
    root=Path(root).resolve(); imu_dir=Path(imu_dir).resolve() if imu_dir else root/"IMU+PPG"
    def rows(path): return [r for r,_ in load(path,["externalid"])["records"] if r]
    folds=rows(root/"惯用手算法设计"/"五折划分"/"subject_folds.csv")
    ids={sid(r["externalid"]) for r in folds}; group_folds={}
    for r in folds: group_folds.setdefault(r["leakage_group"],set()).add(int(r["fold"]))
    if any(len(v)>1 for v in group_folds.values()): raise ValueError("身份组跨折")
    sensors=rows(root/"IMU+PPG.csv")
    selected=sorted({(sid(r["externalid"]),zip_name(r["sensorData"])) for r in sensors
                     if sid(r["externalid"]) in ids and sid(r["externalid"])!="6C7"})
    original_count=len(selected); exclusions=read_exclusions(root)
    excluded=[dict(subject=s,zip=n,reason=exclusions[n]["reason"]) for s,n in selected if n in exclusions]
    selected=[(s,n) for s,n in selected if n not in exclusions]
    missing=[]; bad=[]; compressed=uncompressed=0; counts=Counter(); encodings=Counter()
    for i,(subject,name) in enumerate(selected,1):
        source=resolve_source(imu_dir,name,source_mode)
        if source is None: missing.append(name); continue
        counts[source.kind]+=1
        original_zip=imu_dir/name
        if original_zip.is_file(): compressed+=original_zip.stat().st_size
        member=None
        try:
            uncompressed+=source.uncompressed_size()
            members=source.members()
            if not members: raise ValueError("来源中没有TXT")
            for member in members:
                with source.open_binary(member) as stream: encoding=inspect_header(stream)
                encodings[encoding]+=1
        except (OSError,ValueError,UnicodeError,zipfile.BadZipFile) as exc:
            bad.append(dict(subject=subject,zip=name,member=member,source_kind=source.kind,error=str(exc)))
        if i%100==0: print(f"CHECK {i}/{len(selected)}",flush=True)
    report=dict(selected_zip_records_before_exclusions=original_count,selected_zip_records=len(selected),
        excluded_zip_records=len(excluded),excluded_sources=excluded,
        affected_meals=affected_meals(root,exclusions),subject_ids=len(ids),leakage_groups=len(group_folds),
        zip_directory=str(imu_dir),source_mode=source_mode,source_counts=dict(counts),text_encodings=dict(encodings),
        compressed_gb=round(compressed/1e9,3),uncompressed_gb=round(uncompressed/1e9,3),
        missing_zip_names=missing,bad_zip_headers=bad,
        dominant_meal_rows=len(rows(root/"三类筛选结果"/"惯用手用餐信息.csv")),
        non_dominant_meal_rows=len(rows(root/"三类筛选结果"/"非惯用手用餐信息.csv")))
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if missing or bad: raise SystemExit("数据检查未通过，见报告；新异常未自动排除，未开始训练")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--imu-dir",type=Path)
    p.add_argument("--source-mode",choices=("auto","directory","zip"),default="auto")
    p.add_argument("--output",type=Path,default=Path("runs/preflight.json"))
    a=p.parse_args(); check(a.root,a.imu_dir,a.output,a.source_mode)
