"""按模型保存的信号配置处理新被试TXT，输出赛题格式CSV。"""
import argparse
import csv
from pathlib import Path
import joblib
import numpy as np
from split_meal_datasets import sid
from .config import Config
from .events import decode
from .features import extract, FEATURE_NAMES
from .signal import process
from .sources import sensor_text
from .inference_windows import window_indices


def predict_folder(model_path, folder, externalid, output_csv):
    bundle=joblib.load(model_path)
    if bundle["feature_names"]!=FEATURE_NAMES:
        raise ValueError("模型特征定义与当前代码不一致")
    cfg=Config(**bundle["signal_config"]); folder=Path(folder)
    if not folder.is_dir(): raise FileNotFoundError(folder)
    if externalid:
        subjects=[(sid(externalid),folder)]
    elif list(folder.glob("*.txt")) and any(p.name.startswith("collect") for p in folder.glob("*.txt")):
        subjects=[(sid(folder.name),folder)]
    else:
        subjects=[(sid(p.name),p) for p in sorted(folder.iterdir()) if p.is_dir() and list(p.glob("*.txt"))]
    if not subjects: raise ValueError("未找到包含TXT的个人目录")
    output_csv=Path(output_csv); output_csv.parent.mkdir(parents=True,exist_ok=True)
    total_windows=total_events=0; no_signal=[]
    with output_csv.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream); writer.writerow(("externalid","beforeTime","afterTime"))
        for subject,person_dir in subjects:
            metadata=[]; features=[]; covered=[]
            paths=sorted(p for p in person_dir.glob("*.txt") if p.name!="userinfobean.txt")
            if not paths: raise ValueError(f"没有传感器TXT: {person_dir}")
            for path in paths:
                with path.open("rb") as raw:
                    with sensor_text(raw) as text: segments,_=process(text,cfg)
                for si,segment in enumerate(segments):
                    t,x,observed=segment["t"],segment["x"],segment["observed"]
                    for begin,end,start_ms,end_ms in window_indices(segment,cfg,covered):
                        features.append(extract(x[begin:end],cfg.fs))
                        metadata.append(dict(externalid=subject,source_zip=path.name,member=path.name,
                            segment=si,start_ms=start_ms,end_ms=end_ms))
            total_windows+=len(features)
            if not features:
                no_signal.append(subject); print(f'WARNING subject={subject} 无有效六轴窗口，空预测不能解释为确定没有进食',flush=True)
            if features:
                probability=bundle["model"].predict_proba(np.asarray(features,dtype=np.float32))[:,1]
                events=decode(metadata,probability,threshold=bundle["threshold"],**bundle["decoder"])
                for a,b in events.get(subject,[]):
                    if not (len(str(a))==len(str(b))==13 and a<b): raise ValueError("输出必须为13位毫秒时间戳")
                    writer.writerow((subject,a,b)); total_events+=1
            print(f"PREDICT subject={subject} windows={len(features)}",flush=True)
    return dict(windows=total_windows,events=total_events,output=str(output_csv),subjects_without_valid_windows=no_signal)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path); p.add_argument("externalid",nargs="?")
    p.add_argument("--model",type=Path,default=Path("runs/baseline/training/final_model.joblib"))
    p.add_argument("--output",type=Path,default=Path("results.csv"))
    a=p.parse_args(); print(predict_folder(a.model,a.folder,a.externalid,a.output))


if __name__=="__main__": main()
