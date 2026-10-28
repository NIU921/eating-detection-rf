"""Competition interface: RF and ActionFormer event union for every subject.

Place ACC_test_data beside this program (or its Windows executable). The output
is results.csv beside the program. No hand/scenario label is required.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import torch


def base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


ROOT = base_dir()
REPO_ROOT = ROOT if getattr(sys, "frozen", False) else ROOT.parent
for package in (ROOT, REPO_ROOT, ROOT / "actionformer",
                ROOT / "actionformer" / "libs" / "utils"):
    sys.path.insert(0, str(package))

from input import DataReader, InfoReader
from output import save_result
from dominant_hand.config import Config
from dominant_hand.events import decode
from dominant_hand.features import FEATURE_NAMES, extract
from dominant_hand.inference_windows import window_indices
from dominant_hand.signal import process
from intake.features import extract_runs, feature_names
from libs.modeling import make_meta_arch

EXCLUDED = {"HNU21007", "HNU21026J", "HNU21030"}

def overlap(a, b):
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.0


def af_deduplicate(items, iou_limit=0.1):
    kept = []
    for item in sorted(items, key=lambda x: -x[2]):
        if all(overlap(item, old) <= iou_limit for old in kept):
            kept.append(item)
    return kept


def fuse_events(rf_events, af_events, iou_limit=0.25):
    # Fixed first version: preserve each detector's own threshold and decoding.
    # On overlap prefer RF's event boundary, then retain distinct AF events.
    chosen = [(int(a), int(b)) for a, b in rf_events]
    for a, b, score in sorted(af_events, key=lambda x: -x[2]):
        if all(overlap((a, b), old) < iou_limit for old in chosen):
            chosen.append((int(round(a)), int(round(b))))
    return sorted(set(chosen))


def build_af_sample(run):
    x = run["features"]
    duration = x.shape[1] * run["bin_ms"] / 1000.0
    return {"video_id": "inference", "feats": torch.from_numpy(np.ascontiguousarray(x)),
            "segments": torch.zeros((0, 2), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64), "fps": 1.0,
            "duration": duration, "feat_stride": run["bin_ms"] / 1000.0,
            "feat_num_frames": 0.0}


def predict_person(person_dir, subject, rf_bundle, af_model, af_settings, device):
    rf_cfg = Config(**rf_bundle["signal_config"])
    af_cfg = af_settings["preprocess"]
    mean = np.asarray(af_settings["normalization"]["mean"], np.float32)[:, None]
    std = np.asarray(af_settings["normalization"]["std"], np.float32)[:, None]
    rf_metadata, rf_features, rf_covered = [], [], []
    af_proposals = []
    failures = []
    reader = DataReader(person_dir)
    for file_id, text in reader.iter_data():
        try:
            segments, _ = process(io.StringIO(text), rf_cfg)
            for si, segment in enumerate(segments):
                x = segment["x"]
                for begin, end, start_ms, end_ms in window_indices(segment, rf_cfg, rf_covered):
                    rf_features.append(extract(x[begin:end], rf_cfg.fs))
                    rf_metadata.append(dict(externalid=subject, source_zip=file_id,
                                            member=file_id, segment=si,
                                            start_ms=start_ms, end_ms=end_ms))
        except (ValueError, UnicodeError) as exc:
            failures.append((file_id, "RF", str(exc)[:120]))
        try:
            for run in extract_runs(text, af_cfg, is_text=True):
                raw = run["features"]
                if raw.shape[0] != len(mean):
                    raise ValueError(f"AF channel mismatch: {raw.shape[0]} != {len(mean)}")
                run = dict(run)
                run["features"] = np.clip((raw - mean) / std, -20, 20)
                sample = build_af_sample(run)
                with torch.inference_mode():
                    output = af_model([sample])[0]
                for seg, score in zip(output["segments"].numpy(), output["scores"].numpy()):
                    a = max(run["valid_start_ms"], run["grid_start_ms"] + float(seg[0]) * 1000)
                    b = min(run["valid_end_ms"], run["grid_start_ms"] + float(seg[1]) * 1000)
                    if b > a:
                        af_proposals.append((a, b, float(score)))
        except (ValueError, UnicodeError) as exc:
            failures.append((file_id, "AF", str(exc)[:120]))
        del text
    if rf_features:
        probability = rf_bundle["model"].predict_proba(np.asarray(rf_features, np.float32))[:, 1]
        rf_events = decode(rf_metadata, probability,
                           threshold=rf_bundle["threshold"], **rf_bundle["decoder"]).get(subject, [])
    else:
        rf_events = []
    af_events = [x for x in af_deduplicate(af_proposals) if x[2] >= 0.60]
    merged = fuse_events(rf_events, af_events)
    return merged, {"rf_windows": len(rf_features), "rf_events": len(rf_events),
                    "af_candidates": len(af_proposals), "af_events": len(af_events),
                    "merged_events": len(merged), "failures": failures}


def main():
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    rf_bundle = joblib.load(REPO_ROOT / "models" / "rf_decoder_best.joblib")
    if rf_bundle["feature_names"] != FEATURE_NAMES:
        raise ValueError("RF release feature names do not match source")
    af_settings = json.loads((ROOT / "af_inference_config.json").read_text())
    if af_settings["feature_names"] != feature_names(af_settings["preprocess"]):
        raise ValueError("AF feature order does not match training cache")
    checkpoint = torch.load(ROOT / "af_weights" / "best.pt", map_location="cpu", weights_only=False)
    af_model = make_meta_arch(checkpoint["config"]["model_name"],
                              **checkpoint["config"]["model"]).to("cpu").eval()
    af_model.load_state_dict(checkpoint["model"])
    working = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else ROOT
    data_root = working / "ACC_test_data"
    info = InfoReader(data_root / "userinfobean.txt")
    people = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.upper().startswith("HNU"))
    output = working / "results.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerow(("externalid", "beforeTime", "afterTime"))
    for person_dir in people:
        subject = "".join(person_dir.name.split()).upper()
        if subject in EXCLUDED:
            print("skip excluded subject", subject, flush=True)
            continue
        info.read_info(subject)  # Required official user-info interface; no hand label exists.
        events, audit = predict_person(person_dir, subject, rf_bundle, af_model,
                                       af_settings, torch.device("cpu"))
        for a, b in events:
            save_result(subject, a, b, 1, output_path=output)
        print(subject, audit, flush=True)
    print("result", output, flush=True)


if __name__ == "__main__":
    main()
