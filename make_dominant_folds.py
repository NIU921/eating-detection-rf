"""生成惯用手专项的身份隔离五折；仅标准库，不读取或修改原始信号。"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from split_meal_datasets import load, sid

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "dominant": "惯用手用餐信息.csv",
    "non_dominant_ignore": "非惯用手用餐信息.csv",
    "candidate_negative": "非进食数据_IMU+PPG.csv",
}


def records(path):
    return [r for r, _ in load(path, ["externalid"])["records"] if r]


def build(root, out, seed=20261021, trials=5000):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    # 相同 healthid 的不同编号仅做保守防泄漏关联，不据此修订标签。
    health_ids = defaultdict(set)
    for name in ("用户信息.csv", "用餐信息.csv", "IMU+PPG.csv"):
        for row in records(root / name):
            subject, health = sid(row["externalid"]), row.get("healthid", "").strip()
            if subject and health:
                health_ids[health].add(subject)
    for subjects in health_ids.values():
        ordered = sorted(subjects)
        for subject in ordered[1:]:
            union(ordered[0], subject)

    data = {role: records(root / "三类筛选结果" / name)
            for role, name in SOURCES.items()}
    subjects = sorted({sid(r["externalid"]) for rows in data.values() for r in rows})
    groups = defaultdict(list)
    for subject in subjects:
        groups[find(subject)].append(subject)
    # 稳定组名不暴露 healthid，且不假定代表编号一定是正确身份。
    representative = {s: min(members) for members in groups.values() for s in members}
    vectors = {min(m): [0.0] * 4 for m in groups.values()}
    for role, rows in data.items():
        for row in rows:
            v = vectors[representative[sid(row["externalid"])]]
            if role == "dominant":
                v[0] += 1
                v[1] += (int(row["timeFrame.endTime"]) - int(row["timeFrame.startTime"])) / 1000
            elif role == "candidate_negative":
                v[2] += 1  # 文件行数只是预划分代理，不是有效负例时长。
    for members in groups.values():
        vectors[min(members)][3] = len(members)
    keys = sorted(vectors)
    if len(keys) < 5:
        raise ValueError("独立身份组少于5，无法五折")
    target = [sum(v[j] for v in vectors.values()) / 5 for j in range(4)]
    rng = random.Random(seed)
    best = None
    # 固定各折身份组数，预划分仅平衡元数据，不用预测成绩挑选划分。
    for _ in range(trials):
        order = keys[:]
        rng.shuffle(order)
        assignment = {key: i % 5 for i, key in enumerate(order)}
        totals = [[0.0] * 4 for _ in range(5)]
        for key, fold in assignment.items():
            for j, value in enumerate(vectors[key]):
                totals[fold][j] += value
        if any(v[0] == 0 or v[2] == 0 for v in totals):
            continue
        score = sum(((v[j] - target[j]) / max(target[j], 1)) ** 2
                    for v in totals for j in range(4))
        if best is None or score < best[0]:
            best = score, assignment, totals
    if best is None:
        raise ValueError("未生成各折均有正例和候选负例的划分")
    _, assignment, totals = best
    subject_fold = {s: assignment[representative[s]] + 1 for s in subjects}
    assert set(subject_fold.values()) == {1, 2, 3, 4, 5}
    for linked in health_ids.values():
        assert len({subject_fold[s] for s in linked if s in subject_fold}) <= 1
    out.mkdir(parents=True, exist_ok=True)

    def write(name, fields, rows):
        with (out / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write("subject_folds.csv", ["externalid", "leakage_group", "fold"],
          [{"externalid": s, "leakage_group": representative[s], "fold": subject_fold[s]}
           for s in subjects])
    manifest = []
    audit = []
    for role, rows in data.items():
        for index, row in enumerate(rows, 2):
            s = sid(row["externalid"])
            manifest.append(dict(role=role, source=SOURCES[role], csv_record=index,
                                 externalid=s, leakage_group=representative[s], fold=subject_fold[s]))
            if role == "dominant":
                duration = (int(row["timeFrame.endTime"]) - int(row["timeFrame.startTime"])) / 1000
                zips = json.loads(row["sensorData"])
                if duration < 10 or not zips:
                    audit.append(dict(externalid=s, csv_record=index, duration_s=duration,
                                      zip_count=len(zips)))
    write("record_folds.csv", list(manifest[0]), manifest)
    summary = dict(seed=seed, trials=trials, subject_ids=len(subjects),
                   leakage_groups=len(keys), source_rows={k: len(v) for k, v in data.items()},
                   linked_ids=[sorted(m) for m in groups.values() if len(m) > 1],
                   dominant_metadata_flags=audit,
                   notes=["fold是外层测试折编号；其余四折仅供训练和内部验证。",
                          "2026-10-21用户确认非进食文件及含餐文件的餐外时间均无进食；6C7等异常身份仍须核查。",
                          "同healthid关联仅用于保守防泄漏，不证明为同一真实被试。",
                          "统计基于索引，尚未按真实信号覆盖过滤；不是训练结果。"],
                   folds=[])
    for fold in range(1, 6):
        test = sorted(s for s in subjects if subject_fold[s] == fold)
        train = sorted(set(subjects) - set(test))
        assert not set(train) & set(test)
        item = dict(fold=fold, test_subject_ids=test, outer_train_subject_ids=train,
                    leakage_groups=sum(f + 1 == fold for f in assignment.values()),
                    dominant_meals=int(totals[fold-1][0]),
                    dominant_duration_min=round(totals[fold-1][1] / 60, 2),
                    candidate_negative_rows=int(totals[fold-1][2]))
        summary["folds"].append(item)
    (out / "fold_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "惯用手算法设计" / "五折划分")
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    result = build(args.root, args.output, args.seed)
    print(json.dumps({k: result[k] for k in ("subject_ids", "leakage_groups", "linked_ids", "dominant_metadata_flags")}, ensure_ascii=False))
    for row in result["folds"]:
        print(json.dumps({k: v for k, v in row.items() if not isinstance(v, list)}, ensure_ascii=False))
