"""所有餐次与完整文件时间轴相交标注；文件角色绝不直接等于窗口标签。"""
from pathlib import Path
import csv
from split_meal_datasets import load, sid


def merged(intervals):
    result = []
    for start, end in sorted((int(a), int(b)) for a, b in intervals if b > a):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def overlap(start, end, intervals):
    return sum(max(0, min(end, b) - max(start, a)) for a, b in merged(intervals))


def subtract(intervals, removed):
    result = []
    for start, end in merged(intervals):
        cursor = start
        for a, b in merged(removed):
            if b <= cursor:
                continue
            if a >= end:
                break
            if a > cursor:
                result.append((cursor, min(a, end)))
            cursor = max(cursor, b)
        if cursor < end:
            result.append((cursor, end))
    return result


def intersections(left, right):
    return merged((max(a,c), min(b,d)) for a,b in left for c,d in right if max(a,c) < min(b,d))


def read_catalog(root):
    catalog = {}
    for dominant, filename in [(True, "惯用手用餐信息.csv"), (False, "非惯用手用餐信息.csv")]:
        rows = load(Path(root)/"三类筛选结果"/filename, ["externalid"])["records"]
        for row, _ in rows:
            if not row:
                continue
            subject = sid(row["externalid"])
            start, end = int(row["timeFrame.startTime"]), int(row["timeFrame.endTime"])
            if end <= start:
                raise ValueError(f"餐次起止无效: {subject}")
            catalog.setdefault(subject, []).append(dict(start=start, end=end, dominant=dominant))
    return catalog


def read_verified(path):
    if path is None:
        return None
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            a, b = int(row["start_ms"]), int(row["end_ms"])
            if b <= a:
                raise ValueError("日志完整覆盖区间起止无效")
            result.setdefault(sid(row["externalid"]), []).append((a,b))
    return {k: merged(v) for k,v in result.items()}


def window_label(start, end, meals, cfg, negative_intervals=None):
    """None表示用户已确认全部餐外时间；显式区间列表表示仅确认这些区间。"""
    if end <= start:
        raise ValueError("窗口起止无效")
    positive = [(m["start"], m["end"]) for m in meals if m["dominant"]]
    ignored = [(m["start"], m["end"]) for m in meals if not m["dominant"]]
    if overlap(start, end, ignored):
        return -1, "non_dominant_or_conflict"
    fraction = overlap(start, end, positive) / (end-start)
    if fraction >= cfg.positive_overlap:
        return 1, "dominant_meal"
    all_meals = positive + ignored
    if overlap(start, end, all_meals):
        return -1, "meal_transition"
    buffer = [(a-cfg.boundary_s*1000, b+cfg.boundary_s*1000) for a,b in all_meals]
    if overlap(start, end, buffer):
        return -1, "boundary_buffer"
    if negative_intervals is None or overlap(start, end, negative_intervals) >= end-start:
        return 0, "confirmed_background"
    return -1, "unverified_background"
