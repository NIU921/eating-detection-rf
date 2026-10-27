"""按餐次与传感器文件时间重叠，生成惯用手、非惯用手、非进食三个 CSV。

仅标准库。原始数据不修改，不解压 ZIP。用餐起止字段是 timeFrame，图片时间不参与。
"""
import csv
import argparse
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MEALS_CSV = ROOT / "用餐信息.csv"
SENSORS_CSV = ROOT / "IMU+PPG.csv"
IMU_DIR = ROOT / "IMU+PPG"
OUTPUT_DIR = ROOT / "三类筛选结果_每餐一行"
INVALID_IDS = {"HNU21007", "HNU21026j", "HNU21030", "AC3", "test"}
MEAL_COLUMNS = ["externalid", "sensorData", "dietaryType", "timeFrame.startTime",
                "timeFrame.endTime", "timeFrame.timestamp"]
# 用户列出的实际为六列：externalid、sensorData、dietaryType、三个 timeFrame 字段。
TZ = timezone(timedelta(hours=8))


def sid(value):
    return "".join(str(value or "").split()).upper()


def hand(value):
    return str(value or "").replace("佩戴", "").replace("佩带", "").strip()


def zip_name(value):
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def shorten_sensor_text(row, original, column):
    """仅替换 sensorData 的 CSV 单元格，保留其他单元格及原始换行/引号。"""
    spans = []
    start = i = 0
    quoted = False
    while i < len(original):
        char = original[i]
        if char == '"':
            if quoted and i+1 < len(original) and original[i+1] == '"':
                i += 2
                continue
            quoted = not quoted
        elif not quoted and char == ',':
            spans.append((start, i))
            start = i+1
        elif not quoted and char in '\r\n':
            break
        i += 1
    spans.append((start, i))
    left, right = spans[column]
    old = original[left:right]
    new = zip_name(row["sensorData"])
    if old.startswith('"') and old.endswith('"'):
        new = '"' + new.replace('"', '""') + '"'
    elif any(char in new for char in ',"\r\n'):
        new = '"' + new.replace('"', '""') + '"'
    return original[:left] + new + original[right:]


def load(path, required):
    raw = path.read_bytes()
    encodings = ["utf-8-sig"] if raw.startswith(b"\xef\xbb\xbf") else ["utf-8", "gb18030"]
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("无法识别编码：%s" % path)
    lines = io.StringIO(text, newline="").readlines()
    reader = csv.reader(lines)
    fields = next(reader)
    if len(fields) != len(set(fields)) or not set(required).issubset(fields):
        raise ValueError("表头缺失字段或有重复字段：%s" % path)
    previous = reader.line_num
    header = "".join(lines[:previous])
    records = []
    for values in reader:
        original = "".join(lines[previous:reader.line_num])
        previous = reader.line_num
        if not values:
            records.append((None, original))
            continue
        if len(values) != len(fields):
            raise ValueError("CSV 列数不一致：%s 物理行 %d" % (path, reader.line_num))
        records.append((dict(zip(fields, values)), original))
    return dict(fields=fields, records=records, header=header, encoding=encoding)


def meal_time(value):
    number = Decimal(str(value).strip())
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("用餐时间不是有限整数毫秒")
    result = int(number)
    if len(str(result)) != 13:
        raise ValueError("用餐时间须为13位毫秒时间戳")
    return result


def sensor_time(value):
    dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return int(dt.timestamp()*1000)


def intersects(start, end, a, b):
    return max(start, a) < min(end, b)


def build(meals, sensors):
    invalid = {sid(x) for x in INVALID_IDS}
    intervals = defaultdict(list)
    counts = Counter()
    retained_meals = []
    for row, _ in meals["records"]:
        if row is None:
            continue
        person = sid(row["externalid"])
        if not person or person in invalid:
            counts["excluded_meal_rows"] += 1
            continue
        start = meal_time(row["timeFrame.startTime"])
        end = meal_time(row["timeFrame.endTime"])
        if end <= start:
            raise ValueError("发现无效用餐区间，停止而不误标负例：%s" % row.get("uniqueid"))
        dh, wh = hand(row["dietaryHand"]), hand(row["wearHand"])
        if dh not in ("左手", "右手") or wh not in ("左手", "右手"):
            raise ValueError("餐次佩戴手或惯用手不明确：%s" % row.get("uniqueid"))
        meal = dict(row=row, start=start, end=end, dominant=dh==wh, matched=0, zip_files={})
        intervals[person].append(meal)
        retained_meals.append(meal)

    dominant, non_dominant = [], []
    non_eating_indices = set()
    selected_zip_names = set()
    mixed = []
    eating_file_ids = set()
    for index, (row, _) in enumerate(sensors["records"]):
        if row is None:
            continue
        person = sid(row["externalid"])
        if person in invalid:
            counts["excluded_sensor_invalid_id"] += 1
            continue
        if not person:
            counts["excluded_sensor_missing_id"] += 1
            continue
        start = sensor_time(row["timeStamp.startTime"])
        last = sensor_time(row["timeStamp.endTime"])
        if last < start:
            raise ValueError("采集文件起止时间倒置：%s" % row.get("uniqueid"))
        # 索引末秒也可能包含采样，保守纳入末秒，减少边界信号被误分到非进食。
        end = last + (1000 if last % 1000 == 0 else 1)
        overlapping = [m for m in intervals[person]
                       if intersects(start, end, m["start"], m["end"])]
        selected_zip_names.add(zip_name(row["sensorData"]))
        if not overlapping:
            non_eating_indices.add(index)
            continue
        eating_file_ids.add(index)
        if {m["dominant"] for m in overlapping} == {True, False}:
            mixed.append({"externalid": row["externalid"], "sensorData": zip_name(row["sensorData"])})
        for meal in overlapping:
            name = zip_name(row["sensorData"])
            meal["zip_files"][name] = min(start, meal["zip_files"].get(name, start))

    # 每条原始有效餐次仅输出一次，保留源餐次顺序和原始时间。
    # 一餐的全部 ZIP 去重后按采集时间排序；无匹配时保留空数组。
    for meal in retained_meals:
        original = meal["row"]
        names = sorted(meal["zip_files"], key=lambda name: (meal["zip_files"][name], name))
        output = {name: original.get(name, "") for name in MEAL_COLUMNS}
        output["sensorData"] = json.dumps(names, ensure_ascii=False)
        (dominant if meal["dominant"] else non_dominant).append(output)
        meal["matched"] = len(names)

    unmatched = [m["row"].get("uniqueid", "") for m in retained_meals if not m["matched"]]
    missing = sorted(name for name in selected_zip_names if not (IMU_DIR/name).is_file())
    report = dict(
        excluded_ids=sorted(INVALID_IDS),
        valid_meal_events=len(retained_meals),
        dominant_meal_events=sum(m["dominant"] for m in retained_meals),
        non_dominant_meal_events=sum(not m["dominant"] for m in retained_meals),
        dominant_output_rows=len(dominant), non_dominant_output_rows=len(non_dominant),
        non_eating_file_rows=len(non_eating_indices), eating_file_rows=len(eating_file_ids),
        exclusions=dict(counts), unmatched_meal_uniqueids=unmatched,
        meal_zip_match_counts=[dict(externalid=m["row"]["externalid"],
                                    meal_uniqueid=m["row"].get("uniqueid", ""),
                                    start_ms=m["start"], end_ms=m["end"],
                                    zip_count=m["matched"], dominant=m["dominant"])
                               for m in retained_meals],
        mixed_scene_files=mixed, missing_zip_names=missing,
        meal_columns=MEAL_COLUMNS,
        notes=["按 externalid 与时间重叠关联；不是两表 uniqueid 相等匹配。",
               "前两个CSV每餐仅一行，sensorData是所有匹配ZIP文件名的JSON数组，去重后按采集时间排序。",
               "餐次起止保留原值；单ZIP也使用数组；无匹配为[]，不丢弃餐次。",
               "第三个CSV的sensorData仍为单ZIP文件名；其他列、值、顺序、编码和文本不变。",
               "非进食只表示与已记录用餐无重叠，未证明日志完整性。",
               "同一文件可同时包含两类餐次，会出现在前两表，后续须按时间裁剪。",
               "未匹配到信号的餐次sensorData为[]，唯一ID也记录在本报告。",
               "没有解压、复制或清洗ZIP，未额外按信号格式筛除文件。"])
    return dominant, non_dominant, non_eating_indices, report


def write_meals(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MEAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(overwrite=False):
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not overwrite:
        raise FileExistsError("输出目录非空，请修改 OUTPUT_DIR：%s" % OUTPUT_DIR)
    meals = load(MEALS_CSV, {"externalid", "dietaryHand", "wearHand",
                             "timeFrame.startTime", "timeFrame.endTime", "dietaryType"})
    sensors = load(SENSORS_CSV, {"externalid", "sensorData", "timeStamp.startTime", "timeStamp.endTime"})
    dominant, non_dominant, non_eating, report = build(meals, sensors)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_meals(OUTPUT_DIR/"惯用手用餐信息.csv", dominant)
    write_meals(OUTPUT_DIR/"非惯用手用餐信息.csv", non_dominant)
    sensor_column = sensors["fields"].index("sensorData")
    retained_text = sensors["header"] + "".join(
        original if row is None else shorten_sensor_text(row, original, sensor_column)
        for index, (row, original) in enumerate(sensors["records"])
        if row is None or index in non_eating)
    with (OUTPUT_DIR/"非进食数据_IMU+PPG.csv").open("wb") as stream:
        stream.write(retained_text.encode(sensors["encoding"]))
    (OUTPUT_DIR/"筛选说明.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "dominant_output_rows", "non_dominant_output_rows", "non_eating_file_rows",
        "eating_file_rows", "exclusions")}, ensure_ascii=False, indent=2))
    print("输出目录：%s" % OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="更新已有的三个结果CSV及筛选说明")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="结果输出目录")
    arguments = parser.parse_args()
    OUTPUT_DIR = arguments.output_dir.resolve()
    main(overwrite=arguments.overwrite)
