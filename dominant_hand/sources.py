"""统一读取同名解压目录或ZIP；不将设备二进制内容猜测为文本编码。"""
import codecs
from contextlib import contextmanager
from dataclasses import dataclass
import io
import json
from pathlib import Path
import zipfile


REQUIRED_COLUMNS={"ACC_TIME","GYRO_TIME"}|{f"{m}_{a}" for m in ("ACC","GYRO") for a in "XYZ"}


class UnsupportedSensorFormat(ValueError):
    pass


def inspect_header(raw):
    prefix=raw.read(65536)
    if prefix.startswith((codecs.BOM_UTF32_LE,codecs.BOM_UTF32_BE)):
        encoding="utf-32"
    elif prefix.startswith((codecs.BOM_UTF16_LE,codecs.BOM_UTF16_BE)):
        encoding="utf-16"
    else:
        encoding="utf-8-sig"
    try:
        text=codecs.getincrementaldecoder(encoding)(errors="strict").decode(prefix,final=False)
    except UnicodeError as exc:
        raise UnsupportedSensorFormat(
            "不是可直接读取的IMU文本字段表，疑似尚未解析的设备二进制；"
            f"需要配套解码工具，单纯解压无效。前16字节: {prefix[:16].hex(' ')}") from exc
    header=text.splitlines()[0] if text.splitlines() else ""
    if not REQUIRED_COLUMNS<=set(header.split()):
        if "\x00" in text or any(ord(c)<32 and c not in "\r\n\t" for c in header):
            raise UnsupportedSensorFormat(
                "成员含二进制控制字节且无IMU表头，需要配套工具解码。"
                f"前16字节: {prefix[:16].hex(' ')}")
        raise UnsupportedSensorFormat(f"缺少IMU字段: {sorted(REQUIRED_COLUMNS-set(header.split()))}")
    return encoding


@contextmanager
def sensor_text(raw):
    encoding=inspect_header(raw)
    raw.seek(0)
    with io.TextIOWrapper(raw,encoding=encoding,errors="strict",newline="") as stream:
        yield stream


@dataclass(frozen=True)
class SensorSource:
    path: Path
    kind: str

    def members(self):
        if self.kind=="directory":
            return sorted(p.relative_to(self.path).as_posix() for p in self.path.rglob("*")
                          if p.is_file() and p.suffix.lower()==".txt")
        with zipfile.ZipFile(self.path) as archive:
            return sorted(n for n in archive.namelist() if n.lower().endswith(".txt"))

    @contextmanager
    def open_binary(self,member):
        if self.kind=="directory":
            with (self.path/member).open("rb") as stream: yield stream
        else:
            with zipfile.ZipFile(self.path) as archive:
                with archive.open(member) as stream: yield stream

    @contextmanager
    def open_text(self,member):
        with self.open_binary(member) as raw:
            with sensor_text(raw) as stream: yield stream

    def fingerprint(self):
        if self.kind=="directory":
            members=[]
            for member in self.members():
                stat=(self.path/member).stat()
                members.append(dict(member=member,size=stat.st_size,mtime_ns=stat.st_mtime_ns))
            return dict(kind=self.kind,members=members)
        stat=self.path.stat()
        return dict(kind=self.kind,size=stat.st_size,mtime_ns=stat.st_mtime_ns)

    def uncompressed_size(self):
        if self.kind=="directory":
            return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())
        with zipfile.ZipFile(self.path) as archive:
            return sum(info.file_size for info in archive.infolist())


def resolve_source(imu_dir,name,mode="auto"):
    root=Path(imu_dir)
    if mode not in ("auto","directory","zip"): raise ValueError("来源模式应为auto/directory/zip")
    if mode in ("auto","directory"):
        for directory in (root/Path(name).stem,root/name):
            if directory.is_dir(): return SensorSource(directory,"directory")
    if mode in ("auto","zip") and (root/name).is_file(): return SensorSource(root/name,"zip")
    return None


def read_exclusions(root):
    path=Path(root)/"experiments"/"excluded_sources.json"
    if not path.exists(): return {}
    payload=json.loads(path.read_text(encoding="utf-8-sig"))
    return {r["zip"]:r for r in payload["sources"]}


def affected_meals(root,excluded_names):
    from split_meal_datasets import load, sid
    excluded=set(excluded_names); result=[]
    for kind,name in (("dominant","惯用手用餐信息.csv"),("non_dominant","非惯用手用餐信息.csv")):
        for row,_ in load(Path(root)/"三类筛选结果"/name,["externalid"])["records"]:
            if not row: continue
            names=set(json.loads(row.get("sensorData") or "[]"))
            if names & excluded:
                result.append(dict(externalid=sid(row["externalid"]),kind=kind,
                    start_ms=int(row["timeFrame.startTime"]),end_ms=int(row["timeFrame.endTime"]),
                    all_sources_excluded=names<=excluded,excluded_zip_names=sorted(names & excluded)))
    return result
