import codecs
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from dominant_hand.sources import inspect_header, sensor_text, resolve_source, UnsupportedSensorFormat
from dominant_hand.check_data import check


HEADER="ACC_TIME GYRO_TIME ACC_X ACC_Y ACC_Z GYRO_X GYRO_Y GYRO_Z\n"
TEXT=HEADER+"1784600000000 1784600000000 1 2 3 4 5 6\n"


class SourceTests(unittest.TestCase):
    def test_prefer_extracted_directory_then_fallback_to_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); name="source.zip"
            with zipfile.ZipFile(root/name,"w") as archive: archive.writestr("data.txt",TEXT)
            source=resolve_source(root,name); self.assertEqual(source.kind,"zip")
            with source.open_text("data.txt") as text: self.assertEqual(text.readline().split(),HEADER.split())
            folder=root/"source"; folder.mkdir(); (folder/"data.txt").write_text(TEXT,encoding="utf-8")
            self.assertEqual(resolve_source(root,name).kind,"directory")
            (root/name).unlink()
            source=resolve_source(root,name); self.assertEqual(source.members(),["data.txt"])
            with source.open_text("data.txt") as text: self.assertEqual(text.readline().split(),HEADER.split())

    def test_real_utf16_bom_text_is_supported(self):
        raw=io.BytesIO(TEXT.encode("utf-16"))
        self.assertEqual(inspect_header(raw),"utf-16"); raw.seek(0)
        with sensor_text(raw) as text: self.assertEqual(text.readline(),HEADER)

    def test_binary_is_not_silently_decoded(self):
        with self.assertRaises(UnsupportedSensorFormat):
            inspect_header(io.BytesIO(b"\x00\xff\xfb\xaa"+b"\x00"*256))

    def test_approved_exclusion_is_recorded_but_other_sources_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); imu=root/"IMU+PPG"; imu.mkdir()
            for name,content in (("valid",TEXT.encode()),("bad",b"\x00\xff"*100)):
                folder=imu/name; folder.mkdir(); (folder/"data.txt").write_bytes(content)
            def write(path,fields,rows):
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open("w",encoding="utf-8",newline="") as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            write(root/"IMU+PPG.csv",["externalid","sensorData"],
                  [dict(externalid="A",sensorData="valid.zip"),dict(externalid="A",sensorData="bad.zip")])
            write(root/"惯用手算法设计/五折划分/subject_folds.csv",["externalid","fold","leakage_group"],
                  [dict(externalid="A",fold=1,leakage_group="A")])
            fields=["externalid","sensorData","timeFrame.startTime","timeFrame.endTime"]
            write(root/"三类筛选结果/惯用手用餐信息.csv",fields,[{"externalid":"A","sensorData":'["bad.zip"]',
                  "timeFrame.startTime":1784600000000,"timeFrame.endTime":1784600060000}])
            write(root/"三类筛选结果/非惯用手用餐信息.csv",fields,[])
            (root/"experiments").mkdir()
            exclusion=root/"experiments/excluded_sources.json"
            exclusion.write_text(json.dumps({"sources":[{"zip":"bad.zip","reason":"user_excluded"}]}),encoding="utf-8")
            result=check(root,imu,root/"report.json")
            self.assertEqual(result["selected_zip_records"],1)
            self.assertEqual(result["source_counts"],{"directory":1})
            self.assertEqual(result["bad_zip_headers"],[])
            self.assertEqual(result["excluded_zip_records"],1)
            self.assertTrue(result["affected_meals"][0]["all_sources_excluded"])
            exclusion.write_text('{"sources":[]}',encoding="utf-8")
            with self.assertRaises(SystemExit): check(root,imu,root/"report.json")


if __name__=="__main__": unittest.main()
