# 惯用手进食检测：RF＋解码调优

已完成实验中事件级五折汇总 F1 最高的随机森林版本：**57.27%**。
实验标识：`smooth120-180-240_stop12-24-36_gap30-60-120_thr0p35-0p45-0p55-0p65_start3_min60_5af8cb3bdb`。

## 结果与适用范围

| 指标 | 数值 |
|---|---:|
| 合并外折事件 F1 | 0.572687 |
| Precision | 0.560345 |
| Recall | 0.585586 |
| TP / FP / FN | 65 / 51 / 46 |
| 五折 F1 均值 / 标准差 | 0.569691 / 0.148549 |
| 匹配事件开始 / 结束 MAE | 226.39 / 180.98 秒 |

任务为惯用手进食检测，非惯用手餐次按原标签规则忽略。事件按 IoU > 0.25 一对一匹配。
森林沿用原分组内折选择，仅在内折概率上优化解码，外折概率不参与参数选择。
这是查看过基线外折结果之后的开发迭代；57.27% 为开发五折结果，独立测试泛化尚未验证。
该成绩不属于“所有标注餐次均为正例”的统一进食留出实验。

## 安装和推理

建议 Python 3.12（训练记录为 3.12.14），在仓库根目录执行：

```sh
python -m pip install -r requirements.txt
python predict.py /path/to/subject HNUxxxxx --output results.csv
```

先从[版本发布附件](https://github.com/NIU921/eating-detection-rf/releases/tag/v1.0.0-rf-decoder-best)下载 `rf_decoder_best.joblib`，放到 `models/`。
运行 `python verify_model.py` 核对 SHA256 和模型设置。
模型为已保存的原森林与更新后解码参数，未重新训练。仅加载来源可信的 joblib 文件。
输入为个人目录下的传感器 `collect*.txt`；多个人目录可以省略第二个位置参数。
输出为 UTF-8 BOM CSV：`externalid,beforeTime,afterTime`，起止时间为13位毫秒时间戳。
模型使用六轴 IMU、25 Hz、10秒窗、5秒步长及153维特征。

部署解码参数为各折内折所选值中位数：平滑240秒，开始阈值0.45，
开始连续3窗，结束连续36窗，合并间隔30秒，最短事件60秒。
部署模型使用全量可用训练数据；五折分数来自各外折模型，不能理解为最终模型的独立测试分数。

## 从原始数据复现实验

仓库提供源代码、准确的候选配置与汇总报告。原始传感器数据、身份与餐次明细、
五折划分以及大型特征/概率缓存未发布，完整复现需要原项目的这些数据文件。

将原项目的 `IMU+PPG/`、`IMU+PPG.csv`、`用餐信息.csv`、`用户信息.csv`、
`三类筛选结果/`、`惯用手算法设计/五折划分/` 放到根目录后执行：

```sh
python run_local.py --stage all --jobs 4
python run_decoder_tuning.py --config experiments/decoder_search.json
```

已有原基线 `runs/local_baseline/` 时只需第二条命令，可复用内外折概率。
`runs/code_review_20261022/source_before/` 保留兼容性核验需要的原代码快照。
重新训练所用版本、平台和并行实现可能造成数值差异。

## 验证

```sh
python -m unittest discover -s tests -v
python verify_model.py
```

`reports/training_report.json` 和 `outer_fold_metrics.csv` 为原实验汇总；
`reports/inner_search_fold_*.csv` 为内折搜索证据；`reports/verification.json` 为本次整理验证记录。
