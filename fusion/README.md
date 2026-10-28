# 双模型进食检测融合

此目录将原仓库的惯用手 RF 与 40 人全量训练的 ActionFormer 一起推理。测试数据无需提供手别标签。RF 事件优先；ActionFormer 置信度 >= 0.60 的事件在与 RF 事件 IoU < 0.25 时补充输出。最终事件匹配遵循赛题的严格 IoU > 0.25。

## 准备与运行

使用 Python 3.12，在仓库根目录运行：

```bash
python -m pip install -r fusion/requirements.txt
```

先按照原仓库 README，从上游 Release 下载 `rf_decoder_best.joblib` 到根目录的 `models/`。ActionFormer 权重已经包含在 `fusion/af_weights/best.pt`。将赛题格式的 `ACC_test_data/` 放在 `fusion/` 下（内含 `userinfobean.txt` 和每个人的目录），然后运行：

```bash
python fusion/merged_predict.py
```

输出为 `fusion/results.csv`，列名 `externalid,beforeTime,afterTime`。`input.py` 和 `output.py` 是组委会接口文件。当前源码已在 macOS 上完成一次推理检查，Windows EXE 尚未构建或验证。

## 模型与结果边界

- RF 采用原仓库发布权重与原事件解码；模型文件仍由上游 Release 提供。
- ActionFormer 使用 67 通道、5 秒网格特征和 `intake_hnm_full40/epoch28.pt`。40 人的全部正样本记录参与训练，负样本沿用困难负样本回采。训练轮次 28 和阈值 0.60 在全量重训前由原 30 人训练实验固定。权重 SHA256：`17f620f550e82b40b43201d6aa068bf564e81adc6f21604ca32c09f841cf1284`。
- 全量模型没有独立验证集。旧 5 人留出集的融合 F1 0.571 来自旧 ActionFormer 权重，不能当作本权重的验证成绩或官方测试成绩。
- `nms_1d_cpu.py` 是为跨平台推理使用的纯 Python 后备实现。ActionFormer 源码来自其原项目，相关许可证见 `ACTIONFORMER_LICENSE`。
