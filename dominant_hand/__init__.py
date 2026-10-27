"""惯用手进食检测：数据准备、嵌套交叉验证、训练和推理。"""
import sys
from pathlib import Path

# 可选的项目局部依赖；不修改全局Python环境。
_local = Path(__file__).resolve().parents[1] / ".python_packages"
if _local.is_dir():
    sys.path.insert(0, str(_local))

VERSION = "1.0.0"
