"""校验本版本的模型文件及部署参数。"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


def verify(path):
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'reports/release_manifest.json').read_text(encoding='utf-8'))
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    if digest != manifest['release_model_sha256']:
        raise ValueError('模型 SHA256 与版本清单不符')
    bundle = joblib.load(path)
    from dominant_hand.features import FEATURE_NAMES
    if bundle['feature_names'] != FEATURE_NAMES:
        raise ValueError('特征定义不符')
    expected = manifest['decoder']
    actual = dict(bundle['decoder'])
    if bundle['threshold'] != expected['threshold']:
        raise ValueError('阈值不符')
    if actual['smooth_windows'] != round(expected['smooth_s'] / bundle['signal_config']['stride_s']):
        raise ValueError('平滑窗数不符')
    for key in ('start_persistence', 'stop_persistence', 'merge_gap_s', 'min_event_s'):
        if actual[key] != expected[key]:
            raise ValueError(f'解码参数不符: {key}')
    probe = bundle['model'].predict_proba(np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32))
    if probe.shape != (1, 2) or not np.isfinite(probe).all():
        raise ValueError('模型推理检查失败')
    return dict(sha256=digest, feature_count=len(FEATURE_NAMES), threshold=bundle['threshold'],
                decoder=actual, n_estimators=bundle['model'].n_estimators, status='passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'models/rf_decoder_best.joblib')
    args = parser.parse_args()
    print(json.dumps(verify(args.model), ensure_ascii=False, indent=2))
