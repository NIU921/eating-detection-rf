"""默认使用本次整理的最佳 RF 解码模型。"""
import argparse
from pathlib import Path
from dominant_hand.predict import predict_folder

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('externalid', nargs='?')
    p.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'models/rf_decoder_best.joblib')
    p.add_argument('--output', type=Path, default=Path('results.csv'))
    a = p.parse_args()
    print(predict_folder(a.model, a.folder, a.externalid, a.output))

if __name__ == '__main__':
    main()
