"""PyCharm右键运行：复用基线概率，搜索餐次解码参数，输出到独立目录。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from run_local import run_step

ROOT=Path(__file__).resolve().parent


def automatic_output(root,baseline,config):
    """Readable parameter values plus a fingerprint prevent config/code/baseline collisions."""
    canonical=json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False)
    digest=hashlib.sha256(canonical.encode('utf-8'))
    digest.update(str(Path(baseline).resolve()).encode('utf-8'))
    for path in [Path(baseline)/'training'/name for name in ('run_manifest.json','training_report.json')]+[
        Path(root)/'dominant_hand'/name for name in ('tune_decoder.py','train.py','events.py','labels.py')]:
        digest.update(path.name.encode('utf-8'))
        if path.is_file(): digest.update(path.read_bytes())
    parts=[]
    for key,label in (('smooth_s','smooth'),('stop_persistence','stop'),('merge_gap_s','gap'),
                      ('threshold','thr'),('start_persistence','start'),('min_event_s','min')):
        values=config[key] if isinstance(config[key],list) else [config[key]]
        if not values or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in values):
            raise ValueError(f'{key}需要数值或非空数值列表')
        parts.append(label+'-'.join(str(v).replace('.','p').replace('-','m') for v in values))
    # 保持Windows路径长度可控；完整候选值保存在search_config.json中。
    name='_'.join(parts)[:108]+'_'+digest.hexdigest()[:10]
    return Path(root)/'runs/rf_decoder_tuning'/name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=ROOT/'runs/local_baseline')
    p.add_argument('--output',type=Path,help='可选：手动指定；默认按参数及实验指纹自动命名')
    p.add_argument('--config',type=Path,default=ROOT/'experiments/decoder_search.json')
    a=p.parse_args(); baseline=a.baseline.resolve()
    config=json.loads(a.config.read_text(encoding='utf-8-sig'))
    output=(a.output or automatic_output(ROOT,baseline,config)).resolve()
    if output==baseline or baseline in output.parents: raise SystemExit('输出目录必须与原基线分开')
    print(f'原基线: {baseline}\n新结果: {output}\n仅复用概率，不重新处理信号或拟合森林。',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    snapshot=output/'search_config.json'
    if snapshot.exists() and json.loads(snapshot.read_text(encoding='utf-8'))!=config:
        raise SystemExit('指定目录已有不同参数，请省略--output以自动生成新目录')
    snapshot.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    log_name='decoder_search_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.log'
    run_step(['dominant_hand.tune_decoder','--root',str(ROOT),'--baseline',str(baseline),
              '--output',str(output),'--config',str(snapshot)],output/'logs'/log_name)
    print(f'DONE decoder tuning: {output}',flush=True)


if __name__=='__main__': main()
