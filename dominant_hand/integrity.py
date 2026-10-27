"""训练前独立复算时间窗标签，阻止旧标签缓存与新餐次标注混用。"""
from collections import defaultdict
import numpy as np
from .labels import read_catalog,merged


def validate_labels(root,metadata,config,background_policy='all_nonmeal_confirmed'):
    catalog=read_catalog(root); by_subject=defaultdict(list)
    for i,row in enumerate(metadata): by_subject[row['externalid']].append(i)
    for subject,indices in by_subject.items():
        rows=[metadata[i] for i in indices]
        a=np.array([r['start_ms'] for r in rows],dtype=np.int64)
        b=np.array([r['end_ms'] for r in rows],dtype=np.int64)
        actual=np.array([r['label'] for r in rows])
        if np.any(b<=a) or not set(actual)<={-1,0,1}: raise ValueError('窗口时间或标签无效')
        if np.any(b-a!=config['window_s']*1000): raise ValueError('窗口长度与信号配置不一致')
        meals=catalog.get(subject,[])
        def duration(intervals):
            total=np.zeros(len(rows),dtype=np.int64)
            for c,d in merged(intervals): total+=np.maximum(0,np.minimum(b,d)-np.maximum(a,c))
            return total
        positive=duration((m['start'],m['end']) for m in meals if m['dominant'])
        ignored=duration((m['start'],m['end']) for m in meals if not m['dominant'])
        any_meal=duration((m['start'],m['end']) for m in meals)
        buffer=config['boundary_s']*1000
        expanded=duration((m['start']-buffer,m['end']+buffer) for m in meals)
        expected=np.zeros(len(rows),dtype=np.int8)
        expected[(any_meal>0)|(expanded>0)]=-1
        expected[positive/(b-a)>=config['positive_overlap']]=1
        expected[ignored>0]=-1
        allowed=(actual==expected)
        if background_policy=='restricted_coverage': allowed|=(expected==0)&(actual==-1)
        if not allowed.all():
            bad=int(np.flatnonzero(~allowed)[0]); row=rows[bad]
            raise ValueError(f"缓存标签与当前标注不一致: {subject} {row['start_ms']} 保存={actual[bad]} 应为={expected[bad]}；重新准备数据")
    return dict(status='passed',windows=len(metadata),subjects=len(by_subject))


def valid_probability(probability,expected):
    values=np.asarray(probability)
    if values.shape!=(expected,) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('概率缓存长度无效，或包含NaN/无穷/超出[0,1]的值')
    return values
