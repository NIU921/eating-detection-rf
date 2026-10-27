"""窗口概率转事件及IoU>0.25一对一事件评分。"""
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

from .labels import read_catalog, subtract, merged, overlap


def decode(metadata, probability, threshold=.6, stop_threshold=None, smooth_windows=12, start_persistence=3,
           stop_persistence=6, merge_gap_s=30, min_event_s=60, stride_s=5):
    stop_threshold = threshold-.2 if stop_threshold is None else stop_threshold
    if not 0 <= stop_threshold < threshold <= 1:
        raise ValueError("事件停止阈值必须低于开始阈值")
    if len(metadata)!=len(probability): raise ValueError("窗口与概率数量不一致")
    if min(smooth_windows,start_persistence,stop_persistence,stride_s)<=0:
        raise ValueError("平滑窗、确认窗及步长必须为正")
    initial=defaultdict(list); coverage=defaultdict(list)
    for row,p in zip(metadata,probability):
        initial[(row["externalid"],row["source_zip"],row["member"],row["segment"])].append(
            (int(row["start_ms"]),int(row["end_ms"]),float(p)))
        coverage[row["externalid"]].append((int(row["start_ms"]),int(row["end_ms"])))
    # 质量筛选可能在同一原始片段内造成空洞，不能跨空洞平滑和确认状态。
    groups={}
    for key,items in initial.items():
        block=0; previous=None
        for item in sorted(items):
            if previous is not None and item[0]-previous>stride_s*1500:
                block+=1
            groups.setdefault((*key,block),[]).append(item); previous=item[0]
    events=defaultdict(list)
    for (subject,_,_,_,_),items in groups.items():
        items=sorted(items)
        starts=np.array([r[0] for r in items]); ends=np.array([r[1] for r in items])
        p=np.array([r[2] for r in items]); smoothed=np.empty_like(p)
        left=(smooth_windows-1)//2; right=smooth_windows-left
        for i in range(len(p)):
            smoothed[i]=p[max(0,i-left):min(len(p),i+right)].mean()
        active=False; high=low=0; begin=finish=None
        for i,value in enumerate(smoothed):
            if not active:
                high=high+1 if value>=threshold else 0
                if high>=start_persistence:
                    active=True; begin=int(starts[i-start_persistence+1]); low=0
            else:
                low=low+1 if value<stop_threshold else 0
                if low>=stop_persistence:
                    finish=int(ends[i-stop_persistence+1]);
                    if finish>begin: events[subject].append((begin,finish))
                    active=False; high=low=0
        if active:
            finish=int(ends[-1])
            if finish>begin: events[subject].append((begin,finish))
    for subject,values in events.items():
        combined=[]; support=merged(coverage[subject])
        for a,b in sorted(values):
            if combined:
                end=combined[-1][1]; gap=max(0,a-end)
                can_merge=gap<=merge_gap_s*1000 and (gap==0 or overlap(end,a,support)>=gap)
            else: can_merge=False
            if can_merge: combined[-1]=(combined[-1][0],max(b,combined[-1][1]))
            else: combined.append((a,b))
        events[subject]=[(a,b) for a,b in combined if b-a>=min_event_s*1000]
    return events


def score_events(predicted, truth_by_subject, iou_threshold=.25):
    tp=fp=fn=0; start_errors=[]; end_errors=[]
    for subject in set(predicted)|set(truth_by_subject):
        pred=predicted.get(subject,[]); truth=truth_by_subject.get(subject,[])
        if not pred or not truth:
            fp+=len(pred); fn+=len(truth); continue
        iou=np.zeros((len(truth),len(pred)),dtype=float)
        for i,(a,b) in enumerate(truth):
            for j,(c,d) in enumerate(pred):
                intersection=max(0,min(b,d)-max(a,c)); union=max(b,d)-min(a,c)
                if union: iou[i,j]=intersection/union
        # 数量权重大于任意IoU和变化，严格先最大化匹配数，再最大化IoU总和。
        reward=np.where(iou>iou_threshold,min(len(truth),len(pred))+1+iou,0)
        ri,ci=linear_sum_assignment(-reward)
        pairs=[(i,j) for i,j in zip(ri,ci) if iou[i,j]>iou_threshold]
        tp+=len(pairs); fp+=len(pred)-len(pairs); fn+=len(truth)-len(pairs)
        for i,j in pairs:
            start_errors.append(abs(pred[j][0]-truth[i][0])/1000)
            end_errors.append(abs(pred[j][1]-truth[i][1])/1000)
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    return dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,f1=f1,
                start_mae_s=float(np.mean(start_errors)) if start_errors else None,
                end_mae_s=float(np.mean(end_errors)) if end_errors else None)


def remove_ignored(predicted, root, subjects):
    catalog=read_catalog(root); result={}
    for subject in subjects:
        ignored=[(m["start"],m["end"]) for m in catalog.get(subject,[]) if not m["dominant"]]
        result[subject]=subtract(predicted.get(subject,[]),ignored)
    return result


def truth_for(root, subjects):
    catalog=read_catalog(root); result={}
    for subject in subjects:
        result[subject]=[(m["start"],m["end"]) for m in catalog.get(subject,[]) if m["dominant"]]
    return result
