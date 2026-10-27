"""推理端使用与prepare相同的片段间去重，保留片段内部有意重叠的滑窗。"""
from .labels import merged,overlap


def window_indices(segment,cfg,covered):
    t=segment['t']; observed=segment['observed']; new=[]
    length,stride=cfg.window_s*cfg.fs,cfg.stride_s*cfg.fs
    for begin in range(0,len(t)-length+1,stride):
        end=begin+length
        if float(observed[begin:end].mean())<cfg.min_observed: continue
        a,b=int(t[begin]),int(t[end-1])+round(1000/cfg.fs)
        if overlap(a,b,covered): continue
        new.append((a,b)); yield begin,end,a,b
    covered[:]=merged(covered+new)
