"""逐TXT处理；保留包内多样本，独立重建ACC/GYRO，禁止跨长缺口滤波。"""
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


AXES = [f"{m}_{a}" for m in ("ACC", "GYRO") for a in "XYZ"]


def read_imu(stream):
    cols = ["ACC_TIME", "GYRO_TIME"] + AXES
    frame = pd.read_csv(stream, sep=r"\s+", usecols=cols, dtype=str)
    raw_rows = len(frame)
    # 重复表头、空字段和非数值按模态报告；不丢弃另一模态的有效样本。
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    arrays = {}
    audit = {"raw_rows": raw_rows}
    for m in ("ACC", "GYRO"):
        t = frame[m+"_TIME"].to_numpy(dtype=float)
        x = frame[[m+"_"+a for a in "XYZ"]].to_numpy(dtype=float)
        valid = np.isfinite(t) & (t > 0) & np.isfinite(x).all(axis=1)
        arrays[m] = (t[valid], x[valid])
        audit[m+"_invalid_rows"] = int((~valid).sum())
    return arrays, audit


def reconstruct(t, x, cfg):
    if len(t) < 2:
        return [], {"packets": 0, "abnormal_packets": 0}
    edges = np.r_[0, np.flatnonzero(np.diff(t) != 0)+1, len(t)]
    anchors, sizes = t[edges[:-1]], np.diff(edges)
    delta = np.diff(anchors)
    ordinary = delta[(delta > 0) & (delta <= 1000)]
    if not len(ordinary):
        return [], {"packets": len(sizes), "abnormal_packets": len(sizes)}
    period = float(np.median(ordinary))
    normal = (delta > 0) & (delta <= 1.5*period)
    estimates = delta[normal] / sizes[:-1][normal]
    dt = float(np.median(estimates)) if len(estimates) else period / float(np.median(sizes))
    if not 0.1 <= dt <= 100:
        raise ValueError(f"不可信的IMU样本间隔: {dt}ms")
    abnormal = sizes > max(30, 3*float(np.median(sizes)))
    # 正常间隔用于包内均匀重建；缺失包附近采用本段采样间隔，不展开缺口。
    times = np.empty(len(t), dtype=float)
    for k, (a,b) in enumerate(zip(edges[:-1], edges[1:])):
        step = dt
        if cfg.anchor == "start":
            if k < len(delta) and normal[k] and not abnormal[k]:
                step = delta[k]/sizes[k]
            times[a:b] = anchors[k] + np.arange(b-a)*step
        else:
            if k > 0 and normal[k-1] and not abnormal[k]:
                step = delta[k-1]/sizes[k]
            times[a:b] = anchors[k] - np.arange(b-a-1,-1,-1)*step
    valid = ~np.repeat(abnormal, sizes)
    # 删除异常包后留下的空洞仍按原始时间分段，不能压缩时间。
    times, x = times[valid], x[valid]
    if len(times) < 2:
        return [], {"packets": len(sizes), "abnormal_packets": int(abnormal.sum())}
    diff = np.diff(times)
    breaks = np.flatnonzero((diff <= 0) | (diff > cfg.max_interpolation_ms) |
                           (diff > min(1000, 3*period)))+1
    blocks = np.r_[0, breaks, len(times)]
    segments = [(times[a:b], x[a:b]) for a,b in zip(blocks[:-1], blocks[1:])
                if b-a >= 4 and times[b-1]-times[a] >= (cfg.window_s+2*cfg.trim_s)*1000]
    return segments, {"packets": len(sizes), "abnormal_packets": int(abnormal.sum()),
                      "packet_interval_ms": period, "sample_interval_ms": dt,
                      "time_backtracks": int((delta < 0).sum()), "segments": len(segments)}


def _resample(t, x, grid, cfg):
    dt = float(np.median(np.diff(t)))
    source_fs = 1000/dt
    if source_fs < 2.1*cfg.lowpass_hz:
        raise ValueError(f"原始采样率{source_fs:.2f}Hz不足以支持{cfg.lowpass_hz}Hz低通")
    # 原生均匀网格上低通后才降采样；整个block不含长缺口。
    native = np.arange(t[0], t[-1]+dt/2, dt)
    values = np.column_stack([np.interp(native,t,x[:,j]) for j in range(3)])
    values = sosfiltfilt(butter(4,cfg.lowpass_hz,fs=source_fs,output="sos"),values,axis=0)
    down = np.column_stack([np.interp(grid,native,values[:,j]) for j in range(3)])
    # 原始观测区间支持：缺样间隙内的格点标记为插值，不能算完整观测。
    right = np.clip(np.searchsorted(t,grid),1,len(t)-1)
    gap = t[right]-t[right-1]
    near = np.minimum(abs(grid-t[right]),abs(grid-t[right-1]))
    observed = (gap <= 1.8*dt) | (near <= 0.6*dt)
    return down, observed


def process(stream, cfg):
    arrays, audit = read_imu(stream)
    blocks = {}
    for mode in ("ACC","GYRO"):
        blocks[mode], audit[mode] = reconstruct(*arrays[mode],cfg)
    result = []
    for at, ax in blocks["ACC"]:
        for gt, gx in blocks["GYRO"]:
            start = max(at[0],gt[0])+cfg.trim_s*1000
            stop = min(at[-1],gt[-1])-cfg.trim_s*1000
            step = 1000/cfg.fs
            grid = np.arange(np.ceil(start/step)*step, stop, step)
            if len(grid) < cfg.window_s*cfg.fs:
                continue
            acc, avalid = _resample(at,ax,grid,cfg)
            gyro, gvalid = _resample(gt,gx,grid,cfg)
            gravity = sosfiltfilt(butter(2,cfg.gravity_hz,fs=cfg.fs,output="sos"),acc,axis=0)
            signals = np.column_stack([acc,gyro,np.linalg.norm(acc,axis=1),
                                       np.linalg.norm(gyro,axis=1),np.linalg.norm(acc-gravity,axis=1)])
            result.append(dict(t=grid.astype(np.int64), x=signals.astype(np.float32),
                               observed=avalid & gvalid))
    # 时间回退可能形成相互覆盖的片段，交给prepare固定来源优先级裁切。
    audit["aligned_segments"] = len(result)
    return result, audit
