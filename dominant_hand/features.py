"""确定性IMU窗口特征；所有特征仅从一个10秒窗口计算。"""
import numpy as np


FEATURE_NAMES = []
for channel in range(9):
    FEATURE_NAMES.extend(f"ch{channel}_{name}" for name in
                         ("mean", "std", "min", "max", "median", "iqr", "rms", "p90_p10", "skew", "kurtosis"))
    FEATURE_NAMES.extend(f"ch{channel}_band_{i}" for i in range(4))
    FEATURE_NAMES.extend((f"ch{channel}_dominant_hz", f"ch{channel}_spectral_entropy"))
FEATURE_NAMES.extend(("acc_xy_corr", "acc_xz_corr", "acc_yz_corr", "gyro_xy_corr",
                      "gyro_xz_corr", "gyro_yz_corr", "dynamic_diff_rms",
                      "gyro_autocorr_max", "gyro_autocorr_lag_s"))


def _corr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract(window, fs=25):
    x = np.asarray(window, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 9 or not np.isfinite(x).all():
        raise ValueError("窗口必须是有限的[n,9] IMU特征信号")
    mean=x.mean(axis=0); sd=x.std(axis=0); centered=x-mean
    quantiles=np.quantile(x,[.1,.25,.5,.75,.9],axis=0)
    skew=np.mean(centered**3,axis=0)/np.maximum(sd,1e-8)**3
    kurt=np.mean(centered**4,axis=0)/np.maximum(sd,1e-8)**4-3
    skew[sd<=1e-8]=0; kurt[sd<=1e-8]=0
    stats=np.stack([mean,sd,x.min(axis=0),x.max(axis=0),quantiles[2],
                    quantiles[3]-quantiles[1],np.sqrt(np.mean(x*x,axis=0)),
                    quantiles[4]-quantiles[0],skew,kurt],axis=1)
    freq=np.fft.rfftfreq(len(x),1/fs)
    power=np.abs(np.fft.rfft(centered*np.hanning(len(x))[:,None],axis=0))**2
    region=(freq>=.1)&(freq<=8); total=power[region].sum(axis=0)
    bands=[power[(freq>=a)&(freq<b)].sum(axis=0)/(total+1e-12)
           for a,b in ((.1,.5),(.5,2),(2,5),(5,8))]
    p=power[region]/(total+1e-12)
    dominant=freq[region][np.argmax(power[region],axis=0)]
    dominant[total<=1e-12]=0
    entropy=-np.sum(p*np.log(p+1e-12),axis=0)/np.log(max(2,len(p)))
    result=np.column_stack([stats,*bands,dominant,entropy]).ravel().tolist()
    for offset in (0,3):
        result.extend(_corr(x[:,offset+a],x[:,offset+b]) for a,b in ((0,1),(0,2),(1,2)))
    result.append(float(np.sqrt(np.mean(np.diff(x[:,8])**2))))
    gyro = x[:,7]
    lags = np.arange(max(1,fs//2), min(len(gyro)//2,fs*4)+1)
    if len(lags) and np.std(gyro)>1e-8:
        # FFT求所有延迟的乘积和，前缀和恢复各延迟的Pearson归一化。
        centered_gyro=gyro-gyro.mean(); n=len(gyro)
        f=np.fft.rfft(centered_gyro,n=2*n)
        cross=np.fft.irfft(f*np.conj(f),n=2*n)[lags]
        sums=np.r_[0,np.cumsum(centered_gyro)]; squares=np.r_[0,np.cumsum(centered_gyro**2)]
        count=n-lags; s1=sums[count]; s2=sums[-1]-sums[lags]
        var1=np.maximum(0,squares[count]-s1*s1/count)
        var2=np.maximum(0,squares[-1]-squares[lags]-s2*s2/count)
        denom=np.sqrt(var1*var2)
        ac=np.divide(cross-s1*s2/count,denom,out=np.zeros_like(cross),where=denom>1e-16)
        k=int(np.argmax(ac)); result.extend((float(ac[k]),float(lags[k]/fs)))
    else:
        result.extend((0.0,0.0))
    return np.asarray(result,dtype=np.float32)
