"""Identical training/inference features; no sample-rate assumption or forward fill.

Rows carry repeated batch timestamps: aggregate all valid rows in absolute-time bins.
Modality timestamps are used independently. Missing/all-zero modality rows are ignored.
Long gaps in ACC coverage start separate runs, never a fabricated continuous interval.
"""
import io
import numpy as np
import pandas as pd


def feature_names(config):
    names = []
    if config.get('use_motion', True):
        for prefix in ('ACC', 'GYRO'):
            for axis in 'XYZ':
                names.extend(prefix + '_' + axis + '_' + stat for stat in ('mean', 'std', 'min', 'max'))
    for channel in range(1, config['ppg_channels'] + 1):
        names.extend('PPG%d_%s' % (channel, stat) for stat in ('mean', 'std'))
    return names + ['ACC_present', 'GYRO_present', 'PPG_present']


def extract_runs(source, config, is_text=False):
    step = config['bin_ms']
    use_motion = config.get('use_motion', True)
    modalities = []
    if use_motion:
        modalities = [('ACC', ['ACC_' + a for a in 'XYZ']),
                      ('GYRO', ['GYRO_' + a for a in 'XYZ'])]
    if config['ppg_channels']:
        modalities.append(('PPG', ['PPG%d' % n for n in range(1, config['ppg_channels'] + 1)]))
    # Runs are segmented by the anchor modality's coverage: ACC by default,
    # PPG when motion channels are ablated (use_motion=false).
    anchor = 'ACC' if use_motion else 'PPG'
    stream = io.StringIO(source) if is_text else source
    # Collect sufficient statistics per modality and bin. Storage scales with time bins.
    stats = {m: {} for m, _ in modalities}
    extrema = {}
    try:
        reader = pd.read_csv(stream, sep='\t', chunksize=100000, encoding='utf-8-sig')
        for frame in reader:
            for modality, columns in modalities:
                if modality + '_TIME' not in frame or not set(columns).issubset(frame.columns):
                    if modality == anchor:
                        raise ValueError('TXT missing %s_TIME or %s columns' % (anchor, anchor))
                    continue  # Optional absent modalities have zeros and presence=0.
                ts = pd.to_numeric(frame[modality + '_TIME'], errors='coerce').to_numpy(dtype=np.float64)
                values = frame[columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
                valid = ((ts >= 10**12) & (ts < 10**13) & np.isfinite(values).all(axis=1)
                         & (values != 0).any(axis=1))
                if not valid.any():
                    continue
                times = ts[valid].astype(np.int64)
                # Log compression avoids unverified physical-unit/ADC scale assumptions.
                vals = values[valid]
                vals = np.sign(vals) * np.log1p(np.abs(vals))
                bins = times // step
                temp = pd.DataFrame(vals)
                temp['bin'] = bins
                grouped = temp.groupby('bin', sort=False)
                sums = grouped.sum()
                squares = pd.DataFrame(vals * vals)
                squares['bin'] = bins
                sumsq = squares.groupby('bin', sort=False).sum()
                lows, highs, counts = grouped.min(), grouped.max(), grouped.size()
                for b in sums.index:
                    item = [int(counts.loc[b]), sums.loc[b].to_numpy(), sumsq.loc[b].to_numpy(),
                            lows.loc[b].to_numpy(), highs.loc[b].to_numpy()]
                    b = int(b)
                    old = stats[modality].get(b)
                    if old is None:
                        stats[modality][b] = item
                    else:
                        old[0] += item[0]; old[1] += item[1]; old[2] += item[2]
                        old[3] = np.minimum(old[3], item[3]); old[4] = np.maximum(old[4], item[4])
                if modality == anchor:
                    extrema['min'] = min(extrema.get('min', int(times.min())), int(times.min()))
                    extrema['max'] = max(extrema.get('max', int(times.max())), int(times.max()))
    except pd.errors.EmptyDataError:
        return []
    acc_bins = sorted(stats[anchor])
    if not acc_bins:
        return []
    groups = [[acc_bins[0]]]
    for b in acc_bins[1:]:
        if (b - groups[-1][-1]) * step > config['gap_ms']:
            groups.append([])
        groups[-1].append(b)
    runs = []
    for group in groups:
        length = group[-1] - group[0] + 1
        if length < config['min_tokens']:
            continue
        features = np.zeros((len(feature_names(config)), length), dtype=np.float32)
        for i, b in enumerate(range(group[0], group[-1] + 1)):
            offset = 0
            for modality, columns in modalities:
                item = stats[modality].get(b)
                width = 4 if modality != 'PPG' else 2
                if item is not None:
                    count, total, total2, low, high = item
                    mean = total / count
                    std = np.sqrt(np.maximum(total2 / count - mean**2, 0))
                    matrix = np.stack((mean, std, low, high) if width == 4 else (mean, std), axis=1)
                    features[offset:offset + len(columns)*width, i] = matrix.reshape(-1)
                    features[-3 + ['ACC', 'GYRO', 'PPG'].index(modality), i] = 1
                offset += len(columns) * width
        runs.append({'features': features, 'grid_start_ms': group[0] * step,
                     'valid_start_ms': max(group[0] * step, extrema['min']),
                     'valid_end_ms': min((group[-1] + 1) * step, extrema['max'] + 1),
                     'bin_ms': step})
    return runs
