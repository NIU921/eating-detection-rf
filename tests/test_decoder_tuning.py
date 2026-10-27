import csv
import json
from pathlib import Path
import tempfile
import unittest
from dominant_hand.config import Config
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from dominant_hand.events import decode
from dominant_hand.train import load_dataset,_hash_files
from dominant_hand.tune_decoder import CachedDecoder,tune


class DecoderTuningTests(unittest.TestCase):
    def test_cached_decoder_matches_existing_rules(self):
        rng=np.random.default_rng(12); metadata=[]; probability=[]
        for subject in ('A','B'):
            for segment in range(2):
                for i in range(160):
                    if 70<=i<76: continue
                    a=1784600000000+segment*1500000+i*5000
                    metadata.append(dict(externalid=subject,source_zip='x',member='x',segment=segment,
                        start_ms=a,end_ms=a+10000))
                    probability.append(float(rng.random()) if i%40<30 else .9)
        cached=CachedDecoder(metadata,probability,5)
        for smooth in (5,30,60,120):
            for stop in (1,6,12):
                settings=dict(smooth_s=smooth,stop_persistence=stop,start_persistence=3,
                    merge_gap_s=120,min_event_s=60,threshold=.55)
                expected=decode(metadata,probability,threshold=.55,smooth_windows=max(1,round(smooth/5)),
                    stop_persistence=stop,start_persistence=3,merge_gap_s=120,min_event_s=60,stride_s=5)
                self.assertEqual(dict(expected),cached.decode(settings))

    def test_cached_nested_selection_export_resume_and_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); baseline=root/'baseline'; dataset=baseline/'dataset'; training=baseline/'training'
            dataset.mkdir(parents=True); (training/'checkpoints').mkdir(parents=True)
            labels_dir=root/'三类筛选结果'; labels_dir.mkdir()
            fold_dir=root/'惯用手算法设计/五折划分'; fold_dir.mkdir(parents=True)
            def write(path,fields,rows):
                with path.open('w',encoding='utf-8-sig',newline='') as stream:
                    w=csv.DictWriter(stream,fieldnames=fields); w.writeheader(); w.writerows(rows)
            metadata=[]; fold_rows=[]; meals=[]; base=1784600000000
            for i in range(10):
                subject=f'S{i}'; fold=i%5+1
                fold_rows.append(dict(externalid=subject,fold=fold,leakage_group=subject))
                meals.append(dict(externalid=subject,**{'timeFrame.startTime':base+40000,'timeFrame.endTime':base+100000}))
                for j in range(40):
                    a=base+j*5000
                    intersection=max(0,min(a+10000,base+100000)-max(a,base+40000))
                    label=1 if intersection>=8000 else -1 if base-20000<a+10000 and a<base+160000 else 0
                    metadata.append(dict(externalid=subject,leakage_group=subject,fold=fold,source_zip='x',
                        member='x',segment=0,start_ms=a,end_ms=a+10000,label=label))
            # 真值中无窗口的人仍须计漏检，不能按窗口索引删除。
            fold_rows.append(dict(externalid='NO_SIGNAL',fold=1,leakage_group='NO_SIGNAL'))
            meals.append(dict(externalid='NO_SIGNAL',**{'timeFrame.startTime':base+40000,'timeFrame.endTime':base+100000}))
            fields=['externalid','timeFrame.startTime','timeFrame.endTime']
            write(labels_dir/'惯用手用餐信息.csv',fields,meals); write(labels_dir/'非惯用手用餐信息.csv',fields,[])
            fold_path=fold_dir/'subject_folds.csv'; write(fold_path,['externalid','fold','leakage_group'],fold_rows)
            np.save(dataset/'X.npy',np.zeros((len(metadata),1),dtype=np.float32))
            (dataset/'metadata.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metadata),encoding='utf-8')
            (dataset/'feature_names.json').write_text(json.dumps(['test']),encoding='utf-8')
            (dataset/'dataset_summary.json').write_text(json.dumps(dict(config=Config().to_dict(),limited_run=False)),encoding='utf-8')
            _,_,_,_,fingerprint=load_dataset(dataset)
            code_dir=Path(__file__).resolve().parents[1]/'dominant_hand'
            manifest=dict(dataset_sha256=fingerprint,labels_sha256=_hash_files([
                labels_dir/'惯用手用餐信息.csv',labels_dir/'非惯用手用餐信息.csv',fold_path]),
                code_sha256=_hash_files([code_dir/n for n in ('train.py','events.py','labels.py','integrity.py')]))
            (training/'run_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            selected=[dict(fold=f,max_depth=None,min_samples_leaf=5,f1=0.) for f in range(1,6)]
            (training/'training_report.json').write_text(json.dumps(dict(outer_fold_metrics=selected,
                settings=dict(inner_folds=3),outer_oof_event_metrics=dict(f1=0.))),encoding='utf-8')
            joblib.dump(dict(model='sentinel-unchanged-forest',threshold=.55,decoder={},signal_config=Config().to_dict()),
                        training/'final_model.joblib')
            folds=np.array([r['fold'] for r in metadata]); y=np.array([r['label'] for r in metadata])
            groups=np.array([r['leakage_group'] for r in metadata])
            for fold in range(1,6):
                train=np.flatnonzero(folds!=fold); test=np.flatnonzero(folds==fold)
                for k,(_,iva) in enumerate(GroupKFold(3).split(train,y[train],groups[train])):
                    indices=train[iva]
                    np.savez_compressed(training/'checkpoints'/f'inner_o{fold}_dNone_l5_k{k}.npz',
                        indices=indices,probability=np.where(y[indices]==1,.9,.1))
                # 外折概率刻意全0，内折选参仍应根据内折表现选择，不能借助外折。
                np.savez_compressed(training/'checkpoints'/f'outer_{fold}.npz',indices=test,probability=np.zeros(len(test)))
            grid=dict(smooth_s=[5],stop_persistence=[1,6],merge_gap_s=[0,30],threshold=[.35,.55],
                      start_persistence=1,min_event_s=10)
            before=(training/'final_model.joblib').read_bytes(); output=root/'tuning'
            result=tune(root,baseline,output,grid)
            self.assertEqual(result['outer_oof_event_metrics']['fn'],11)
            self.assertEqual(result['outer_oof_event_metrics']['tp'],0)
            self.assertTrue(all(r['inner_event_f1']>0 for r in result['outer_fold_metrics']))
            self.assertEqual(joblib.load(output/'final_model.joblib')['model'],'sentinel-unchanged-forest')
            self.assertEqual(before,(training/'final_model.joblib').read_bytes())
            first=(output/'checkpoints/fold_1.json').stat().st_mtime_ns
            tune(root,baseline,output,grid)
            self.assertEqual(first,(output/'checkpoints/fold_1.json').stat().st_mtime_ns)
            with self.assertRaisesRegex(ValueError,'原基线目录之外'): tune(root,baseline,baseline/'bad',grid)
            with self.assertRaisesRegex(ValueError,'不同配置'):
                tune(root,baseline,output,dict(grid,merge_gap_s=[10]))


if __name__=='__main__': unittest.main()
