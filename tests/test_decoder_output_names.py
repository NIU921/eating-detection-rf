import json
from pathlib import Path
import tempfile
import unittest
from run_decoder_tuning import automatic_output


class OutputNameTests(unittest.TestCase):
    def test_names_are_stable_and_isolate_parameter_and_code_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); baseline=root/'baseline'
            config=dict(smooth_s=[30,60,120],stop_persistence=[6,12,18],merge_gap_s=[30,60,120],
                        threshold=[.35,.45,.55,.65],start_persistence=3,min_event_s=60)
            first=automatic_output(root,baseline,config)
            self.assertEqual(first,automatic_output(root,baseline,json.loads(json.dumps(config))))
            self.assertIn('smooth30-60-120',first.name)
            self.assertIn('thr0p35-0p45-0p55-0p65',first.name)
            self.assertNotEqual(first,automatic_output(root,baseline,dict(config,merge_gap_s=[30,120])))
            self.assertNotEqual(first,automatic_output(root,root/'other_baseline',config))
            (root/'dominant_hand').mkdir()
            (root/'dominant_hand/tune_decoder.py').write_text('# changed',encoding='utf-8')
            self.assertNotEqual(first,automatic_output(root,baseline,config))
            long=automatic_output(root,baseline,dict(config,smooth_s=list(range(1,100))))
            self.assertLessEqual(len(long.name),119)
            self.assertFalse(first.exists())


if __name__=='__main__': unittest.main()
