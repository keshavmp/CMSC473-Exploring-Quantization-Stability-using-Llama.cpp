from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imatrix.metrics import extract
from imatrix.reports import build_reports
import pandas as pd

class MetricsTests(unittest.TestCase):
 def test_correct_keys_not_sample_len(self):
  raw={
   'gsm8k':{'sample_len':2,'exact_match,strict-match':0.0,'exact_match,flexible-extract':0.5},
   'ifeval':{'sample_len':2,'prompt_level_strict_acc,none':0.5},
   'mmlu_college_computer_science_generative':{'sample_len':2,'exact_match,get_response':0.0},
  }
  val=extract(raw)
  self.assertEqual(val['gsm8k_strict'],0)
  self.assertEqual(val['gsm8k_flexible'],50)
  self.assertEqual(val['ifeval_strict'],50)
  self.assertEqual(val['mmlu_generative'],0)
 def test_out_of_range_fails(self):
  with self.assertRaises(ValueError):extract({'gsm8k':{'exact_match,strict-match':2}})
 def test_report_baseline_delta(self):
  frame=pd.DataFrame([
   {'model':'bf16','size_mib':2200,'gsm8k_strict':80,'ifeval_strict':70,'mmlu_generative':60},
   {'model':'uniform-q4','size_mib':770,'gsm8k_strict':70,'ifeval_strict':65,'mmlu_generative':50}
  ])
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);build_reports(frame,out)
   saved=pd.read_csv(out/'accuracy_retention.csv')
   self.assertEqual(float(saved.loc[1,'gsm8k_strict_drop_pp']),10)
   self.assertAlmostEqual(float(saved.loc[1,'compression_ratio']),2200/770)
   self.assertTrue((out/'benchmark_accuracy.png').exists())

if __name__=='__main__':unittest.main()
