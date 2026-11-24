Fix for existing imatrix_standalone_clean folder; only two files changed.

From the imatrix_standalone_clean folder:
  cp imatrix/backend.py imatrix/backend.py.bak
  cp imatrix/evaluate.py imatrix/evaluate.py.bak
  unzip -o ~/Downloads/imatrix_chat_evaluation_fix.zip

Smoke test (new experiment name; no prior raw scores mixed in):
  python -m imatrix evaluate --root .. --models uniform-q4 --limit 1 --experiment imatrix-chat-smoke --prompt-format chat --threads 8 --max-new-tokens 512

Research pilot (paired comparison, same settings and sampled items):
  python -m imatrix evaluate --root .. --models uniform-q4 chat-q4 math-q4 code-q4 --limit 20 --experiment imatrix-chat-q4-n20 --prompt-format chat --threads 8 --max-new-tokens 512

Outputs in ../results/<experiment>/<model>/{metrics,samples,performance,protocol}.json
And ../results/<experiment>/benchmark_scores.csv
The metrics.py in your existing folder already retains GSM8K strict AND flexible metrics.
Don't compare raw versus chat scores without noting the protocol change.
The program checks protocol settings before reusing existing results in the same experiment.
Do not run BF16 by default; use --models for explicit choices.
This fix is not a validation of the recovery branch's historic likelihood-based results.
