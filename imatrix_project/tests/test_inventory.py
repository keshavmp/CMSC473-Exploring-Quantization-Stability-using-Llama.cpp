import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imatrix.evaluate import inventory

class InventoryTests(unittest.TestCase):
    def test_discovers_existing_and_new_model_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'models/imatrix_experiments').mkdir(parents=True)
            (root/'models/imatrix_quants').mkdir(parents=True)
            (root/'models/Llama-3.2-1B-Instruct-BF16.gguf').write_bytes(b'b')
            (root/'models/imatrix_experiments/uniform-q4.gguf').write_bytes(b'c')
            (root/'models/imatrix_quants/Llama-3.2-1B-Instruct-math_q4_imatrix.gguf').write_bytes(b'd')
            found=inventory(root)
            self.assertEqual(set(found),{'bf16','uniform-q4','math-q4'})

if __name__=='__main__':unittest.main()
