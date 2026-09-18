import ast
import importlib
from pathlib import Path
import subprocess
import unittest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

class LayoutTests(unittest.TestCase):
    def test_public_imports_and_tokenizer(self):
        from mica import MicaConfig
        from model.model_mica import MicaConfig as Impl
        self.assertIs(MicaConfig, Impl)
        for name in ("dataset.indexed", "dataset.prepare", "trainer.common.engine",
                     "evaluation.loss", "inference.generation", "model.model_lora"):
            importlib.import_module(name)
        tok = AutoTokenizer.from_pretrained(ROOT / "tokenizer", local_files_only=True)
        self.assertTrue(tok.encode("你好 Mica"))
        self.assertEqual(len(tok), 6400)

    def test_no_upstream_or_old_package_imports(self):
        tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        self.assertFalse(any(p.startswith(("minimind/", "src/mica_llm/", "recipes/")) for p in tracked))
        for folder in ("model", "dataset", "trainer", "evaluation", "inference", "scripts"):
            for path in (ROOT / folder).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or "").startswith(("minimind", "mica_llm")) or node.module == "model.model_minimind", str(path))