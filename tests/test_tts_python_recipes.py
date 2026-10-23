"""Recipe admission is separate from actual hardware/inference qualification."""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from app.core import tts_runtime as r


class PythonRecipeTests(unittest.TestCase):
    def test_cp313_depends_on_engine_and_backend(self):
        for engine in ('edge-tts', 'qwen3', 'dots'):
            r._check_python_recipe(engine, 'online' if engine == 'edge-tts' else 'cpu', (3, 13))
        for engine in ('indextts2', 'gpt-sovits', 'cosyvoice3'):
            with self.assertRaisesRegex(r.RuntimeSetupError, '3.12'):
                r._check_python_recipe(engine, 'cpu', (3, 13))
        with self.assertRaisesRegex(r.RuntimeSetupError, '3.12'):
            r._check_python_recipe('qwen3', 'rocm-gfx120X', (3, 13))
        r._check_python_recipe('qwen3', 'rocm-gfx120X', (3, 12))

    def test_future_versions_and_frozen_executable_are_not_assumed(self):
        with self.assertRaises(r.RuntimeSetupError):
            r._check_python_recipe('edge-tts', 'online', (3, 14))
        with patch.object(r.sys, 'frozen', True, create=True):
            with self.assertRaises(r.RuntimeSetupError):
                r._check_python_recipe('edge-tts', 'online', (3, 13))

    def test_gpt_model_assets_before_or_after_environment(self):
        for engine_exists in (False, True):
            with self.subTest(engine_exists=engine_exists), tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / 'source'
                sentinel = source / 'GPT_SoVITS/infer.py'
                if engine_exists:
                    sentinel.parent.mkdir(parents=True)
                    sentinel.write_text('preserve engine')
                def fetch(url, dest, *args, **kwargs):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if str(dest).endswith('.zip'):
                        with zipfile.ZipFile(dest, 'w') as z:
                            z.writestr('G2PWModel/POLYPHONIC_CHARS.txt', 'fixture')
                            z.writestr('G2PWModel/g2pw.onnx', 'fixture')
                    else:
                        with dest.open('wb') as f:
                            f.truncate(131266198)
                    return r._sha256(dest)
                def nltk(source, cancel):
                    path = source / '.nltk_data/corpora/cmudict.zip'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'fixture')
                with patch.object(r, '_fetch', side_effect=fetch), patch.object(r, '_prepare_nltk', side_effect=nltk), patch.object(r, '_command', side_effect=AssertionError('must not install environment')):
                    rows = r._prepare_gpt_assets(source, None, lambda *a: None)
                    self.assertTrue(r._verify_auxiliary(source, rows))
                self.assertTrue((source/'GPT_SoVITS/text/G2PWModel/g2pw.onnx').is_file())
                if engine_exists:
                    self.assertEqual(sentinel.read_text(), 'preserve engine')
