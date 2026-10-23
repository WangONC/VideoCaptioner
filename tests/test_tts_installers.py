import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from app.core import tts_runtime as r
from app.core.tts_catalog import MODELS, SOURCES, MODEL_REVISIONS


class InstallerTests(unittest.TestCase):
    def manifest(self, root, model='dots-soar'):
        files = []
        for name in r._required_files(model):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = b'asset'
            path.write_bytes(data)
            files.append(dict(repo=MODELS[model]['repo'], revision=MODEL_REVISIONS[MODELS[model]['repo']],
                              relative=name, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
        (root / 'videocaptioner-models.json').write_text(json.dumps(dict(schema=2, model_id=model, files=files)))
        return files

    def test_catalog_sources_are_pinned_and_isolated(self):
        self.assertEqual(len(MODELS), 10)
        for model, spec in MODELS.items():
            project, revision, marker, _ = SOURCES[spec['engine']]
            if project:
                self.assertEqual(len(revision), 40)
                self.assertEqual(len(MODEL_REVISIONS[spec['repo']]), 40)
        self.assertNotEqual(r.default_paths('dots-soar')['python_path'], r.default_paths('qwen3-0.6b')['python_path'])
        self.assertEqual(r.default_paths('qwen3-custom')['python_path'], r.default_paths('qwen3-design')['python_path'])

    def test_config_file_alone_is_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'config.json').write_text('{}')
            self.assertFalse(r.verify_models(root, 'dots-soar'))

    def test_complete_manifest_and_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = self.manifest(root)
            self.assertTrue(r.verify_models(root, 'dots-soar'))
            (root / files[0]['relative']).write_bytes(b'wrong')
            self.assertTrue(r.verify_models(root, 'dots-soar', full=False))
            self.assertFalse(r.verify_models(root, 'dots-soar'))

    def test_manifest_wrong_model_revision_and_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = self.manifest(root)
            self.assertFalse(r.verify_models(root, 'dots-mf'))
            (root / files[0]['relative']).unlink()
            self.assertFalse(r.verify_models(root, 'dots-soar', full=False))

    def test_inventory_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = self.manifest(root)
            files.append(dict(relative='../outside', size=1, sha256='a', repo=MODELS['dots-soar']['repo'], revision=MODEL_REVISIONS[MODELS['dots-soar']['repo']]))
            (root / 'videocaptioner-models.json').write_text(json.dumps(dict(schema=2, model_id='dots-soar', files=files)))
            self.assertFalse(r.verify_models(root, 'dots-soar'))

    def test_cancellation_during_verification_is_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.manifest(root)
            cancel = threading.Event(); cancel.set()
            with self.assertRaises(r.RuntimeSetupError): r.verify_models(root, 'dots-soar', cancel)

    def test_edge_no_gpu_no_model_download(self):
        with patch.object(r, '_fetch') as fetch, patch.object(r, 'detect_backend') as detect:
            self.assertTrue(r.models_ready({'model_id': 'edge-tts'}))
            progress = []
            r.download_models(lambda *v: progress.append(v), None, {'model_id': 'edge-tts'})
            fetch.assert_not_called(); detect.assert_not_called()
            self.assertEqual(progress[-1][0], 100)
            self.assertNotIn('import torch', r._runtime_probe('edge-tts', Path('/unused')))

    def test_design_includes_fixed_clone_base(self):
        repos = r._model_repos('qwen3-design')
        self.assertEqual(repos[1][:2], ('Qwen/Qwen3-TTS-12Hz-1.7B-Base', 'voice_clone_base'))
        self.assertIn('voice_clone_base/model.safetensors', r._required_files('qwen3-design'))

    def test_non_lfs_git_blob_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'config'; p.write_bytes(b'{}')
            row = dict(size=2, blob_id=hashlib.sha1(b'blob 2\0{}').hexdigest())
            self.assertTrue(r._file_matches(p, row))
            p.write_bytes(b'[]'); self.assertFalse(r._file_matches(p, row))

    def test_dots_optional_normalizer_patch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root / 'src/dots_tts/utils/text.py'; p.parent.mkdir(parents=True)
            p.write_text('from tn.chinese.normalizer import Normalizer as ZhNormalizer\nfrom tn.english.normalizer import Normalizer as EnNormalizer\ndef get_chinese_text_normalizer() -> ZhNormalizer:\n    return ZhNormalizer()\ndef get_english_text_normalizer() -> EnNormalizer:\n    return EnNormalizer()\n')
            r._dots_windows_compat(root)
            s = p.read_text()
            self.assertNotIn('\nfrom tn.', '\n' + s)
            self.assertIn('    from tn.chinese', s)

    def test_cancelled_model_download_reuses_verified_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cancel = threading.Event()
            data = b'asset'
            revision = MODEL_REVISIONS[MODELS['dots-soar']['repo']]
            siblings = [dict(rfilename=name, size=len(data), lfs={'sha256': hashlib.sha256(data).hexdigest()})
                        for name in ('first.bin', 'second.bin')]
            calls = []
            def fetch(url, path, cancel_arg=None, expected=None, progress=None):
                calls.append(Path(path).name)
                Path(path).write_bytes(data)
                if len(calls) == 1:
                    cancel.set()
            with patch.object(r, '_paths', return_value={'model_dir': root, 'repo_path': root / 'source'}), \
                 patch.object(r, '_required_files', return_value={'first.bin', 'second.bin'}), \
                 patch.object(r, '_json_url', return_value={'sha': revision, 'siblings': siblings}), \
                 patch.object(r, '_fetch', side_effect=fetch):
                with self.assertRaises(r.RuntimeSetupError):
                    r.download_models(lambda *v: None, cancel, {'model_id': 'dots-soar'})
                self.assertFalse((root / 'videocaptioner-models.json').exists())
                cancel.clear()
                r.download_models(lambda *v: None, cancel, {'model_id': 'dots-soar'})
                self.assertEqual(calls, ['first.bin', 'second.bin'])
                self.assertTrue(r.verify_models(root, 'dots-soar'))

    @unittest.skipIf(os.name == 'nt', 'POSIX process-group test; Windows uses taskkill /T')
    def test_command_cancel_stops_child_process(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / 'should-not-appear'
            code = 'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",' + repr('import time,pathlib;time.sleep(1.5);pathlib.Path(' + repr(str(marker)) + ').write_text("bad")') + ']);time.sleep(10)'
            cancel = threading.Event()
            timer = threading.Timer(.3, cancel.set); timer.start()
            with self.assertRaises(r.RuntimeSetupError): r._command([sys.executable, '-c', code], cancel)
            timer.join(); time.sleep(1.6)
            self.assertFalse(marker.exists())


if __name__ == '__main__': unittest.main()
