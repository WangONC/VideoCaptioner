import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import tts_model_metadata as metadata
from app.core import tts_runtime as runtime
from app.core.tts_catalog import MODELS, MODEL_REVISIONS


class ModelMetadataTests(unittest.TestCase):
    def fake_qwen(self, url):
        repo = url.split('/api/models/')[1].split('/revision/')[0]
        names = ['config.json', 'model.safetensors', 'speech_tokenizer/model.safetensors', 'speech_tokenizer/config.json']
        return dict(sha=MODEL_REVISIONS[repo], siblings=[dict(rfilename=n, size=10, blobId='b'*40) for n in names])

    def test_all_local_catalog_sizes_available_offline(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runtime, '_json_url', side_effect=AssertionError('network')):
            for model_id in MODELS:
                value = metadata.get_cached_model_bytes(model_id, folder)
                self.assertIsInstance(value, int)
                self.assertGreaterEqual(value, 0)
        self.assertEqual(metadata.get_cached_model_bytes('qwen3-0.6b'), 2516100892)
        self.assertEqual(metadata.get_cached_model_bytes('gpt-sovits'), 1826743304)

    def test_design_contains_clone_base(self):
        rows = metadata.build_model_inventory('qwen3-design', self.fake_qwen)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len([r for r in rows if r['relative'].startswith('voice_clone_base/')]), 4)

    def test_cache_atomic_and_recipe_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(metadata.fetch_model_bytes('qwen3-0.6b', folder, self.fake_qwen), 40)
            self.assertEqual(metadata.get_cached_model_bytes('qwen3-0.6b', folder), 40)
            self.assertEqual(len(list(Path(folder).glob('*'))), 1)
            with patch.dict(MODEL_REVISIONS, {'Qwen/Qwen3-TTS-12Hz-0.6B-Base': 'new-revision'}):
                self.assertIsNone(metadata.get_cached_model_bytes('qwen3-0.6b', folder))

    def test_invalid_cache_uses_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            path = metadata._cache_path('qwen3-0.6b', folder)
            path.write_text('{invalid')
            self.assertEqual(metadata.get_cached_model_bytes('qwen3-0.6b', folder), 2516100892)

    def test_failed_fetch_preserves_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            metadata.fetch_model_bytes('qwen3-0.6b', folder, self.fake_qwen)
            with self.assertRaises(OSError):
                metadata.fetch_model_bytes('qwen3-0.6b', folder, lambda _: (_ for _ in ()).throw(OSError('offline')))
            self.assertEqual(metadata.get_cached_model_bytes('qwen3-0.6b', folder), 40)

    def test_revision_missing_file_bad_size_and_path_rejected(self):
        for mutation in ('revision', 'missing', 'size', 'path', 'duplicate'):
            def fetch(url):
                data = self.fake_qwen(url)
                if mutation == 'revision': data['sha'] = 'wrong'
                elif mutation == 'missing': data['siblings'].pop()
                elif mutation == 'size': data['siblings'][0]['size'] = True
                elif mutation == 'path': data['siblings'].append(dict(rfilename='safe/../../evil', size=1, blobId='a'))
                else: data['siblings'].append(data['siblings'][0])
                return data
            with self.subTest(mutation=mutation), self.assertRaises(runtime.RuntimeSetupError):
                metadata.build_model_inventory('qwen3-0.6b', fetch)

    def test_cancellation_and_edge_do_not_network(self):
        cancel = threading.Event(); cancel.set()
        with patch.object(runtime, '_json_url') as fetch:
            with self.assertRaises(runtime.RuntimeSetupError):
                metadata.build_model_inventory('qwen3-0.6b', cancel=cancel)
            self.assertEqual(metadata.fetch_model_bytes('edge-tts'), 0)
            fetch.assert_not_called()

    def test_gpt_auxiliary_and_audio_cpp_design(self):
        self.assertEqual(metadata.GPT_AUXILIARY_BYTES, 725084747)
        from app.core.audio_cpp_runtime import required_models
        self.assertEqual(len(required_models('qwen3-design')), 2)
