"""Small real subprocess/venv checks; no network, model or GPU downloads."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import zipfile
from unittest.mock import patch

from app.core import tts_runtime as r


class SetupReuseTests(unittest.TestCase):
    def test_real_subprocess_cache_isolation_and_live_output(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(r, 'ROOT', Path(folder)):
            updates = []
            text = r._command([sys.executable, '-c',
                'import os,time,json;print("PACKAGE downloading",flush=True);time.sleep(.2);'
                'print(json.dumps({k:os.environ[k] for k in ("PIP_CACHE_DIR","TEMP","HF_HOME","XDG_CACHE_HOME")}))'], progress=updates.append)
            env = json.loads(text.splitlines()[-1])
            self.assertTrue(all(Path(value).is_relative_to(folder) for value in env.values()))
            self.assertTrue(any('PACKAGE downloading' in line for line in updates))
            self.assertEqual(list((Path(folder)/'AppData/tts/_cache/tmp').iterdir()), [])

    def test_cancel_retry_reuses_real_venv_without_python_download(self):
        real_command = r._command
        with tempfile.TemporaryDirectory() as folder, patch.object(r, 'ROOT', Path(folder)), patch.object(r, '_fetch', side_effect=AssertionError('no downloads')):
            cancel = threading.Event()
            commands = []
            def command(args, *a, **kw):
                commands.append([str(x) for x in args])
                if '-m' in args and 'pip' in args:
                    if 'install' in args and not cancel.is_set():
                        cancel.set()
                        raise r.RuntimeSetupError('cancel during pip')
                    return 'edge-tts==7.3.1'
                if '-c' in args and 'VC_GPU:' in str(args[-1]):
                    return 'VC_GPU:{"backend":"online","device":"Edge TTS"}'
                return real_command(args, *a, **kw)
            with patch.object(r, '_command', side_effect=command):
                with self.assertRaisesRegex(r.RuntimeSetupError, 'cancel during pip'):
                    r.prepare_runtime(lambda *a: None, cancel, {'model_id':'edge-tts'})
            python = Path(r.default_paths('edge-tts')['python_path'])
            self.assertTrue(python.is_file())
            timestamp = python.stat().st_mtime_ns
            self.assertFalse(r.runtime_installed({'model_id':'edge-tts'}))
            # Retry with pip/network replaced only; actual interpreter isolation is checked.
            def retry(args, *a, **kw):
                if 'pip' in args: return 'edge-tts==7.3.1'
                if '-c' in args and 'VC_GPU:' in str(args[-1]): return 'VC_GPU:{"backend":"online"}'
                return real_command(args, *a, **kw)
            cancel.clear()
            with patch.object(r, '_command', side_effect=retry):
                r.prepare_runtime(lambda *a: None, cancel, {'model_id':'edge-tts'})
            self.assertEqual(python.stat().st_mtime_ns, timestamp)
            self.assertTrue(r.runtime_installed({'model_id':'edge-tts'}))
            self.assertEqual(sum('venv' in args for args in commands), 1)

    def test_source_cache_and_model_first_safe_merge(self):
        with tempfile.TemporaryDirectory() as folder:
            stage = Path(folder)/'cache';stage.mkdir()
            destination = Path(folder)/'source'
            asset = destination/'GPT_SoVITS/pretrained_models/asset.bin'
            asset.parent.mkdir(parents=True);asset.write_bytes(b'model')
            downloads = []
            def fetch(url, path, *args, **kwargs):
                downloads.append(url)
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr('upstream/GPT_SoVITS/infer.py', 'code')
                return r._sha256(path)
            with patch.object(r, '_fetch', side_effect=fetch):
                first = r._download_source('owner/repo', 'revision', stage, destination, None)
                second = r._download_source('owner/repo', 'revision', stage, destination, None)
            self.assertEqual(first, second)
            self.assertEqual(len(downloads), 1)
            self.assertEqual(asset.read_bytes(), b'model')
            self.assertEqual((destination/'GPT_SoVITS/infer.py').read_text(), 'code')

    def test_check_never_downloads_or_invokes_pip(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(r, 'ROOT', Path(folder)), patch.object(r, '_fetch', side_effect=AssertionError('download')), patch.object(r, '_command') as command:
            self.assertFalse(r.inspect_runtime({'model_id':'edge-tts'})['ready'])
            command.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_audio_cpp_explicit_cpu_does_not_probe_vulkan(self):
        from app.core import audio_cpp_runtime as audio
        with tempfile.TemporaryDirectory() as folder, patch.object(audio, 'ROOT', Path(folder)):
            paths = audio._paths({'model_id':'qwen3-0.6b'})
            exe = paths['python_path'];exe.parent.mkdir(parents=True);exe.write_bytes(b'fixture')
            (paths['repo_path']/'videocaptioner-runtime.json').write_text(json.dumps(dict(
                release=audio.RELEASE, archive_sha256=audio.ASSET_SHA256,
                files={exe.name:audio._digest(exe)})))
            with patch.object(r, '_command', return_value='usage: audiocpp_cli') as command:
                state = audio.inspect_runtime({'model_id':'qwen3-0.6b','device':'cpu'})
                self.assertTrue(state['ready'])
                self.assertEqual(state['device'], 'cpu')
                self.assertEqual(command.call_args.args[0][-1], '--help')

    def test_cpu_is_explicit_and_uses_separate_environment(self):
        gpu = r.default_paths('qwen3-0.6b')
        cpu = r.default_paths('qwen3-0.6b', 'cpu')
        self.assertNotEqual(cpu['python_path'], gpu['python_path'])
        self.assertEqual(cpu['model_dir'], gpu['model_dir'])
        probe = r._runtime_probe('qwen3', Path('/source'), 'cpu')
        self.assertNotIn('select_gpu', probe)
        self.assertIn("device='cpu'", probe)
        self.assertIn('HF_HUB_OFFLINE', probe)
        self.assertEqual(r.default_paths('edge-tts'), r.default_paths('edge-tts', 'cpu'))

    def test_amd_wrong_python_does_not_create_environment(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(r, 'ROOT', Path(folder)), patch.object(r, 'detect_backend', return_value='rocm-gfx120X'), patch.object(r.sys, 'version_info', (3,10,11)), patch.object(r, '_command') as command:
            with self.assertRaisesRegex(r.RuntimeSetupError, '3.12'):
                r.prepare_runtime(lambda *a: None, threading.Event())
            command.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_unowned_environment_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(r, 'ROOT', Path(folder)):
            root = Path(r.default_paths('edge-tts')['repo_path']).parent
            (root/'runtime').mkdir(parents=True)
            existing = root/'runtime/user-file';existing.write_text('keep')
            with self.assertRaisesRegex(r.RuntimeSetupError, '未覆盖'):
                r.prepare_runtime(lambda *a: None, threading.Event(), {'model_id':'edge-tts'})
            self.assertEqual(existing.read_text(), 'keep')


if __name__ == '__main__': unittest.main()
