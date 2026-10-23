import io
import json
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from app.core import tts_runtime as runtime

class RuntimeTests(unittest.TestCase):
    def test_paths_are_project_scoped(self):
        root=Path(__file__).resolve().parents[1]
        self.assertTrue(Path(runtime.default_paths()['model_dir']).is_relative_to(root/'AppData'))
    def test_gpu_selection(self):
        self.assertEqual(runtime.choose_backend([{'Name':'AMD Radeon Graphics'},{'Name':'NVIDIA GeForce RTX 3060'}]),'cuda')
        self.assertEqual(runtime.choose_backend([{'Name':'AMD Radeon RX 9070 XT'}]),'rocm-gfx120X')
        for name in ('AMD Radeon Graphics','AMD Radeon RX 580','Intel UHD'):
            with self.assertRaises(runtime.RuntimeSetupError):runtime.choose_backend([{'Name':name}])
    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            archive=Path(folder)/'bad.zip'
            with zipfile.ZipFile(archive,'w') as zipped:zipped.writestr('../escape','bad')
            with self.assertRaises(runtime.RuntimeSetupError):runtime._extract(archive,Path(folder)/'out')
            self.assertFalse((Path(folder)/'escape').exists())
    def test_zip_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            archive=Path(folder)/'bad.zip'
            entry=zipfile.ZipInfo('link');entry.external_attr=(0o120777<<16)
            with zipfile.ZipFile(archive,'w') as zipped:zipped.writestr(entry,'../other')
            with self.assertRaises(runtime.RuntimeSetupError):runtime._extract(archive,Path(folder)/'out')
    def test_download_hash_failure_does_not_publish(self):
        class Response(io.BytesIO):headers={'Content-Length':'3'}
        with tempfile.TemporaryDirectory() as folder,patch.object(runtime.urllib.request,'urlopen',return_value=Response(b'bad')):
            path=Path(folder)/'asset'
            with self.assertRaises(runtime.RuntimeSetupError):runtime._fetch('https://example.invalid',path,expected='0'*64)
            self.assertFalse(path.exists());self.assertFalse(path.with_suffix('.download').exists())
    def test_cancel_before_download(self):
        event=threading.Event();event.set()
        with patch.object(runtime.urllib.request,'urlopen') as request:
            with self.assertRaises(runtime.RuntimeSetupError):runtime._fetch('https://example.invalid',Path('never'),cancel=event)
            request.assert_not_called()
    def test_unknown_platform_does_not_install(self):
        with patch.object(runtime.platform,'system',return_value='Linux'),patch.object(runtime,'_fetch') as download:
            with self.assertRaises(runtime.RuntimeSetupError):runtime.prepare_runtime(lambda *a:None,threading.Event())
            download.assert_not_called()
    def test_patch_is_exact_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'module.py';path.write_text('before\n')
            runtime._patch(path,'before','after');runtime._patch(path,'before','after')
            self.assertEqual(path.read_text(),'after\n')
            with self.assertRaises(runtime.RuntimeSetupError):runtime._patch(path,'missing','different')
    def test_missing_runtime_not_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            info=runtime.inspect_runtime(dict(python_path=folder+'/python',repo_path=folder+'/repo',model_dir=folder+'/models'))
            self.assertFalse(info['ready']);self.assertFalse(info['models_ready'])

if __name__=='__main__':unittest.main()
