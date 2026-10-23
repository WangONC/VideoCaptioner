"""CLI contract/process tests with a local fake, not audio.cpp/model inference."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave

from app.core.dubbing.audio_cpp_provider import AudioCppProvider
from app.core.dubbing.pipeline import DubbingError, CancelledError


_FAKE = r'''
import json, pathlib, sys, time, wave
args = sys.argv[1:]
def option(name): return args[args.index(name)+1]
request = json.loads(pathlib.Path(option('--request-sequence')).read_text('utf-8'))['requests'][0]
with open('calls.jsonl', 'a', encoding='utf-8') as log:
    log.write(json.dumps({'args': args, 'request': request}, ensure_ascii=False) + '\n')
text = request['text']
if text == '__hang': time.sleep(20)
if text == '__fail':
    print('fixture inference failed', file=sys.stderr)
    sys.exit(17)
output = pathlib.Path(option('--out-dir')) / (request['id'] + '.wav')
if text == '__missing': sys.exit(0)
if text == '__invalid':
    output.write_bytes(b'not a WAV')
    sys.exit(0)
with wave.open(str(output), 'wb') as stream:
    stream.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
    stream.writeframes(b'\0\0' * 2400)
if text == '__truncated': output.write_bytes(output.read_bytes()[:-20])
'''


class AudioCppProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.exe = self.root / 'fake-native-cli'
        self.exe.write_text(_FAKE, 'utf-8')
        self.models = self.root / 'models'
        self.models.mkdir()
        for leaf in ('index-tts2-q8_0.gguf', 'qwen3-tts-12hz-0.6b-base-q8_0.gguf',
                     'qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf'):
            (self.models / leaf).write_bytes(b'GGUF fixture, not model weights')
        self.ref = self.root / '参考声音.wav'
        with wave.open(str(self.ref), 'wb') as stream:
            stream.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            stream.writeframes(b'\0\0' * 2400)
        self.provider = AudioCppProvider(self.exe, self.models, 'indextts2', timeout=3)
        real_popen = subprocess.Popen
        self.popen = patch('app.core.dubbing.audio_cpp_provider.subprocess.Popen',
            side_effect=lambda args, **kwargs: real_popen([sys.executable, str(self.exe)] + args[1:], **kwargs))
        self.popen_mock = self.popen.start()

    def tearDown(self):
        self.provider.close()
        self.popen.stop()
        self.temp.cleanup()

    def latest_call(self):
        return json.loads((self.root / 'calls.jsonl').read_text('utf-8').splitlines()[-1])

    def test_utf8_request_and_pcm_output(self):
        text = '中文 "引号"; & $(nothing)\n第二行'
        output = self.root / '输出音频.wav'
        self.provider.synthesize(text, self.ref, output)
        call = self.latest_call()
        self.assertEqual(call['request']['text'], text)
        self.assertEqual(call['request']['voice_ref'], str(self.ref))
        self.assertNotIn(text, call['args'])
        self.assertEqual(call['args'][call['args'].index('--family') + 1], 'index_tts2')
        self.assertEqual(call['args'][call['args'].index('--backend') + 1], 'vulkan')
        self.assertTrue(output.is_file())
        self.assertEqual(list(self.root.glob('.audiocpp-*')), [])
        self.assertIsNone(self.provider.process)
        self.assertNotIn('shell', self.popen_mock.call_args.kwargs)

    def test_qwen_transcript_and_embedding_modes(self):
        provider = AudioCppProvider(self.exe, self.models, 'qwen3-1.7b')
        provider.reference_text = 'Reference 中文 transcript'
        provider.synthesize('你好', self.ref, self.root / 'with.wav')
        call = self.latest_call()
        self.assertEqual(call['request']['reference_text'], provider.reference_text)
        self.assertNotIn('x_vector_only_mode=true', call['args'])
        provider.reference_text = ''
        provider.synthesize('你好', self.ref, self.root / 'without.wav')
        call = self.latest_call()
        self.assertNotIn('reference_text', call['request'])
        self.assertEqual(call['request']['options']['x_vector_only_mode'], 'true')
        self.assertEqual(call['args'][call['args'].index('--model') + 1],
                         str(self.models / 'qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf'))

    def test_failure_log_missing_invalid_and_truncated_outputs(self):
        for text, error in [('__fail', 'fixture inference failed'), ('__missing', 'valid PCM WAV'),
                            ('__invalid', 'valid PCM WAV'), ('__truncated', 'truncated WAV')]:
            with self.subTest(text=text):
                output = self.root / (text + '.wav')
                with self.assertRaisesRegex(DubbingError, error):
                    self.provider.synthesize(text, self.ref, output)
                self.assertFalse(output.exists())
                self.assertEqual(list(self.root.glob('.audiocpp-*')), [])

    def test_preexisting_output_and_input_are_not_overwritten(self):
        output = self.root / 'kept.wav'
        output.write_bytes(b'keep')
        for path in (output, self.ref, self.exe, self.provider.model):
            original = path.read_bytes()
            with self.assertRaisesRegex(DubbingError, 'overwrite'):
                self.provider.synthesize('你好', self.ref, path)
            self.assertEqual(path.read_bytes(), original)
        self.popen_mock.assert_not_called()

    def test_late_output_is_not_overwritten(self):
        output = self.root / 'late.wav'
        real_validate = self.provider._validate_wav
        def validate_then_create(path):
            real_validate(path)
            output.write_bytes(b'late user file')
        with patch.object(self.provider, '_validate_wav', side_effect=validate_then_create):
            with self.assertRaisesRegex(DubbingError, 'appeared'):
                self.provider.synthesize('你好', self.ref, output)
        self.assertEqual(output.read_bytes(), b'late user file')

    def test_cancellation_stops_process_and_removes_staging(self):
        cancel = threading.Event()
        timer = threading.Timer(.2, cancel.set)
        timer.start()
        try:
            with self.assertRaises(CancelledError):
                self.provider.synthesize('__hang', self.ref, self.root / 'cancelled.wav', cancel)
        finally:
            timer.cancel()
        self.assertIsNone(self.provider.process)
        self.assertEqual(list(self.root.glob('.audiocpp-*')), [])

    def test_timeout_stops_process_and_allows_retry(self):
        self.provider.timeout = .15
        with self.assertRaisesRegex(DubbingError, 'timed out'):
            self.provider.synthesize('__hang', self.ref, self.root / 'timeout.wav')
        self.assertIsNone(self.provider.process)
        self.provider.timeout = 3
        self.provider.synthesize('retry', self.ref, self.root / 'retry.wav')
        self.assertTrue((self.root / 'retry.wav').exists())

    def test_fingerprint_changes_with_model_runtime_and_backend(self):
        initial = self.provider.fingerprint()
        self.provider.model.write_bytes(b'GGUF changed')
        model_changed = self.provider.fingerprint()
        self.assertNotEqual(initial['model_sha256'], model_changed['model_sha256'])
        self.exe.write_text(_FAKE + '\n# changed', 'utf-8')
        self.assertNotEqual(model_changed['runtime_sha256'], self.provider.fingerprint()['runtime_sha256'])
        cpu = AudioCppProvider(self.exe, self.models, 'indextts2', 'cpu')
        self.assertNotEqual(self.provider.fingerprint(), cpu.fingerprint())

    def test_invalid_model_or_backend_refused(self):
        for model_id, backend in [('bogus', 'vulkan'), ('indextts2', 'best')]:
            with self.assertRaises(DubbingError):
                AudioCppProvider(self.exe, self.models, model_id, backend)
        self.provider.model.write_bytes(b'not GGUF')
        with self.assertRaisesRegex(DubbingError, 'GGUF'):
            AudioCppProvider(self.exe, self.models, 'indextts2')

    def test_direct_gguf_path_and_precancel(self):
        provider = AudioCppProvider(self.exe, self.provider.model, 'indextts2')
        self.assertEqual(provider.model, self.provider.model)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            provider.synthesize('你好', self.ref, self.root / 'unused.wav', cancel)
        self.popen_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()

class AudioCppExtendedTests(unittest.TestCase):
    setUp = AudioCppProviderTests.setUp
    tearDown = AudioCppProviderTests.tearDown
    latest_call = AudioCppProviderTests.latest_call
    def test_custom_design_and_dots_requests(self):
        from app.core.audio_cpp_runtime import AUDIO_CPP_MODELS
        for model_id in ('qwen3-custom', 'qwen3-design', 'dots-mf', 'dots-soar'):
            spec = AUDIO_CPP_MODELS[model_id]
            (self.models / spec['filename']).write_bytes(b'GGUF fixture')
            provider = AudioCppProvider(self.exe, self.models, model_id, voice='Vivian', description='沉稳男声', seed=7)
            provider.reference_text = '参考文本'
            provider.synthesize('你好', self.ref if provider.requires_reference else None, self.root / (model_id + '.wav'))
            call = self.latest_call()
            self.assertEqual(call['request']['seed'], 7)
            if model_id == 'qwen3-custom':
                self.assertEqual(call['request']['voice_id'], 'Vivian')
                self.assertNotIn('voice_ref', call['request'])
            if model_id == 'qwen3-design':
                self.assertEqual(call['request']['instruct'], '沉稳男声')
                self.assertNotIn('voice_ref', call['request'])
            if model_id == 'dots-mf':
                self.assertEqual(call['request']['options']['num_inference_steps'], '4')


class AudioCppInstallTests(unittest.TestCase):
    def test_supported_models_are_exact(self):
        from app.core.audio_cpp_runtime import AUDIO_CPP_MODELS, default_paths
        self.assertEqual(len(AUDIO_CPP_MODELS), 7)
        for unsupported in ('gpt-sovits', 'cosyvoice3', 'edge-tts'):
            with self.assertRaises(Exception):
                default_paths(unsupported)
        for model_id in AUDIO_CPP_MODELS:
            paths = default_paths(model_id)
            self.assertIn('audio.cpp', paths['python_path'])
            self.assertIn('audio.cpp', paths['model_dir'])

    def test_partial_model_not_ready_and_hash_verified(self):
        import hashlib
        import app.core.audio_cpp_runtime as runtime
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b'GGUF test payload'
            spec = dict(filename='test.gguf', size=len(data), sha256=hashlib.sha256(data).hexdigest())
            config = dict(model_id='indextts2', model_dir=directory)
            with patch.dict(runtime.AUDIO_CPP_MODELS, {'indextts2':spec}):
                (root/'test.gguf.download').write_bytes(data)
                self.assertFalse(runtime.models_ready(config))
                (root/'test.gguf').write_bytes(data[:-1])
                self.assertFalse(runtime.models_ready(config))
                (root/'test.gguf').write_bytes(data)
                self.assertTrue(runtime.models_ready(config, verify_hash=True))
                self.assertTrue(runtime.models_ready(config))
                (root/'test.gguf').write_bytes(b'X'+data[1:])
                self.assertFalse(runtime.models_ready(config))

    def test_download_pinned_hash_and_existing_file_reuse(self):
        import hashlib
        import app.core.audio_cpp_runtime as runtime
        from app.core.tts_runtime import RuntimeSetupError
        with tempfile.TemporaryDirectory() as directory:
            data=b'GGUF model'
            spec=dict(filename='test.gguf', path='tests/test.gguf', size=len(data),sha256=hashlib.sha256(data).hexdigest())
            config=dict(model_id='indextts2',model_dir=directory)
            def fetch(url, target, cancel, expected, progress):
                self.assertIn(runtime.MODEL_REVISION, url)
                self.assertEqual(expected,spec['sha256'])
                Path(target).write_bytes(data)
            with patch.dict(runtime.AUDIO_CPP_MODELS, {'indextts2':spec}), patch('app.core.tts_runtime._fetch',side_effect=fetch) as mocked:
                runtime.download_models(lambda *args:None, None, config)
                runtime.download_models(lambda *args:None, None, config)
                self.assertEqual(mocked.call_count,1)

    def test_runtime_manifest_tamper_does_not_execute(self):
        import hashlib
        import app.core.audio_cpp_runtime as runtime
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); exe=root/'audiocpp_cli.exe';exe.write_bytes(b'original')
            manifest=dict(release=runtime.RELEASE,archive_sha256=runtime.ASSET_SHA256,files={'audiocpp_cli.exe':hashlib.sha256(b'original').hexdigest()})
            (root/'videocaptioner-runtime.json').write_text(json.dumps(manifest))
            config=dict(model_id='indextts2',python_path=str(exe),repo_path=str(root),model_dir=str(root/'model'))
            with patch('app.core.tts_runtime._command', return_value='available_devices=1\nvulkan:0 "AMD Radeon" [GPU]\n') as mocked:
                self.assertTrue(runtime.inspect_runtime(config)['ready'])
                self.assertEqual(mocked.call_count,1)
                exe.write_bytes(b'changed')
                state=runtime.inspect_runtime(config)
                self.assertFalse(state['ready'])
                self.assertEqual(mocked.call_count,1)

class AudioCppRuntimeSetupTests(unittest.TestCase):
    def test_atomic_install_and_existing_runtime_check(self):
        import zipfile
        import app.core.audio_cpp_runtime as runtime
        with tempfile.TemporaryDirectory() as directory:
            def fetch(url, output, cancel, expected, progress):
                self.assertEqual(expected, runtime.ASSET_SHA256)
                with zipfile.ZipFile(output, 'w') as z:
                    z.writestr('bundle/audiocpp_cli.exe', b'fixture, not executable')
                    z.writestr('bundle/ggml-vulkan.dll', b'fixture')
                    z.writestr('bundle/model_specs/qwen3_tts.json', '{}')
            with patch.object(runtime,'ROOT',Path(directory)), patch.object(runtime.platform,'system',return_value='Windows'), patch.object(runtime.platform,'machine',return_value='AMD64'), patch('app.core.tts_runtime._fetch',side_effect=fetch) as download, patch('app.core.tts_runtime._command',return_value='vulkan:0 "Radeon" [GPU]\n'):
                runtime.prepare_runtime(lambda *a:None,None)
                paths=runtime.default_paths()
                self.assertTrue(Path(paths['python_path']).is_file())
                self.assertTrue((Path(paths['repo_path'])/'model_specs/qwen3_tts.json').is_file())
                runtime.prepare_runtime(lambda *a:None,None)
                self.assertEqual(download.call_count,1)
                self.assertEqual(list(Path(paths['repo_path']).parent.glob('.audiocpp-setup-*')),[])

    def test_unsafe_archive_never_published(self):
        import zipfile
        import app.core.audio_cpp_runtime as runtime
        from app.core.tts_runtime import RuntimeSetupError
        with tempfile.TemporaryDirectory() as directory:
            def fetch(url, output, *args):
                with zipfile.ZipFile(output, 'w') as z:
                    z.writestr('../escape.exe', b'bad')
            with patch.object(runtime,'ROOT',Path(directory)), patch.object(runtime.platform,'system',return_value='Windows'), patch.object(runtime.platform,'machine',return_value='AMD64'), patch('app.core.tts_runtime._fetch',side_effect=fetch), patch('app.core.tts_runtime._command') as execute:
                with self.assertRaises(RuntimeSetupError):
                    runtime.prepare_runtime(lambda *a:None,None)
                self.assertFalse(Path(runtime.default_paths()['repo_path']).exists())
                execute.assert_not_called()

class AudioCppDesignedVoiceTests(unittest.TestCase):
    setUp = AudioCppProviderTests.setUp
    tearDown = AudioCppProviderTests.tearDown
    latest_call = AudioCppProviderTests.latest_call

    def test_design_once_then_clone_and_reuse_across_jobs(self):
        from types import SimpleNamespace
        from app.core.audio_cpp_runtime import AUDIO_CPP_MODELS, required_models
        from app.core.dubbing.audio_cpp_provider import DesignedAudioCppProvider
        for spec in required_models('qwen3-design'):
            (self.models/spec['filename']).write_bytes(b'GGUF fixture')
        config=SimpleNamespace(python_path=str(self.exe),model_dir=str(self.models),cache_dir=str(self.root/'cache'),language='中文',description='沉稳讲师',seed=-1)
        provider=DesignedAudioCppProvider(config)
        fingerprint=provider.fingerprint()
        provider.synthesize('第一句',None,self.root/'one.wav')
        provider.synthesize('第二句',None,self.root/'two.wav')
        calls=[json.loads(row) for row in (self.root/'calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(calls),3)
        self.assertEqual(calls[0]['args'][calls[0]['args'].index('--task')+1],'vdes')
        self.assertEqual(calls[1]['request']['voice_ref'],calls[2]['request']['voice_ref'])
        self.assertEqual(calls[1]['request']['reference_text'],provider.prompt_text)
        second=DesignedAudioCppProvider(config)
        self.assertEqual(second.fingerprint(),fingerprint)
        second.synthesize('第三句',None,self.root/'three.wav')
        self.assertEqual(len((self.root/'calls.jsonl').read_text().splitlines()),4)
        config.description='不同声音'
        third=DesignedAudioCppProvider(config)
        self.assertNotEqual(third.fingerprint(),fingerprint)
        self.assertEqual(len((self.root/'calls.jsonl').read_text().splitlines()),5)
        provider.close();second.close();third.close()

    def test_design_download_requires_base_companion(self):
        import app.core.audio_cpp_runtime as runtime
        self.assertEqual(len(runtime.required_models('qwen3-design')),2)
        self.assertEqual(runtime.required_models('qwen3-design')[1],runtime.AUDIO_CPP_MODELS['qwen3-1.7b'])
