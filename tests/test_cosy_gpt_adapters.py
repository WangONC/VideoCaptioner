import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
try:
    import numpy as np
    import soundfile as sf
except ImportError:
    np = sf = None
from app.core.dubbing.adapter_cosy_gpt import (
    CosyVoice3Adapter, GPTSoVITSAdapter, COSY_PROMPT, _write_audio, create_adapter,
)


class AdapterContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ref = self.root / 'reference.wav'
        self.ref.write_bytes(b'fixture')
        self.request = dict(text='你好', reference_audio=str(self.ref), reference_text='Hello',
                            output_path=str(self.root / 'output.wav'), language='中文', seed=7)

    def tearDown(self):
        self.temp.cleanup()

    def test_cosy_zero_shot_exact_prompt(self):
        obj = CosyVoice3Adapter.__new__(CosyVoice3Adapter)
        obj.model = Mock(sample_rate=24000)
        obj.model.inference_zero_shot.return_value = iter([{'tts_speech': [[.2] * 20]}])
        with patch('app.core.dubbing.adapter_cosy_gpt._write_audio') as save:
            obj.synthesize(self.request)
        obj.model.inference_zero_shot.assert_called_once_with('你好', COSY_PROMPT+'Hello',
            str(self.ref.resolve()), stream=False, speed=1.0)
        self.assertEqual(save.call_args.args[2], 24000)

    def test_cosy_without_transcript_uses_crosslingual(self):
        obj = CosyVoice3Adapter.__new__(CosyVoice3Adapter)
        obj.model = Mock(sample_rate=24000)
        obj.model.inference_cross_lingual.return_value = iter([{'tts_speech': [[.2] * 20]}])
        with patch('app.core.dubbing.adapter_cosy_gpt._write_audio'):
            obj.synthesize(dict(self.request, reference_text=''))
        obj.model.inference_cross_lingual.assert_called_once_with(COSY_PROMPT+'你好',
            str(self.ref.resolve()), stream=False, speed=1.0)

    def test_cosy_late_error_does_not_write(self):
        obj = CosyVoice3Adapter.__new__(CosyVoice3Adapter)
        obj.model = Mock(sample_rate=24000)
        def broken():
            yield {'tts_speech': [[0] * 20]}
            raise RuntimeError('inference failed')
        obj.model.inference_zero_shot.return_value = broken()
        with patch('app.core.dubbing.adapter_cosy_gpt._write_audio') as save:
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                obj.synthesize(self.request)
            save.assert_not_called()

    def test_gpt_uses_v2_local_config_and_does_not_cut_first_yield(self):
        obj = GPTSoVITSAdapter.__new__(GPTSoVITSAdapter)
        obj.model = Mock()
        obj.model.run.return_value = iter([(32000, [1] * 20),
                                           (32000, [1] * 30)])
        with patch('app.core.dubbing.adapter_cosy_gpt._write_audio') as save:
            obj.synthesize(self.request)
        options = obj.model.run.call_args.args[0]
        self.assertEqual(options['seed'], 7)
        self.assertEqual(options['text_lang'], 'zh')
        self.assertEqual(options['prompt_lang'], 'auto')
        self.assertFalse(options['use_flash_attention'])
        self.assertEqual(len(save.call_args.args[1]), 2)

    def test_gpt_silence_before_exception_is_failure(self):
        obj = GPTSoVITSAdapter.__new__(GPTSoVITSAdapter)
        obj.model = Mock()
        def broken():
            yield 16000, [0] * 16000
            raise RuntimeError('model failed')
        obj.model.run.return_value = broken()
        with patch('app.core.dubbing.adapter_cosy_gpt._write_audio') as save:
            with self.assertRaisesRegex(RuntimeError, 'model failed'):
                obj.synthesize(self.request)
            save.assert_not_called()

    def test_bad_language_and_missing_reference_refused(self):
        obj = GPTSoVITSAdapter.__new__(GPTSoVITSAdapter)
        obj.model = Mock()
        with self.assertRaisesRegex(RuntimeError, '不支持'):
            obj.synthesize(dict(self.request, language='German'))
        with self.assertRaisesRegex(RuntimeError, '参考音频'):
            obj.synthesize(dict(self.request, reference_audio=''))
        obj.model.run.assert_not_called()

    @unittest.skipUnless(np is not None and sf is not None, "NumPy/soundfile installed in model runtime only")
    def test_write_valid_pcm_and_reject_empty_nan(self):
        import soundfile as sf
        _write_audio(self.request['output_path'], [np.ones(50, dtype=np.int16)*2048], 32000)
        data, rate = sf.read(self.request['output_path'], dtype='int16')
        self.assertEqual(rate, 32000)
        self.assertTrue((data == 2048).all())
        with self.assertRaises(RuntimeError): _write_audio('unused', [], 32000)
        with self.assertRaises(RuntimeError): _write_audio('unused', [np.array([np.nan])], 32000)

    def test_missing_weights_fail_before_import_or_download(self):
        with self.assertRaisesRegex(RuntimeError, '缺少本地模型'):
            create_adapter('cosyvoice3', self.root, self.root)
        with self.assertRaisesRegex(RuntimeError, '缺少本地模型'):
            create_adapter('gpt-sovits', self.root, self.root)


if __name__ == '__main__':
    unittest.main()
