import unittest
from types import SimpleNamespace
from unittest.mock import patch
from app.core.tts_gpu import select_gpu


class Tensor:
    def __add__(self, other): return self
    def sum(self): return self
    def item(self): return 16


class FakeTorch:
    __version__ = 'test'
    def __init__(self, hip=True, fail=False, names=None):
        self.version = SimpleNamespace(hip='7.2' if hip else None, cuda=None)
        self.props = names or [SimpleNamespace(name='AMD Radeon Graphics', gcnArchName='gfx1151', total_memory=2),
                              SimpleNamespace(name='AMD Radeon RX 9070 XT', gcnArchName='gfx1201', total_memory=16)]
        self.cuda = self
        self.selected = None
        self.probed = []
        self.fail = fail
    def is_available(self): return True
    def device_count(self): return len(self.props)
    def get_device_properties(self, index): return self.props[index]
    def set_device(self, index): self.selected = index
    def synchronize(self, index): assert index == self.selected
    def ones(self, size, device):
        self.probed.append(device)
        if self.fail: raise RuntimeError('hipErrorInvalidImage')
        return Tensor()


class GPUSelectionTests(unittest.TestCase):
    @patch('app.core.tts_gpu._drivers', return_value=[])
    def test_igpu_zero_is_never_probed(self, _):
        torch = FakeTorch()
        result = select_gpu(torch)
        self.assertEqual(result['device'], 'cuda:1')
        self.assertEqual(torch.selected, 1)
        self.assertEqual(torch.probed, ['cuda:1'])
        self.assertEqual(result['architecture'], 'gfx1201')
        self.assertFalse(result['devices'][0]['supported'])

    @patch('app.core.tts_gpu._drivers', return_value=[])
    def test_invalid_kernel_is_diagnostic_failure_not_fallback(self, _):
        torch = FakeTorch(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'hipErrorInvalidImage') as exc:
            select_gpu(torch)
        self.assertIn('gfx1201', str(exc.exception))
        self.assertIn('cuda:1', str(exc.exception))
        self.assertEqual(torch.probed, ['cuda:1'])

    @patch('app.core.tts_gpu._drivers', return_value=[])
    def test_unsupported_amd_is_not_run(self, _):
        torch = FakeTorch()
        torch.props = torch.props[:1]
        with self.assertRaisesRegex(RuntimeError, '未找到'):
            select_gpu(torch)
        self.assertEqual(torch.probed, [])

    @patch('app.core.tts_gpu._drivers', return_value=[])
    def test_cuda_selection(self, _):
        torch = FakeTorch(hip=False, names=[SimpleNamespace(name='NVIDIA RTX', total_memory=24)])
        self.assertEqual(select_gpu(torch)['device'], 'cuda:0')

    @patch('app.core.tts_gpu._drivers', return_value=[])
    def test_unavailable_is_failure(self, _):
        torch = FakeTorch()
        torch.is_available = lambda: False
        with self.assertRaisesRegex(RuntimeError, '没有可用'):
            select_gpu(torch)
        self.assertEqual(torch.probed, [])

class AdapterDeviceBindingTests(unittest.TestCase):
    def test_cosy_thread_and_onnx_devices_are_explicit(self):
        import sys
        from unittest.mock import Mock
        from app.core.dubbing.adapter_cosy_gpt import CosyVoice3Adapter
        session = Mock()
        session.get_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        model = SimpleNamespace(model=SimpleNamespace(), frontend=SimpleNamespace(speech_tokenizer_session=session))
        device = SimpleNamespace(index=1)
        cuda = Mock()
        with patch.dict(sys.modules, {
            'torch': SimpleNamespace(cuda=cuda, device=lambda _:device),
            'cosyvoice.cli.cosyvoice': SimpleNamespace(AutoModel=lambda **_:model),
        }), patch('app.core.dubbing.adapter_cosy_gpt._require', side_effect=lambda path, **_:path):
            adapter = CosyVoice3Adapter('/models', '/repo', device='cuda:1')
        cuda.set_device.assert_called_once_with(device)
        self.assertIs(adapter.model.model.device, device)
        self.assertIs(adapter.model.frontend.device, device)
        session.set_providers.assert_called_once_with([
            ('CUDAExecutionProvider', {'device_id': 1}), 'CPUExecutionProvider'])

class ExplicitCPUContracts(unittest.TestCase):
    def test_cpu_selection_never_touches_cuda(self):
        from app.core.tts_gpu import select_device
        torch = SimpleNamespace(__version__='fake', version=SimpleNamespace(hip=None))
        seen = []
        torch.ones = lambda n, device: (seen.append(device) or Tensor())
        self.assertEqual(select_device('cpu', torch)['backend'], 'cpu')
        self.assertEqual(seen, ['cpu'])

    def test_qwen_cpu_uses_float32_and_no_cuda_api(self):
        import sys
        from unittest.mock import Mock
        from app.core.dubbing.adapter_standard import StandardAdapter
        model = Mock()
        torch = SimpleNamespace(float32='fp32', float16='fp16')
        with patch.dict(sys.modules, {'torch': torch,
                'qwen_tts': SimpleNamespace(Qwen3TTSModel=model)}):
            StandardAdapter('qwen3', '/models', '/repo', device='cpu')
        self.assertEqual(model.from_pretrained.call_args.kwargs['dtype'], 'fp32')
        self.assertEqual(model.from_pretrained.call_args.kwargs['device_map'], 'cpu')

    def test_dots_cpu_uses_float32(self):
        import sys
        from unittest.mock import Mock
        from app.core.dubbing.adapter_standard import StandardAdapter
        runtime = Mock()
        torch = SimpleNamespace(device=lambda value:value)
        with patch.dict(sys.modules, {'torch': torch,
                'dots_tts.runtime': SimpleNamespace(DotsTtsRuntime=runtime)}):
            adapter = StandardAdapter('dots', '/models', '/repo', device='cpu')
        self.assertEqual(runtime.from_pretrained.call_args.kwargs['precision'], 'float32')
        self.assertEqual(adapter.model.device, 'cpu')

    def test_index_cpu_disables_half_and_kernel(self):
        import sys
        from unittest.mock import Mock
        from app.core.dubbing.adapter_standard import StandardAdapter
        model = Mock()
        with patch.dict(sys.modules, {'torch': SimpleNamespace(),
                'indextts.infer_v2': SimpleNamespace(IndexTTS2=model)}):
            StandardAdapter('indextts2', '/models', '/repo', device='cpu')
        self.assertEqual(model.call_args.kwargs['device'], 'cpu')
        self.assertFalse(model.call_args.kwargs['use_fp16'])
        self.assertFalse(model.call_args.kwargs['use_cuda_kernel'])
