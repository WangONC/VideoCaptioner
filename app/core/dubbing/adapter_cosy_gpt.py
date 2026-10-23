"""Offline adapters for pinned public APIs, imported only inside a TTS worker.

CosyVoice: QwenAudio/CosyVoice@074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc,
  cosyvoice/cli/cosyvoice.py and example.py.
GPT-SoVITS: RVC-Boss/GPT-SoVITS@b4570b876f5c086819ba84e3215472733c6a7b70,
  GPT_SoVITS/TTS_infer_pack/TTS.py. This adapter selects the v2 base weights,
  not an arbitrary user fine-tune, v3/v4/v5 vocoder, or HTTP API server.
GPU/network isolation and cancellation are owned by the worker process.
"""
from pathlib import Path
import os
import sys
import tempfile

COSY_PROMPT = 'You are a helpful assistant.<|endofprompt|>'
GPT_FILES = {
    't2s_weights_path': 'gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt',
    'vits_weights_path': 'gsv-v2final-pretrained/s2G2333k.pth',
    'bert_base_path': 'chinese-roberta-wwm-ext-large',
    'cnhuhbert_base_path': 'chinese-hubert-base',
}
GPT_LANGUAGES = {'中文': 'zh', 'Chinese': 'zh', 'zh': 'zh', '英语': 'en',
    '英文': 'en', 'English': 'en', 'en': 'en', '日语': 'ja', 'Japanese': 'ja',
    'ja': 'ja', '韩语': 'ko', 'Korean': 'ko', 'ko': 'ko', '粤语': 'yue',
    'Cantonese': 'yue', 'yue': 'yue', '自动': 'auto', 'Auto': 'auto', 'auto': 'auto'}


def _require(path, directory=False):
    path = Path(path).resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise RuntimeError('缺少本地模型文件，请先完成下载：' + str(path))
    return path


def _reference(request):
    path = request.get('reference_audio')
    if not path:
        raise RuntimeError('此模型需要参考音频')
    return str(_require(path))


def _write_audio(output, chunks, sample_rate):
    import numpy as np
    import soundfile as sf
    if not chunks or int(sample_rate) <= 0:
        raise RuntimeError('模型没有返回有效音频')
    arrays = []
    for chunk in chunks:
        if hasattr(chunk, 'detach'):
            chunk = chunk.detach().float().cpu().numpy()
        data = np.asarray(chunk)
        if data.ndim == 2 and data.shape[0] == 1:
            data = data[0]
        if data.ndim != 1 or not data.size or not np.isfinite(data).all():
            raise RuntimeError('模型返回了无效音频形状或数值')
        arrays.append(data)
    # int16 from GPT is already PCM; soundfile preserves its scaling. Cosy is float.
    sf.write(str(output), np.concatenate(arrays), int(sample_rate), subtype='PCM_16')


class CosyVoice3Adapter:
    def __init__(self, models, repo):
        root = Path(models).resolve()
        for name in ('cosyvoice3.yaml', 'llm.pt', 'flow.pt', 'hift.pt',
                     'campplus.onnx', 'speech_tokenizer_v3.onnx'):
            _require(root / name)
        _require(root / 'CosyVoice-BlankEN', directory=True)
        matcha = _require(Path(repo) / 'third_party' / 'Matcha-TTS', directory=True)
        sys.path.insert(0, str(matcha))
        from cosyvoice.cli.cosyvoice import AutoModel
        self.model = AutoModel(model_dir=str(root), load_trt=False, load_vllm=False, fp16=False)

    def synthesize(self, request):
        reference = _reference(request)
        transcript = (request.get('reference_text') or '').strip()
        style = (request.get('style') or '').strip()
        if style:
            iterator = self.model.inference_instruct2(request['text'],
                'You are a helpful assistant. ' + style + '<|endofprompt|>',
                reference, stream=False, speed=1.0)
        elif transcript:
            iterator = self.model.inference_zero_shot(request['text'], COSY_PROMPT + transcript,
                reference, stream=False, speed=1.0)
        else:
            # Official cross-lingual mode does not require a transcript.
            iterator = self.model.inference_cross_lingual(COSY_PROMPT + request['text'],
                reference, stream=False, speed=1.0)
        # Exhaust the iterator before publication; late inference errors must propagate.
        chunks = [part['tts_speech'] for part in iterator]
        _write_audio(request['output_path'], chunks, self.model.sample_rate)


class GPTSoVITSAdapter:
    def __init__(self, models, repo):
        root = Path(models).resolve()
        values = {key: str(_require(root / relative, directory=key.endswith('base_path')))
                  for key, relative in GPT_FILES.items()}
        # Chinese G2PW reads this environment key at module import time.
        os.environ['bert_path'] = values['bert_base_path']
        sys.path.insert(0, str(Path(repo).resolve() / 'GPT_SoVITS'))
        from TTS_infer_pack.TTS import TTS, TTS_Config
        config = TTS_Config({'custom': dict(values, device='cuda', is_half=True, version='v2')})
        if str(config.device) == 'cpu':
            raise RuntimeError('GPT-SoVITS 未启用 GPU；不会自动回退 CPU')
        # Upstream saves its configuration on weight load. Keep those writes in a
        # worker-owned temporary directory, never rewrite the engine/user config.
        self._temporary = tempfile.TemporaryDirectory(prefix='vc-gpt-sovits-')
        config.configs_path = str(Path(self._temporary.name) / 'tts_infer.yaml')
        self.model = TTS(config)

    def synthesize(self, request):
        reference = _reference(request)
        language = GPT_LANGUAGES.get(request.get('language', '中文'))
        if language is None:
            raise RuntimeError('GPT-SoVITS v2 不支持所选语言')
        if language == 'ko':
            raise RuntimeError('当前 GPT-SoVITS 配方不支持韩语文本前端，请使用中文、英文或日语')
        transcript = (request.get('reference_text') or '').strip()
        # Reference language is independent of target language (e.g. English
        # lecturer cloned into Chinese). Upstream auto segments mixed transcripts.
        prompt_language = GPT_LANGUAGES.get(request.get('reference_language', 'auto'))
        if prompt_language is None:
            raise RuntimeError('GPT-SoVITS v2 不支持参考文本语言')
        options = dict(text=request['text'], text_lang=language,
            ref_audio_path=reference, prompt_text=transcript, prompt_lang=prompt_language,
            seed=int(request.get('seed', -1)), text_split_method='cut0', batch_size=1,
            split_bucket=False, speed_factor=1.0, fragment_interval=0.0,
            return_fragment=False, streaming_mode=False, parallel_infer=False,
            use_cuda_graph=False, use_flash_attention=False, super_sampling=False)
        chunks, rate = [], None
        # Some upstream exception handlers yield silence before re-raising: never
        # use next() and report success; consume the complete generator first.
        for sample_rate, audio in self.model.run(options):
            if rate is not None and rate != sample_rate:
                raise RuntimeError('GPT-SoVITS 返回了不一致的采样率')
            rate = sample_rate
            chunks.append(audio)
        _write_audio(request['output_path'], chunks, rate or 0)


def create_adapter(engine, models, repo):
    if engine == 'cosyvoice3':
        return CosyVoice3Adapter(models, repo)
    if engine in ('gpt-sovits', 'gpt_sovits'):
        return GPTSoVITSAdapter(models, repo)
    raise ValueError('未知本地配音引擎：' + str(engine))
