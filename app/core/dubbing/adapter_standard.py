"""Pinned-upstream inference calls, imported only in a dedicated worker.

Sources: QwenLM/Qwen3-TTS qwen_tts/inference/qwen3_tts_model.py
(022e286b98fbec7e1e916cb940cdf532cd9f488e); dots runtime.py
(b65ca7149fb43b8fb8d52916aa12738deda7b188); rany2/edge-tts
(bd44c1d1aaebdc1db7f47456ee12431b42ec7cac).
No model downloads or CPU fallback are performed here.
"""
from pathlib import Path
import asyncio
import gc
import hashlib
import json
import os
import subprocess
import tempfile

LANGUAGES = {'中文': 'Chinese', '英语': 'English', '英文': 'English', '日语': 'Japanese',
             '韩语': 'Korean', '自动': 'Auto', '自动检测': 'Auto'}


def language_name(value):
    return LANGUAGES.get(value, value or 'Chinese')


def write_audio(path, audio, sample_rate):
    import soundfile as sf
    if hasattr(audio, 'detach'):
        audio = audio.detach().float().cpu().squeeze().numpy()
    sf.write(path, audio, sample_rate, subtype='PCM_16')


class StandardAdapter:
    def __init__(self, engine, models, repo, device='cuda'):
        self.engine = engine
        self.models = Path(models)
        self.repo = Path(repo)
        self.prompt_cache = {}
        if engine == 'edge-tts':
            import edge_tts
            # edge-tts 7.3.1 pins certifi-only contexts. Also honor the OS /
            # explicit SSL_CERT_FILE trust store (e.g. an installed enterprise
            # proxy CA), while keeping hostname and certificate verification.
            import ssl
            for name in ('voices', 'communicate'):
                context = getattr(getattr(edge_tts, name, None), '_SSL_CTX', None)
                if context is not None:
                    context.load_default_certs()
                    if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
                        raise RuntimeError('Edge TTS TLS 证书校验未启用')
            self.edge_tts = edge_tts
            self.voices = None
            return
        import torch
        self.torch = torch
        self.device = device
        if device != 'cpu':
            torch.cuda.set_device(torch.device(device))
        if engine.startswith('qwen3'):
            from qwen_tts import Qwen3TTSModel
            self.qwen_class = Qwen3TTSModel
            self.model = None if engine == 'qwen3-design' else self._load_qwen(self.models)
        elif engine == 'dots':
            from dots_tts.runtime import DotsTtsRuntime
            self.model = DotsTtsRuntime.from_pretrained(str(self.models), precision='float32' if self.device == 'cpu' else 'float16', optimize=False)
            self.model.device = torch.device(self.device)
        elif engine == 'indextts2':
            from indextts.infer_v2 import IndexTTS2
            auxiliary = self.models / 'hf_cache'
            self.model = IndexTTS2(cfg_path=str(self.models / 'config.yaml'), model_dir=str(self.models),
                use_fp16=self.device != 'cpu', device=self.device, use_cuda_kernel=False, use_deepspeed=False, use_qwen_emo=False,
                aux_paths=dict(w2v_bert=str(auxiliary/'w2v-bert-2.0'),
                    semantic_codec=str(auxiliary/'semantic_codec_model.safetensors'),
                    campplus=str(auxiliary/'campplus_cn_common.bin'), bigvgan=str(auxiliary/'bigvgan')))
        else:
            raise ValueError('Unknown inference engine: ' + engine)

    def synthesize(self, request):
        text, output = request['text'], request['output_path']
        if self.engine == 'edge-tts':
            # Only text and voice go to Microsoft's online service. Reference
            # paths/text are intentionally not read or transmitted.
            voice = request.get('voice')
            if not voice:
                raise ValueError('请选择 Edge TTS 音色')
            with tempfile.TemporaryDirectory(prefix='edge-tts-', dir=Path(output).parent) as directory:
                mp3 = Path(directory) / 'speech.mp3'
                async def generate():
                    if self.voices is None:
                        self.voices = {item['ShortName']: item for item in await self.edge_tts.list_voices()}
                    if voice not in self.voices:
                        raise ValueError('Edge TTS 音色不可用，请重新选择：' + voice)
                    language = language_name(request.get('language'))
                    expected = {'Chinese':'zh', 'English':'en', 'Japanese':'ja', 'Korean':'ko'}.get(language)
                    locale = self.voices[voice].get('Locale', '').lower()
                    if expected and not locale.startswith(expected + '-'):
                        raise ValueError('Edge TTS 音色与目标语言不一致，请选择对应语言音色')
                    await self.edge_tts.Communicate(text=text, voice=voice).save(str(mp3))
                asyncio.run(generate())
                result = subprocess.run([request.get('ffmpeg') or 'ffmpeg', '-v', 'error', '-y', '-i', str(mp3),
                    '-ac', '1', '-ar', '24000', '-c:a', 'pcm_s16le', output],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                if result.returncode:
                    raise RuntimeError(result.stderr.decode('utf-8', 'replace')[-2000:])
            return
        seed = int(request.get('seed', -1))
        if seed >= 0:
            import random
            import numpy as np
            random.seed(seed); np.random.seed(seed % (2**32)); self.torch.manual_seed(seed)
        language = language_name(request.get('language'))
        reference = request.get('reference_audio')
        prompt_text = request.get('reference_text') or None
        if self.engine == 'indextts2':
            self.model.infer(spk_audio_prompt=reference, text=text, output_path=output, verbose=False)
        elif self.engine == 'qwen3':
            key = (reference, prompt_text)
            if key not in self.prompt_cache:
                self.prompt_cache[key] = self.model.create_voice_clone_prompt(ref_audio=reference,
                    ref_text=prompt_text, x_vector_only_mode=prompt_text is None)
            wavs, rate = self.model.generate_voice_clone(text=text, language=language,
                voice_clone_prompt=self.prompt_cache[key])
            if len(wavs) != 1:
                raise RuntimeError('Unexpected TTS output count')
            write_audio(output, wavs[0], rate)
        elif self.engine == 'qwen3-custom':
            wavs, rate = self.model.generate_custom_voice(text=text, language=language,
                speaker=request.get('voice', ''), instruct=request.get('style') or None)
            if len(wavs) != 1:
                raise RuntimeError('Unexpected TTS output count')
            write_audio(output, wavs[0], rate)
        elif self.engine == 'qwen3-design':
            self._design_clone(request, language)
        elif self.engine == 'dots':
            if not reference:
                raise ValueError('dots.tts 需要参考音频')
            language = {'Chinese':'zh', 'English':'en', 'Japanese':'ja', 'Korean':'ko', 'Auto':'auto_detect'}.get(language, language)
            result = self.model.generate(text=text, prompt_audio_path=reference, prompt_text=prompt_text,
                                         language=language, normalize_text=False)
            write_audio(output, result['audio'], result['sample_rate'])

    def _load_qwen(self, path):
        return self.qwen_class.from_pretrained(str(path), device_map=self.device,
            dtype=self.torch.float32 if self.device == 'cpu' else self.torch.float16, attn_implementation='sdpa', local_files_only=True)

    def _design_clone(self, request, language):
        description = request.get('description', '').strip()
        if not description:
            raise ValueError('请输入声音描述')
        identity = request.get('voice_identity')
        if not identity or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
            raise ValueError('声音设计缺少固定模型身份')
        base = self.models / 'voice_clone_base'
        if not (base / 'config.json').is_file():
            raise RuntimeError('声音设计需要配套 Base 克隆模型，请重新下载完整模型')
        if identity not in self.prompt_cache:
            cache = self.repo.parent / 'voice-design-cache'
            cache.mkdir(parents=True, exist_ok=True)
            reference, metadata = cache / (identity + '.wav'), cache / (identity + '.json')
            stored = None
            try:
                stored = json.loads(metadata.read_text('utf-8'))
                if stored['sha256'] != hashlib.sha256(reference.read_bytes()).hexdigest():
                    stored = None
            except (OSError, ValueError, KeyError, TypeError):
                stored = None
            if stored is None:
                reference_text = {
                    'Chinese': '你好，欢迎来到今天的课程。我们将用清晰自然的声音，一步一步地解释重要概念，让学习变得更加轻松。',
                    'English': 'Welcome to today’s lesson. We will explain the important ideas clearly and naturally, one step at a time, so that everyone can follow along.',
                    'Japanese': '今日のレッスンへようこそ。大切な考え方を、自然でわかりやすい声で、一つずつ丁寧に説明していきます。',
                    'Korean': '오늘 수업에 오신 것을 환영합니다. 중요한 개념을 자연스럽고 명확한 목소리로 하나씩 차근차근 설명하겠습니다.',
                }.get(language, '你好，欢迎来到今天的课程。我们将用清晰自然的声音，一步一步地解释重要概念，让学习变得更加轻松。')
                design = self._load_qwen(self.models)
                wavs, rate = design.generate_voice_design(text=reference_text, language=language, instruct=description)
                if len(wavs) != 1:
                    raise RuntimeError('Unexpected TTS output count')
                # Cache one designed voice, then use Base for every subtitle cue.
                # A seed alone cannot guarantee identity across independent designs.
                with tempfile.TemporaryDirectory(prefix='.voice-', dir=cache) as directory:
                    temporary = Path(directory) / 'reference.wav'
                    write_audio(temporary, wavs[0], rate)
                    stored = dict(text=reference_text, sha256=hashlib.sha256(temporary.read_bytes()).hexdigest())
                    info = Path(directory) / 'reference.json'
                    info.write_text(json.dumps(stored, ensure_ascii=False), 'utf-8')
                    os.replace(temporary, reference); os.replace(info, metadata)
                del design, wavs
                gc.collect()
                self.torch.cuda.empty_cache()
            if self.model is None:
                self.model = self._load_qwen(base)
            self.prompt_cache[identity] = self.model.create_voice_clone_prompt(
                ref_audio=str(reference), ref_text=stored['text'], x_vector_only_mode=False)
        wavs, rate = self.model.generate_voice_clone(text=request['text'], language=language,
            voice_clone_prompt=self.prompt_cache[identity])
        if len(wavs) != 1:
            raise RuntimeError('Unexpected TTS output count')
        write_audio(request['output_path'], wavs[0], rate)
