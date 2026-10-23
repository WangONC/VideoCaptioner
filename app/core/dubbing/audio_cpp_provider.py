"""Cancellable audio.cpp CLI adapter, isolated from the GUI's Python runtime.

Uses the v0.9.2 CLI request-sequence contract. One UTF-8 JSON request per cue
avoids putting Chinese speech/transcripts in a Windows narrow command line.
This adapter does not install anything, start a server, or download models.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import wave

from .pipeline import DubbingError, _check, _hash


from app.core.audio_cpp_runtime import AUDIO_CPP_MODELS

_MODELS = {key: (value['family'], value['filename']) for key, value in AUDIO_CPP_MODELS.items()}
_LANGUAGE = {'中文': 'zh', '英语': 'en', '英文': 'en', '日语': 'ja', '韩语': 'ko', '自动': 'auto'}


def create_provider(config, cancel_event=None):
    from app.core.audio_cpp_runtime import models_ready
    if not models_ready(dict(model_id=config.model_id, model_dir=config.model_dir), cancel_event, verify_hash=True):
        raise DubbingError('audio.cpp GGUF 模型不完整，请重新下载或检查模型')
    if config.model_id == 'qwen3-design':
        return DesignedAudioCppProvider(config, cancel_event)
    provider = AudioCppProvider(config.python_path, config.model_dir, config.model_id,
                                language=getattr(config, 'language', '中文'),
                                voice=getattr(config, 'voice', ''),
                                description=getattr(config, 'description', ''),
                                seed=getattr(config, 'seed', -1))
    provider.reference_text = config.reference_text or ''
    provider.cancel_event = cancel_event
    return provider



class AudioCppProvider:
    """An offline CLI invocation per subtitle, with no silent backend fallback.

    Runtime/device and package qualification belongs to the managed installer.
    Construction only checks local file shape; it is not GPU/model qualification.
    """

    def __init__(self, executable, model_dir, model_id, compute_backend='vulkan', timeout=1200,
                 language='中文', voice='', description='', seed=-1):
        if model_id not in _MODELS:
            raise DubbingError('Unknown audio.cpp model')
        if compute_backend not in ('vulkan', 'cuda', 'cpu', 'metal'):
            raise DubbingError('Unsupported audio.cpp compute backend')
        if not executable or not model_dir:
            raise DubbingError('Install the audio.cpp environment and model first')
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise DubbingError('audio.cpp timeout must be positive and finite')
        self.executable = Path(executable).expanduser().resolve()
        self.models = Path(model_dir).expanduser().resolve()
        self.model_id = model_id
        self.family, filename = _MODELS[model_id]
        self.model = self.models if self.models.is_file() else self.models / filename
        self.backend, self.timeout = compute_backend, timeout
        self.reference_text = ''
        self.language = _LANGUAGE.get(language, language)
        if model_id == 'indextts2' and self.language not in ('zh', 'en'):
            raise DubbingError('IndexTTS2 仅支持中文和英语')
        self.voice, self.description, self.seed = voice, description, seed
        self.requires_reference = model_id not in ('qwen3-custom', 'qwen3-design')
        self.process = None
        self.cancel_event = None
        self._synthesis_lock = threading.Lock()
        if not self.executable.is_file() or not self.model.is_file():
            raise DubbingError('audio.cpp executable or compatible GGUF model is missing; install them first')
        with self.model.open('rb') as stream:
            if stream.read(4) != b'GGUF':
                raise DubbingError('audio.cpp requires its compatible GGUF package, not PyTorch weights')

    def fingerprint(self):
        """Include actual file bytes, not merely a selected path or package name."""
        from app.core.audio_cpp_runtime import _digest
        def checked_hash(path):
            return _digest(path, self.cancel_event)
        digest = hashlib.sha256()
        files = [self.model] if self.models.is_file() else sorted(p for p in self.models.rglob('*') if p.is_file())
        for path in files:
            relative = path.name if self.models.is_file() else path.relative_to(self.models).as_posix()
            digest.update(relative.encode('utf-8'))
            digest.update(checked_hash(path).encode('ascii'))
        # Runtime DLLs can change the inference result even if the CLI is unchanged.
        libraries = sorted(p for p in self.executable.parent.rglob('*') if p.is_file()
                           and (p.suffix.lower() in ('.dll', '.dylib') or '.so' in p.name))
        runtime = hashlib.sha256(checked_hash(self.executable).encode('ascii'))
        for path in libraries:
            runtime.update(path.relative_to(self.executable.parent).as_posix().encode('utf-8'))
            runtime.update(checked_hash(path).encode('ascii'))
        return {'provider': 'audio.cpp-cli-v0.9.2-contract-v1', 'model_id': self.model_id,
                'family': self.family, 'compute_backend': self.backend, 'language': self.language,
                'voice': self.voice, 'description': self.description, 'seed': self.seed,
                'output_format': 'pcm16', 'model_sha256': digest.hexdigest(),
                'runtime_sha256': runtime.hexdigest(), 'adapter_sha256': _hash(Path(__file__)),
                'reference_mode': 'transcript' if self.reference_text else 'speaker_embedding'}

    @staticmethod
    def _validate_wav(path):
        try:
            with wave.open(str(path), 'rb') as stream:
                channels, width, rate, frames = (stream.getnchannels(), stream.getsampwidth(),
                                               stream.getframerate(), stream.getnframes())
                if channels <= 0 or width != 2 or rate <= 0 or frames <= 0:
                    raise DubbingError('audio.cpp produced empty or non-PCM16 audio')
                remaining = frames * channels * width
                while remaining:
                    block = stream.readframes(min(frames, 65536))
                    if not block:
                        raise DubbingError('audio.cpp produced a truncated WAV')
                    remaining -= len(block)
        except (OSError, EOFError, wave.Error) as error:
            raise DubbingError('audio.cpp did not produce a valid PCM WAV: ' + str(error)) from error

    @staticmethod
    def _tail(log):
        log.seek(0, 2)
        log.seek(max(0, log.tell() - 6000))
        return log.read().decode('utf-8', 'replace').strip()

    def synthesize(self, text, reference_audio, output_path, cancel_event=None):
        if not self._synthesis_lock.acquire(blocking=False):
            raise DubbingError('audio.cpp is already synthesizing a cue')
        try:
            self._synthesize(text, reference_audio, output_path, cancel_event)
        finally:
            self.close()
            self._synthesis_lock.release()

    def _synthesize(self, text, reference_audio, output_path, cancel_event):
        _check(cancel_event)
        if not isinstance(text, str) or not text.strip() or '\0' in text:
            raise DubbingError('Speech text must be nonempty and contain no NUL')
        reference = Path(reference_audio).expanduser().resolve() if reference_audio else None
        output = Path(output_path).expanduser().resolve()
        if self.requires_reference and (reference is None or not reference.is_file()):
            raise DubbingError('Reference audio is missing')
        if output.exists() or output in (reference, self.executable, self.model):
            raise DubbingError('Refusing to overwrite an existing output or input file')
        if self.models.is_dir() and output.is_relative_to(self.models):
            raise DubbingError('Output must not be written inside the model package')
        output.parent.mkdir(parents=True, exist_ok=True)
        # Put staging on the output volume so hard-link publication is exclusive
        # and atomic. Existing outputs, including late-created ones, are untouched.
        with tempfile.TemporaryDirectory(prefix='.audiocpp-', dir=output.parent) as folder:
            stage = Path(folder)
            request = {'id': 'cue', 'text': text, 'language': self.language}
            task = 'clon' if self.requires_reference and self.family != 'dots_tts' else 'tts'
            if self.requires_reference:
                request['voice_ref'] = str(reference)
                if self.reference_text:
                    request['reference_text'] = self.reference_text
                elif self.family == 'qwen3_tts':
                    request['options'] = {'x_vector_only_mode': 'true'}
            elif self.model_id == 'qwen3-custom':
                if not self.voice:
                    raise DubbingError('Choose a packaged Qwen CustomVoice speaker')
                request['voice_id'] = self.voice
            elif self.model_id == 'qwen3-design':
                if not self.description.strip():
                    raise DubbingError('Enter a Qwen VoiceDesign description')
                request['instruct'] = self.description
                task = 'vdes'
            if self.seed != -1:
                if not isinstance(self.seed, int) or self.seed < 0:
                    raise DubbingError('Seed must be -1 or a nonnegative integer')
                request['seed'] = self.seed
            elif self.family == 'dots_tts':
                import secrets
                request['seed'] = secrets.randbelow(2**31)
            if self.model_id == 'dots-mf':
                request.setdefault('options', {})['num_inference_steps'] = '4'
            sequence = stage / 'request.json'
            sequence.write_text(json.dumps({'requests': [request]}, ensure_ascii=False), encoding='utf-8')
            command = [str(self.executable), '--task', task, '--family', self.family,
                       '--model', str(self.model), '--backend', self.backend,
                       '--mode', 'offline', '--language', self.language, '--request-sequence', str(sequence),
                       '--out-dir', str(stage), '--out-format', 'pcm16']
            # No shell is involved; text and reference transcript live in JSON.
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            self._run_command(command, env, cancel_event)
            generated = stage / 'cue.wav'
            self._validate_wav(generated)
            _check(cancel_event)
            try:
                if os.name == 'nt':
                    os.rename(generated, output)
                else:
                    os.link(generated, output)
            except FileExistsError as error:
                raise DubbingError('Output appeared while synthesizing; it was not overwritten') from error
            except OSError as error:
                raise DubbingError('Cannot publish WAV without overwriting files: ' + str(error)) from error

    def _run_command(self, command, env, cancel_event):
        with tempfile.TemporaryFile(mode='w+b') as log:
            try:
                try:
                    self.process = subprocess.Popen(command, cwd=self.executable.parent,
                        env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except OSError as error:
                    raise DubbingError('Could not start audio.cpp: ' + str(error)) from error
                process = self.process
                deadline = time.monotonic() + self.timeout
                while process.poll() is None:
                    _check(cancel_event)
                    if time.monotonic() >= deadline:
                        raise DubbingError('audio.cpp timed out; completed cue audio remains cached')
                    time.sleep(.05)
                _check(cancel_event)
                if process.returncode:
                    raise DubbingError(f'audio.cpp exited with code {process.returncode}:\n' + self._tail(log))
            finally:
                # Stop before the log and staging directory are closed/deleted.
                self.close()

    def close(self):
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class DesignedAudioCppProvider:
    """Generate one durable reference voice, then clone it with Qwen Base.

    Seed alone does not fix identity. The actual generated reference WAV is
    cached across jobs and hashed into the synthesis cache identity.
    """
    requires_reference = False
    reference_text = ''
    _PROMPTS = {
        'zh': '你好，欢迎来到今天的课程。我们会一步一步讲解，让每一个知识点都清楚易懂。',
        'en': 'Hello, and welcome to this lesson. We will explain each idea step by step so that everything is clear and easy to understand.',
        'ja': 'こんにちは。今日の授業へようこそ。一つ一つの考え方を、わかりやすく順番に説明していきます。',
        'ko': '안녕하세요. 오늘 수업에 오신 것을 환영합니다. 모든 내용을 이해하기 쉽도록 차근차근 설명하겠습니다.',
    }

    def __init__(self, config, cancel_event=None):
        self.cancel_event = cancel_event
        self.language = _LANGUAGE.get(config.language, config.language)
        if self.language not in self._PROMPTS:
            raise DubbingError('Unsupported language for the designed reference voice')
        self.designer = AudioCppProvider(config.python_path, config.model_dir, 'qwen3-design',
            language=config.language, description=config.description, seed=config.seed)
        self.cloner = AudioCppProvider(config.python_path, config.model_dir, 'qwen3-1.7b',
            language=config.language, seed=config.seed)
        self.designer.cancel_event = self.cloner.cancel_event = cancel_event
        self.prompt_text = self._PROMPTS[self.language]
        self.cloner.reference_text = self.prompt_text
        self.cache_dir = Path(config.cache_dir).expanduser().resolve() / 'designed-voices' / 'audio.cpp'
        self.reference = None
        self._fingerprint = None

    def _prepare_reference(self):
        if self.reference is not None:
            return
        _check(self.cancel_event)
        identity = {'designer': self.designer.fingerprint(), 'cloner': self.cloner.fingerprint(),
                    'prompt_text': self.prompt_text, 'schema': 1}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        reference = self.cache_dir / (key + '.wav')
        if not reference.exists():
            self.designer.synthesize(self.prompt_text, None, reference, self.cancel_event)
        self.designer._validate_wav(reference)
        self.reference = reference
        self._fingerprint = dict(identity, reference_sha256=_hash(reference), provider='audio.cpp-design-then-clone-v1')

    def fingerprint(self):
        self._prepare_reference()
        return self._fingerprint

    def synthesize(self, text, reference_audio, output_path, cancel_event=None):
        self.cancel_event = cancel_event if cancel_event is not None else self.cancel_event
        self._prepare_reference()
        self.cloner.synthesize(text, str(self.reference), output_path, self.cancel_event)

    def close(self):
        self.designer.close()
        self.cloner.close()
