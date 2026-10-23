"""A persistent JSON-lines subprocess, keeping torch outside the GUI process."""
from pathlib import Path
import hashlib
import json
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from .pipeline import DubbingError, CancelledError, _check, _hash

_PREFIX = 'DUBBING_JSON:'
SOURCE_FILES = {'indextts2':'indextts/infer_v2.py', 'qwen3':'qwen_tts/__init__.py',
    'qwen3-custom':'qwen_tts/__init__.py', 'qwen3-design':'qwen_tts/__init__.py',
    'dots':'src/dots_tts/runtime.py', 'cosyvoice3':'cosyvoice/cli/cosyvoice.py',
    'gpt-sovits':'GPT_SoVITS/TTS_infer_pack/TTS.py', 'edge-tts':None}
CONFIG_FILES = {'indextts2':'config.yaml', 'qwen3':'config.json', 'qwen3-custom':'config.json',
    'qwen3-design':'config.json', 'dots':'config.json', 'cosyvoice3':'cosyvoice3.yaml',
    'gpt-sovits':None, 'edge-tts':None}



class IndexTTS2Provider:
    def __init__(self, python_path, repo_path, model_dir, timeout=1200, engine="indextts2"):
        if not all((python_path, repo_path, model_dir)):
            raise DubbingError('请先配置独立配音环境与模型目录')
        if engine not in SOURCE_FILES: raise DubbingError('Unknown TTS engine')
        self.engine=engine
        self.reference_text=''
        self.python = Path(python_path).expanduser().resolve()
        self.repo = Path(repo_path).expanduser().resolve()
        self.models = Path(model_dir).expanduser().resolve()
        source = SOURCE_FILES[engine]
        source_ok = source is None or (self.repo / source).is_file()
        config_file = CONFIG_FILES[engine]
        config_ok = config_file is None or (self.models / config_file).is_file()
        if not self.python.is_file() or not source_ok or not config_ok:
            raise DubbingError('配音环境或模型文件不完整，请先安装环境并下载模型。')
        self.options = {}
        self.cancel_event = None
        self.timeout = timeout
        self.process = None
        self._errors = None
        self._responses = queue.Queue()
        self._fingerprint = None
        self._voice_identity = None
        self._design_reference_hash = None

    def _file_hash(self, path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                _check(self.cancel_event)
                digest.update(chunk)
        _check(self.cancel_event)
        return digest.hexdigest()

    def fingerprint(self):
        if self._fingerprint is None:
            digest = hashlib.sha256()
            # Include model bytes and inference source, not merely a configurable path.
            for root, paths in ((self.models, sorted(p for p in self.models.rglob('*') if p.is_file())),
                                (self.repo, sorted(self.repo.rglob('*.py')))):
                for path in paths:
                    digest.update(str(path.relative_to(root)).encode())
                    digest.update(self._file_hash(path).encode())
            installation = self.repo / '.videocaptioner-source.json'
            if installation.is_file():
                digest.update(self._file_hash(installation).encode())
            for adapter in sorted(Path(__file__).parent.glob('adapter_*.py')) + [Path(__file__).with_name('worker.py')]:
                digest.update(self._file_hash(adapter).encode())
            self._fingerprint = {'provider': self.engine+('-online-v1' if self.engine == 'edge-tts' else '-offline-v2'), 'runtime': str(self.python),
                                 'runtime_mtime_ns': self.python.stat().st_mtime_ns,
                                 'models_and_code_sha256': digest.hexdigest(), 'fp16': self.engine != 'edge-tts',
                                 'cuda_kernel': False, 'deepspeed': False, 'options': dict(self.options)}
            if self.engine == 'qwen3-design':
                self._voice_identity = hashlib.sha256(json.dumps(self._fingerprint, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                self._design_reference_hash = self._read_design_reference_hash()
                # First generation has no reference yet: use a unique cache namespace.
                # On the next job the persisted WAV hash replaces this nonce; a one-time
                # cache miss is preferable to mixing old and newly designed voices.
                self._fingerprint['design_reference'] = (self._design_reference_hash or 'new:' + uuid.uuid4().hex)
        return self._fingerprint

    def _read_design_reference_hash(self):
        if not self._voice_identity:
            return None
        cache = self.repo.parent / 'voice-design-cache'
        try:
            metadata = json.loads((cache / (self._voice_identity + '.json')).read_text('utf-8'))
            digest = self._file_hash(cache / (self._voice_identity + '.wav'))
            return digest if digest == metadata.get('sha256') and metadata.get('text') else None
        except (OSError, ValueError, TypeError):
            return None

    def _start(self):
        self.close()
        self._responses = queue.Queue()
        responses = self._responses
        self._errors = tempfile.TemporaryFile(mode='w+b')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        self.process = subprocess.Popen([str(self.python), '-u', str(Path(__file__).with_name('worker.py')),
                                         '--repo', str(self.repo), '--models', str(self.models), '--engine', self.engine],
                                        cwd=self.repo if self.repo.is_dir() else self.python.parent, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self._errors, text=True, encoding='utf-8', errors='replace', bufsize=1,
                                        start_new_session=os.name != 'nt',
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        process = self.process
        def read():
            try:
                for line in process.stdout:
                    if line.startswith(_PREFIX):
                        try:
                            responses.put(json.loads(line[len(_PREFIX):]))
                        except ValueError:
                            responses.put({'error': 'Invalid response from TTS worker'})
            finally:
                responses.put({'error': 'TTS worker exited'})
        threading.Thread(target=read, daemon=True).start()

    def _receive(self, cancel):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            _check(cancel)
            try:
                response = self._responses.get(timeout=.1)
            except queue.Empty:
                continue
            if 'error' in response:
                if self._errors:
                    self._errors.seek(0, 2)
                    size = self._errors.tell(); self._errors.seek(max(0, size - 3000))
                    details = self._errors.read().decode('utf-8', 'replace')
                else:
                    details = ''
                raise DubbingError(response['error'] + ('\n' + details if details else ''))
            return response
        raise DubbingError('TTS worker timed out; completed cue audio remains cached')

    def synthesize(self, text, reference_audio, output_path, cancel_event=None):
        try:
            _check(cancel_event)
            if self.process is None or self.process.poll() is not None:
                self._start()
                ready = self._receive(cancel_event)
                if not ready.get('ready'):
                    raise DubbingError('TTS worker did not initialize')
            if self.engine == 'qwen3-design':
                self.fingerprint()
                identity = self._voice_identity
                if self._design_reference_hash and self._read_design_reference_hash() != self._design_reference_hash:
                    raise DubbingError('固定声音参考已删除或损坏，请重新开始任务以避免混用声音缓存')
            request = {'text': text, 'reference_audio': reference_audio, 'output_path': output_path, 'reference_text':self.reference_text, **self.options}
            if self.engine == 'qwen3-design':
                request['voice_identity'] = identity
            if self.engine == 'edge-tts':
                request.pop('reference_audio'); request.pop('reference_text')
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
            self.process.stdin.flush()
            result = self._receive(cancel_event)
            if not result.get('ok'):
                raise DubbingError('TTS worker returned an invalid completion')
            if self.engine == 'qwen3-design':
                current_hash = self._read_design_reference_hash()
                if not current_hash or (self._design_reference_hash and current_hash != self._design_reference_hash):
                    raise DubbingError('固定声音参考校验失败，未接受本句输出')
                self._design_reference_hash = current_hash
        except Exception:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                else:
                    try: os.killpg(self.process.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if os.name != 'nt':
                        try: os.killpg(self.process.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                    else: self.process.kill()
                    self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
            self.process = None
        if self._errors is not None:
            self._errors.close(); self._errors = None


class Qwen3TTSProvider(IndexTTS2Provider):
    def __init__(self, python_path, repo_path, model_dir, timeout=1200):
        super().__init__(python_path,repo_path,model_dir,timeout,engine="qwen3")


def create_provider(model_id, python_path, repo_path, model_dir, *, language='中文',
                    voice='', description='', seed=-1, style='', ffmpeg='ffmpeg', cancel_event=None):
    engines = {'indextts2':'indextts2', 'qwen3-0.6b':'qwen3', 'qwen3-1.7b':'qwen3',
        'qwen3-custom':'qwen3-custom', 'qwen3-design':'qwen3-design', 'dots-soar':'dots',
        'dots-mf':'dots', 'cosyvoice3':'cosyvoice3', 'gpt-sovits':'gpt-sovits', 'edge-tts':'edge-tts'}
    if model_id not in engines:
        raise DubbingError('未接入的配音模型：' + model_id)
    _check(cancel_event)
    if model_id != 'edge-tts' and (Path(model_dir) / 'videocaptioner-models.json').is_file():
        from app.core.tts_runtime import verify_models
        try:
            valid = verify_models(model_dir, model_id, cancel_event, full=True, repo_path=repo_path)
        except Exception as error:
            _check(cancel_event)
            raise DubbingError('模型完整性检查失败：' + str(error)) from error
        _check(cancel_event)
        if not valid:
            raise DubbingError('模型清单或文件校验失败，请重新下载完整模型')
    provider = IndexTTS2Provider(python_path, repo_path, model_dir, engine=engines[model_id])
    provider.cancel_event = cancel_event
    provider.options = dict(language=language, voice=voice, description=description,
                            seed=int(seed), ffmpeg=str(ffmpeg), model_id=model_id)
    if style:
        provider.options['style'] = style
    return provider
