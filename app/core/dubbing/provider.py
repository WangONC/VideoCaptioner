"""A persistent JSON-lines subprocess, keeping torch outside the GUI process."""
from pathlib import Path
import hashlib
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from .pipeline import DubbingError, CancelledError, _check, _hash

_PREFIX = 'DUBBING_JSON:'


class IndexTTS2Provider:
    def __init__(self, python_path, repo_path, model_dir, timeout=1200, engine="indextts2"):
        if not all((python_path, repo_path, model_dir)):
            raise DubbingError('Configure the dedicated IndexTTS2 Python, repository and model directory first')
        if engine not in ('indextts2','qwen3'):raise DubbingError('Unknown TTS engine')
        self.engine=engine
        self.reference_text=''
        self.python = Path(python_path).expanduser().resolve()
        self.repo = Path(repo_path).expanduser().resolve()
        self.models = Path(model_dir).expanduser().resolve()
        source_ok=(self.repo / ('indextts/infer_v2.py' if engine=='indextts2' else 'qwen_tts/__init__.py')).is_file()
        config_ok=(self.models / ('config.yaml' if engine=='indextts2' else 'config.json')).is_file()
        if not self.python.is_file() or not source_ok or not config_ok:
            raise DubbingError('IndexTTS2 runtime or config.yaml is missing. Model installation is required.')
        self.timeout = timeout
        self.process = None
        self._errors = None
        self._responses = queue.Queue()
        self._fingerprint = None

    def fingerprint(self):
        if self._fingerprint is None:
            digest = hashlib.sha256()
            # Include model bytes and inference source, not merely a configurable path.
            for root, paths in ((self.models, sorted(p for p in self.models.rglob('*') if p.is_file())),
                                (self.repo, sorted((self.repo / ('indextts' if self.engine=='indextts2' else 'qwen_tts')).rglob('*.py')))):
                for path in paths:
                    digest.update(str(path.relative_to(root)).encode())
                    digest.update(_hash(path).encode())
            digest.update(_hash(Path(__file__).with_name('worker.py')).encode())
            self._fingerprint = {'provider': self.engine+'-offline-v1', 'runtime': str(self.python),
                                 'runtime_mtime_ns': self.python.stat().st_mtime_ns,
                                 'models_and_code_sha256': digest.hexdigest(), 'fp16': True,
                                 'cuda_kernel': False, 'deepspeed': False}
        return self._fingerprint

    def _start(self):
        self.close()
        self._responses = queue.Queue()
        responses = self._responses
        self._errors = tempfile.TemporaryFile(mode='w+b')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        self.process = subprocess.Popen([str(self.python), '-u', str(Path(__file__).with_name('worker.py')),
                                         '--repo', str(self.repo), '--models', str(self.models), '--engine', self.engine],
                                        cwd=self.repo, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self._errors, text=True, encoding='utf-8', errors='replace', bufsize=1)
        process = self.process
        def read():
            try:
                for line in process.stdout:
                    if line.startswith(_PREFIX):
                        try:
                            responses.put(json.loads(line[len(_PREFIX):]))
                        except ValueError:
                            responses.put({'error': 'Invalid response from IndexTTS2 worker'})
            finally:
                responses.put({'error': 'IndexTTS2 worker exited'})
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
        raise DubbingError('IndexTTS2 worker timed out; completed cue audio remains cached')

    def synthesize(self, text, reference_audio, output_path, cancel_event=None):
        try:
            _check(cancel_event)
            if self.process is None or self.process.poll() is not None:
                self._start()
                ready = self._receive(cancel_event)
                if not ready.get('ready'):
                    raise DubbingError('IndexTTS2 worker did not initialize')
            request = {'text': text, 'reference_audio': reference_audio, 'output_path': output_path, 'reference_text':self.reference_text}
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
            self.process.stdin.flush()
            result = self._receive(cancel_event)
            if not result.get('ok'):
                raise DubbingError('IndexTTS2 worker returned an invalid completion')
        except Exception:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
            self.process = None
        if self._errors is not None:
            self._errors.close(); self._errors = None


class Qwen3TTSProvider(IndexTTS2Provider):
    def __init__(self, python_path, repo_path, model_dir, timeout=1200):
        super().__init__(python_path,repo_path,model_dir,timeout,engine="qwen3")
