"""Managed isolated TTS setup. Downloads are explicit; no driver changes.

Windows recipes are implementation candidates, NOT hardware qualification. A
runtime is usable only after its own GPU tensor probe succeeds. No driver edits.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
from app.core.tts_catalog import MODELS, SOURCES, MODEL_REVISIONS
QWEN_SOURCE_REVISION = '022e286b98fbec7e1e916cb940cdf532cd9f488e'
SOURCE_REVISION = '4f8792ff120cd3ea470dd511e997a17c86cddd10'
PYTHON_URL = 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip'
PYTHON_SHA256 = '608619f8619075629c9c69f361352a0da6ed7e62f83a0e19c63e0ea32eb7629d'
# Keep this separate from the GUI's Python and never install the upstream project
# with dependency resolution: its CUDA pins would replace the AMD torch wheels.
DEPENDENCIES = ['accelerate==1.8.1','cn2an==0.5.22','Cython==3.0.7',
    'descript-audiotools==0.7.2','einops==0.8.2','ffmpeg-python==0.2.0',
    'fugashi==1.5.2','g2p-en==2.1.0','huggingface_hub==0.36.2','jieba==0.42.1',
    'json5==0.10.0','librosa==0.10.2.post1','matplotlib==3.10.0','modelscope==1.27.0',
    'more-itertools==11.1.0','munch==4.0.0','numba==0.63.0','numpy==1.26.4',
    'omegaconf==2.3.1','opencv-python==4.9.0.80','pandas==2.3.2','protobuf==3.19.6',
    'pyloudnorm==0.2.0','safetensors==0.5.2','sentencepiece==0.2.2','soundfile==0.14.0',
    'tensorboard==2.20.0','textstat==0.7.13','tiktoken==0.13.0','tokenizers==0.21.0',
    'tqdm==4.69.1','transformers==4.52.1','unidic-lite==1.0.8','wetext==0.1.4']


class RuntimeSetupError(RuntimeError):
    pass


def default_paths(model_id='indextts2'):
    if model_id not in MODELS:raise RuntimeSetupError('未知配音模型')
    spec=MODELS[model_id]
    base = ROOT/'AppData'/'tts'/spec['engine']
    return dict(python_path=str(base/'runtime'/'python.exe'),repo_path=str(base/'source'),
                model_dir=str(ROOT/'AppData'/'models'/spec['folder']))


def _paths(config=None):
    values=default_paths((config or {}).get("model_id","indextts2"))
    if config: values.update({key:str(config[key]) for key in values if config.get(key)})
    return {key:Path(value).expanduser().resolve() for key,value in values.items()}


def _check(cancel):
    if cancel is not None and cancel.is_set():raise RuntimeSetupError('已取消准备任务。')


def _json_url(url):
    request=urllib.request.Request(url,headers={'User-Agent':'VideoCaptioner-local-dubbing'})
    with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)


def _fetch(url,path,cancel=None,expected=None,progress=None):
    _check(cancel);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+'.download')
    digest=hashlib.sha256()
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'VideoCaptioner-local-dubbing'}),timeout=30) as response, partial.open('wb') as stream:
            total=int(response.headers.get('Content-Length',0));done=0
            while True:
                _check(cancel);chunk=response.read(1024*1024)
                if not chunk:break
                stream.write(chunk);digest.update(chunk);done+=len(chunk)
                if progress:progress(done,total)
        if expected and digest.hexdigest()!=expected:raise RuntimeSetupError('下载校验失败：'+path.name)
        os.replace(partial,path)
        return digest.hexdigest()
    finally:
        partial.unlink(missing_ok=True)


def _extract(archive,directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            target=(directory/item.filename).resolve()
            if not target.is_relative_to(directory) or (item.external_attr>>16)&0o170000==0o120000:
                raise RuntimeSetupError('安装包包含不安全路径。')
        zipped.extractall(directory)


def _command(command,cancel=None,cwd=None,timeout=1800):
    _check(cancel)
    with tempfile.TemporaryFile() as log:
        flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
        process=subprocess.Popen([str(x) for x in command],cwd=cwd,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,creationflags=flags, start_new_session=os.name != 'nt')
        deadline=time.monotonic()+timeout
        try:
            while process.poll() is None:
                _check(cancel)
                if time.monotonic()>deadline:raise RuntimeSetupError('环境准备超时，请查看网络与磁盘空间。')
                time.sleep(.1)
            log.seek(0,2);size=log.tell();log.seek(max(0,size-12000));text=log.read().decode('utf8','replace')
            if process.returncode:raise RuntimeSetupError('命令未完成：\n'+text)
            return text
        finally:
            if process.poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                else:
                    try: os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                try: process.wait(5)
                except subprocess.TimeoutExpired:
                    if os.name != 'nt':
                        try: os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                    else: process.kill()
                    process.wait()


def choose_backend(adapters):
    # Prefer a discrete NVIDIA adapter over an unrelated integrated AMD adapter.
    names=[str(row.get('Name','')) for row in adapters]
    if any('nvidia' in name.lower() for name in names):return 'cuda'
    if any('9070' in name and ('radeon' in name.lower() or 'amd' in name.lower()) for name in names):return 'rocm-gfx120X'
    raise RuntimeSetupError('未找到此版本支持自动配置的显卡。此版本不会自动回退 CPU。')


def detect_backend():
    if platform.system()!='Windows':
        raise RuntimeSetupError('当前自动环境准备面向 Windows。此云端仅可预览界面和测试流程，不能安装 Windows GPU 环境。')
    text=_command(['powershell','-NoProfile','-NonInteractive','-Command',
        'Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,DriverVersion | ConvertTo-Json -Compress'],timeout=30)
    rows=json.loads(text.lstrip('\ufeff'))
    return choose_backend([rows] if isinstance(rows,dict) else rows)


_GPU_PROBE = '''import json,torch
assert torch.cuda.is_available(), 'GPU unavailable; CPU fallback disabled'
x=torch.ones(8,device='cuda'); y=(x+x).sum().item(); torch.cuda.synchronize()
assert y==16
print('VC_GPU:'+json.dumps(dict(device=torch.cuda.get_device_name(0),backend='rocm' if torch.version.hip else 'cuda',torch=torch.__version__)))
'''


def inspect_runtime(config=None, cancel=None):
    paths = _paths(config)
    model_id = (config or {}).get('model_id', 'indextts2')
    engine = MODELS[model_id]['engine']
    models = engine == 'edge-tts' or verify_models(paths['model_dir'], model_id, cancel, repo_path=paths['repo_path'])
    result = dict(ready=False, models_ready=models, device=None, message='尚未准备运行环境')
    marker = SOURCES[engine][2]
    if not paths['python_path'].is_file() or (marker and not (paths['repo_path'] / marker).is_file()):
        return result
    try:
        probe = _runtime_probe(engine, paths['repo_path'])
        text = _command([paths['python_path'], '-c', probe], cancel=cancel, cwd=paths['repo_path'] if marker else None, timeout=90)
        info = json.loads(next(line[7:] for line in text.splitlines() if line.startswith('VC_GPU:')))
        return result | info | dict(ready=True, message='在线语音环境可用' if engine == 'edge-tts' else 'GPU 与基础依赖检查通过，模型推理待实测' if engine == 'gpt-sovits' else 'GPU 与引擎导入检查通过')
    except Exception as error:
        _check(cancel)
        return result | dict(message=str(error))


def _runtime_probe(engine, source):
    # Import the actual adapter dependency, not just torch. No weights are loaded.
    import_paths = [str(source), str(source / 'src'), str(source / 'GPT_SoVITS'), str(source / 'third_party' / 'Matcha-TTS')]
    setup = 'import sys,os;sys.path[:0]=' + repr(import_paths) + ';os.environ[\"NLTK_DATA\"]=' + repr(str(source / '.nltk_data')) + '\n'
    if engine == 'edge-tts':
        return setup + SOURCES[engine][3] + "\nprint('VC_GPU:' + __import__('json').dumps(dict(backend='online', device='Edge TTS')))"
    if engine == 'gpt-sovits':
        # The text front-end may initialize assets during import; preparation
        # checks dependencies without triggering model downloads or loading weights.
        imports = 'import transformers, librosa, soundfile, onnxruntime, peft, nltk, pyopenjtalk, jieba'
        return setup + _GPU_PROBE + '\n' + imports
    return setup + _GPU_PROBE + '\n' + SOURCES[engine][3]


def _patch(path,old,new):
    text=path.read_text('utf8').replace('\r\n','\n')
    if new and new in text:return
    if not new and old not in text:return
    if text.count(old)!=1:raise RuntimeSetupError('兼容补丁版本不匹配：'+path.name)
    path.write_text(text.replace(old,new,1),'utf8')


def _amd_compat(runtime,source):
    site=runtime/'Lib'/'site-packages'
    _patch(site/'transformers'/'modeling_utils.py','import torch.distributed.tensor\n',
        'try:\n    import torch.distributed.tensor\nexcept (ImportError, ModuleNotFoundError):\n    pass  # Single-device inference does not use DTensor.\n')
    _patch(site/'audiotools'/'ml'/'decorators.py','        op: dist.ReduceOp = dist.ReduceOp.AVG,',
        '        op = getattr(getattr(dist, "ReduceOp", None), "AVG", None),')
    _patch(source/'indextts'/'infer_v2.py','            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)',
        '            import soundfile as sf\n            sf.write(output_path, wav.squeeze(0).numpy().astype("int16"), sampling_rate, subtype="PCM_16")')


def _dots_windows_compat(source):
    # Default upstream normalize_text=False needs no WFST. Lazy imports preserve
    # that default without installing Pynini, which has no official Windows wheel.
    path = source / 'src/dots_tts/utils/text.py'
    if '# VideoCaptioner: optional WFST imports are lazy.' in path.read_text('utf8'):
        return
    _patch(path, 'from tn.chinese.normalizer import Normalizer as ZhNormalizer\nfrom tn.english.normalizer import Normalizer as EnNormalizer\n', '# VideoCaptioner: optional WFST imports are lazy.\n')
    _patch(path, 'def get_chinese_text_normalizer() -> ZhNormalizer:\n    return ZhNormalizer()',
           'def get_chinese_text_normalizer():\n    from tn.chinese.normalizer import Normalizer as ZhNormalizer\n    return ZhNormalizer()')
    _patch(path, 'def get_english_text_normalizer() -> EnNormalizer:\n    return EnNormalizer()',
           'def get_english_text_normalizer():\n    from tn.english.normalizer import Normalizer as EnNormalizer\n    return EnNormalizer()')


def _gpt_windows_compat(source):
    # jieba_fast has no cp310 Windows wheels. Use the upstream-compatible Python
    # implementation explicitly, not a hidden import-time fallback.
    for relative in ('GPT_SoVITS/text/chinese.py', 'GPT_SoVITS/text/chinese2.py'):
        path = source / relative
        _patch(path, 'import jieba_fast\n', 'import jieba as jieba_fast\n')
        _patch(path, 'import jieba_fast.posseg as psg', 'import jieba.posseg as psg')
    _patch(source / 'GPT_SoVITS/text/tone_sandhi.py', 'import jieba_fast as jieba', 'import jieba  # VideoCaptioner: pure Python Windows frontend')


def _download_source(project, revision, stage, destination, cancel):
    archive = stage / (destination.name + '.zip')
    digest = _fetch('https://codeload.github.com/' + project + '/zip/' + revision, archive, cancel)
    unpacked = stage / (destination.name + '-unpacked')
    _extract(archive, unpacked)
    children = list(unpacked.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        raise RuntimeSetupError('引擎安装包目录异常')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        # GitHub archives contain an empty directory for gitlinks.
        if any(destination.iterdir()):
            raise RuntimeSetupError('子模块目标不为空，未覆盖')
        destination.rmdir()
    shutil.move(str(children[0]), destination)
    return digest


def _engine_dependencies(engine):
    # Inference-only recipes. GPU packages are installed separately and constrained
    # so a dependency cannot silently replace ROCm with an NVIDIA or CPU wheel.
    if engine == 'indextts2':
        return DEPENDENCIES
    if engine == 'qwen3':
        return ['transformers==4.57.3', 'accelerate==1.12.0', 'librosa==0.10.2.post1',
                'soundfile==0.14.0', 'sox==1.5.0', 'onnxruntime==1.23.2', 'einops==0.8.2']
    if engine == 'dots':
        return ['transformers==4.57.0', 'librosa==0.11.0', 'soundfile==0.13.1',
                'numpy==2.2.6', 'pydantic==2.12.5', 'PyYAML==6.0.3', 'safetensors==0.8.0rc0',
                'huggingface-hub==0.36.2', 'loguru==0.7.3', 'langcodes[data]==3.5.0',
                'einops==0.8.2', 'torchdiffeq==0.2.5', 'tqdm==4.67.1',
                'lingua-language-detector==2.1.1']
    if engine == 'cosyvoice3':
        return ['conformer==0.3.2', 'diffusers==0.29.0', 'hydra-core==1.3.2',
                'HyperPyYAML==1.2.3', 'inflect==7.3.1', 'librosa==0.10.2', 'lightning==2.2.4',
                'matplotlib==3.7.5', 'modelscope==1.20.0', 'networkx==3.1', 'numpy==1.26.4',
                'omegaconf==2.3.0', 'onnx==1.16.0', 'onnxruntime==1.18.0', 'protobuf==4.25',
                'pyarrow==18.1.0', 'rich==13.7.1', 'soundfile==0.12.1', 'tensorboard==2.14.0',
                'transformers==4.51.3', 'x-transformers==2.11.24', 'wetext==0.0.4', 'wget==3.2']
    if engine == 'gpt-sovits':
        return ['numpy==1.26.4', 'scipy==1.15.3', 'tensorboard==2.20.0', 'librosa==0.10.2',
                'numba==0.61.2', 'pytorch-lightning==2.4.0', 'ffmpeg-python==0.2.0',
                'onnxruntime==1.23.2', 'tqdm==4.67.1', 'cn2an==0.5.22', 'pypinyin==0.55.0',
                'pyopenjtalk-plus==0.4.1.post10', 'g2p-en==2.1.0', 'sentencepiece==0.2.1',
                'transformers==4.51.3', 'chardet==5.2.0', 'PyYAML==6.0.3', 'psutil==7.0.0',
                'jieba==0.42.1', 'split-lang==2.1.0',
                'fast-langdetect==0.3.2', 'wordsegment==1.3.1', 'rotary-embedding-torch==0.8.9',
                'ToJyutping==3.2.0', 'g2pk2==0.0.3', 'ko-pron==1.3', 'opencc==1.1.9',
                'x-transformers==2.11.24', 'torchmetrics==1.5.0', 'pydantic==2.10.6',
                'peft==0.17.1', 'soundfile==0.13.1', 'huggingface-hub==0.36.2', 'einops==0.8.2', 'nltk==3.9.1']
    if engine == 'edge-tts':
        return ['edge-tts==7.3.1', 'soundfile==0.13.1']
    raise RuntimeSetupError('未知配音引擎')


def prepare_runtime(progress, cancel, config=None):
    model_id = (config or {}).get('model_id', 'indextts2')
    engine = MODELS[model_id]['engine']
    paths = _paths(config)
    managed = _paths(dict(model_id=model_id))
    if paths != managed:
        state = inspect_runtime(config, cancel)
        if not state['ready']:
            raise RuntimeSetupError(state['message'])
        progress(100, state['message'])
        return
    if platform.system() != 'Windows':
        raise RuntimeSetupError('自动安装目前面向 Windows；其他系统请使用已配置的独立环境。')
    backend = 'online' if engine == 'edge-tts' else detect_backend()
    _check(cancel)
    if paths['python_path'].exists():
        state = inspect_runtime(config, cancel)
        if state['ready']:
            progress(100, state['message'])
            return
        raise RuntimeSetupError('已有环境未通过检查，已保留原环境，不会覆盖：\n' + state['message'])
    root = paths['repo_path'].parent
    root.mkdir(parents=True, exist_ok=True)
    needed = (1 if engine == 'edge-tts' else 15) * 1024**3
    if shutil.disk_usage(root).free < needed:
        raise RuntimeSetupError('运行环境准备空间不足。')
    if paths['repo_path'].exists():
        raise RuntimeSetupError('目标源码目录已存在，未覆盖。')
    project, revision, _, _ = SOURCES[engine]
    with tempfile.TemporaryDirectory(prefix='.setup-', dir=root) as folder:
        stage = Path(folder)
        runtime, source = stage / 'runtime', stage / 'source'
        progress(2, '正在下载独立 Python')
        _fetch(PYTHON_URL, stage / 'python.zip', cancel, PYTHON_SHA256)
        _extract(stage / 'python.zip', runtime)
        (runtime / 'Lib' / 'site-packages').mkdir(parents=True, exist_ok=True)
        (runtime / 'python310._pth').write_text('python310.zip\n.\nLib\\site-packages\nimport site\n', 'ascii')
        python = runtime / 'python.exe'
        metadata = _json_url('https://pypi.org/pypi/pip/25.0.1/json')
        wheel = next(row for row in metadata['urls'] if row['filename'].endswith('py3-none-any.whl'))
        _fetch(wheel['url'], stage / 'pip.whl', cancel, wheel['digests']['sha256'])
        _extract(stage / 'pip.whl', runtime / 'Lib' / 'site-packages')
        progress(10, '正在下载配音引擎')
        archive_hash = None
        if project:
            archive_hash = _download_source(project, revision, stage, source, cancel)
        else:
            source.mkdir()
        if engine == 'cosyvoice3':
            _download_source('shivammehta25/Matcha-TTS', 'dd9105b34bf2be2230f4aa1e4769fb586a3c824e', stage,
                             source / 'third_party' / 'Matcha-TTS', cancel)
        pip = [python, '-m', 'pip', 'install', '--disable-pip-version-check']
        # Build dependencies are explicit; some upstream packages may need a local
        # compiler. Fail with pip's diagnostic rather than silently skipping them.
        _command(pip + ['setuptools==75.8.0', 'wheel==0.45.1'], cancel)
        constraints = stage / 'constraints.txt'
        torch = []
        if backend != 'online':
            if backend == 'cuda':
                torch = ['torch==2.8.0', 'torchaudio==2.8.0']
                index = 'https://download.pytorch.org/whl/cu128'
            else:
                # Official gfx120X index includes cp310 win_amd64 wheels. A tensor
                # probe is still required; this recipe is not hardware certification.
                torch = ['torch==2.11.0+rocm7.13.0', 'torchaudio==2.11.0+rocm7.13.0', 'torchvision==0.26.0+rocm7.13.0']
                index = 'https://repo.amd.com/rocm/whl/gfx120X-all'
            progress(20, '正在安装 GPU 运行时')
            _command(pip + ['--index-url', index, '--extra-index-url', 'https://pypi.org/simple'] + torch, cancel)
        constraints.write_text('\n'.join(torch) + '\n', 'utf8')
        progress(55, '正在安装配音依赖')
        _command(pip + ['-c', constraints] + _engine_dependencies(engine), cancel)
        if engine in ('indextts2', 'cosyvoice3'):
            _command(pip + ['-c', constraints, '--no-deps', 'openai-whisper==20250625'], cancel)
        if engine in ('indextts2', 'gpt-sovits'):
            _prepare_nltk(source, cancel)
        if engine == 'indextts2' and backend != 'cuda':
            _amd_compat(runtime, source)
        if engine == 'dots':
            _dots_windows_compat(source)
        if engine == 'gpt-sovits':
            _gpt_windows_compat(source)
        progress(90, '正在检查运行时和引擎导入')
        text = _command([python, '-c', _runtime_probe(engine, source)], cancel, cwd=source, timeout=90)
        info = json.loads(next(line[7:] for line in text.splitlines() if line.startswith('VC_GPU:')))
        if backend != 'online' and ((backend == 'cuda') != (info['backend'] == 'cuda')):
            raise RuntimeSetupError('实际 GPU 后端与安装方案不一致。')
        _check(cancel)
        installed = _command([python, '-m', 'pip', 'freeze'], cancel, timeout=30)
        (source / '.videocaptioner-source.json').write_text(json.dumps(dict(
            revision=revision, archive_sha256=archive_hash, backend=backend, engine=engine,
            patches=(['lazy-optional-wfst-v1'] if engine == 'dots' else ['jieba-python-v1', 'pyopenjtalk-plus-v1'] if engine == 'gpt-sovits' else ['amd-compat-v1'] if engine == 'indextts2' and backend != 'cuda' else []),
            packages=installed.splitlines()), ensure_ascii=False), 'utf8')
        # Publish both directories only after all checks. If publication fails,
        # the installed directory remains visibly incomplete, never reported ready.
        shutil.move(str(source), paths['repo_path'])
        try:
            shutil.move(str(runtime), paths['python_path'].parent)
        except Exception as error:
            # This is only the just-created source; restore it into staging so a
            # retry is possible and no unrelated existing environment is removed.
            shutil.move(str(paths['repo_path']), source)
            raise RuntimeSetupError('运行环境发布失败，尚未安装；可重试。') from error
    progress(100, '在线语音环境已准备' if engine == 'edge-tts' else 'GPU 环境已准备；请继续下载模型')


# The upstream infer_v2 accepts these flat auxiliary locations. No network is
# permitted by the inference worker, so all auxiliary weights are prepared here.
MODEL_REPOS = [
 ('IndexTeam/IndexTTS-2','',None),
 ('facebook/w2v-bert-2.0','hf_cache/w2v-bert-2.0',{'config.json','preprocessor_config.json','model.safetensors'}),
 ('amphion/MaskGCT','hf_cache',{'semantic_codec/model.safetensors'}),
 ('funasr/campplus','hf_cache',{'campplus_cn_common.bin'}),
 ('nvidia/bigvgan_v2_22khz_80band_256x','hf_cache/bigvgan',{'config.json','bigvgan_generator.pt'})]


def _model_repos(model_id):
    if model_id == 'qwen3-design':
        return [(MODELS[model_id]['repo'], '', None), ('Qwen/Qwen3-TTS-12Hz-1.7B-Base', 'voice_clone_base', None)]
    if model_id == 'indextts2':
        return MODEL_REPOS
    if model_id == 'gpt-sovits':
        return [(MODELS[model_id]['repo'], '', {
            'gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt',
            'gsv-v2final-pretrained/s2G2333k.pth',
            'chinese-hubert-base/config.json', 'chinese-hubert-base/preprocessor_config.json',
            'chinese-hubert-base/pytorch_model.bin', 'chinese-roberta-wwm-ext-large/config.json',
            'chinese-roberta-wwm-ext-large/pytorch_model.bin', 'chinese-roberta-wwm-ext-large/tokenizer.json'})]
    return [(MODELS[model_id]['repo'], '', None)]


def _required_files(model_id):
    engine = MODELS[model_id]['engine']
    if engine == 'indextts2':
        return {'config.yaml', 'hf_cache/w2v-bert-2.0/model.safetensors',
                'hf_cache/semantic_codec_model.safetensors', 'hf_cache/campplus_cn_common.bin',
                'hf_cache/bigvgan/bigvgan_generator.pt'}
    if engine == 'qwen3':
        return {'config.json', 'model.safetensors', 'speech_tokenizer/model.safetensors', 'speech_tokenizer/config.json'} | ({'voice_clone_base/config.json', 'voice_clone_base/model.safetensors', 'voice_clone_base/speech_tokenizer/model.safetensors'} if model_id == 'qwen3-design' else set())
    if engine == 'dots':
        return {'config.json', 'model.safetensors', 'vocoder.safetensors', 'speaker_encoder.safetensors', 'latent_stats.pt'}
    if engine == 'cosyvoice3':
        return {'cosyvoice3.yaml', 'llm.pt', 'flow.pt', 'hift.pt', 'campplus.onnx',
                'speech_tokenizer_v3.onnx', 'CosyVoice-BlankEN/model.safetensors'}
    if engine == 'gpt-sovits':
        return _model_repos(model_id)[0][2]
    return set()


def _file_matches(path, row, cancel=None, full=True):
    if not path.is_file() or path.stat().st_size != row['size']:
        return False
    if not full:
        return True
    digest = hashlib.sha256() if row.get('sha256') else hashlib.sha1()
    if not row.get('sha256'):
        digest.update(('blob ' + str(row['size']) + '\0').encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            _check(cancel)
            digest.update(chunk)
    return digest.hexdigest() == (row.get('sha256') or row.get('blob_id'))


def verify_models(root, model_id, cancel=None, full=True, repo_path=None):
    if MODELS[model_id]['engine'] == 'edge-tts':
        return True
    root = Path(root).resolve()
    try:
        manifest = json.loads((root / 'videocaptioner-models.json').read_text('utf8'))
        if manifest.get('schema') != 2 or manifest.get('model_id') != model_id:
            return False
        files = manifest['files']
        if not files or not _required_files(model_id).issubset({row['relative'] for row in files}):
            return False
        for row in files:
            _check(cancel)
            path = (root / row['relative']).resolve()
            if not path.is_relative_to(root) or row['revision'] != MODEL_REVISIONS.get(row['repo']):
                return False
            if not (row.get('sha256') or row.get('blob_id')) or not _file_matches(path, row, cancel, full):
                return False
        if model_id == 'gpt-sovits':
            if repo_path is None:
                return False
            if not _verify_auxiliary(Path(repo_path), manifest.get('auxiliary', []), cancel, full):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def models_ready(config=None):
    """Fast UI availability: full recorded inventory and sizes, no multi-GB hashing.

    Explicit runtime checks and downloads do the full cryptographic verification.
    """
    model_id = (config or {}).get('model_id', 'indextts2')
    return verify_models(_paths(config)['model_dir'], model_id, full=False, repo_path=_paths(config)['repo_path'])


def download_models(progress, cancel, config=None):
    model_id = (config or {}).get('model_id', 'indextts2')
    if MODELS[model_id]['engine'] == 'edge-tts':
        _check(cancel)
        progress(100, 'Edge TTS 使用在线音色，无需下载模型')
        return
    root = _paths(config)['model_dir']
    root.mkdir(parents=True, exist_ok=True)
    inventory = []
    progress(0, '正在获取固定版本模型清单')
    for repo, subfolder, selected in _model_repos(model_id):
        _check(cancel)
        revision = MODEL_REVISIONS[repo]
        meta = _json_url('https://huggingface.co/api/models/' + repo + '/revision/' + revision + '?blobs=true')
        if meta['sha'] != revision:
            raise RuntimeSetupError('模型版本与固定清单不一致')
        for row in meta['siblings']:
            name = row['rfilename']
            if selected is not None and name not in selected:
                continue
            if selected is None and (name.startswith('.') or name.endswith(('.png', '.jpg', '.mp4', '.md'))):
                continue
            # IndexTTS's optional emotion Qwen is deliberately disabled, no extra LLM.
            if model_id == 'indextts2' and selected is None and name.startswith('qwen'):
                continue
            relative = 'semantic_codec_model.safetensors' if name == 'semantic_codec/model.safetensors' else name
            relative = (Path(subfolder) / relative).as_posix()
            destination = (root / relative).resolve()
            if not destination.is_relative_to(root.resolve()):
                raise RuntimeSetupError('模型清单包含非法路径')
            lfs = row.get('lfs') or {}
            size = row.get('size', lfs.get('size'))
            sha256, blob_id = lfs.get('sha256'), row.get('blobId')
            if not isinstance(size, int) or size < 0 or not (sha256 or blob_id):
                raise RuntimeSetupError('模型清单缺少文件身份：' + name)
            inventory.append(dict(repo=repo, revision=revision, name=name, relative=relative,
                                  sha256=sha256, blob_id=blob_id, size=size))
        if selected is not None and not selected.issubset({row['rfilename'] for row in meta['siblings']}):
            raise RuntimeSetupError('必要模型文件不存在：' + repo)
    if not _required_files(model_id).issubset({row['relative'] for row in inventory}):
        raise RuntimeSetupError('模型下载清单不完整')
    missing = [row for row in inventory if not _file_matches(root / row['relative'], row, cancel)]
    total = sum(row['size'] for row in missing)
    if total and shutil.disk_usage(root).free < total:
        raise RuntimeSetupError('模型下载空间不足。')
    done_bytes = 0
    for row in missing:
        _check(cancel)
        path = root / row['relative']
        progress(int(95 * done_bytes / max(total, 1)), '下载 ' + row['name'])
        url = 'https://huggingface.co/' + row['repo'] + '/resolve/' + row['revision'] + '/' + urllib.parse.quote(row['name'], safe='/')
        def update(done, size):
            progress(min(95, int(95 * (done_bytes + done) / max(total, 1))), '下载 ' + row['name'])
        # _fetch stages each file atomically; the inventory marker is published only
        # after EVERY file is verified. Retry reuses verified complete files.
        _fetch(url, path, cancel, row['sha256'], update)
        if not _file_matches(path, row, cancel):
            raise RuntimeSetupError('模型文件身份校验失败：' + row['name'])
        done_bytes += row['size']
    _check(cancel)
    auxiliary = _prepare_gpt_assets(_paths(config)['repo_path'], cancel, progress) if model_id == 'gpt-sovits' else []
    marker = root / 'videocaptioner-models.json'
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(schema=2, model_id=model_id, files=inventory, auxiliary=auxiliary), indent=2), 'utf8')
    os.replace(temporary, marker)
    if not verify_models(root, model_id, cancel, repo_path=_paths(config)['repo_path']):
        raise RuntimeSetupError('模型完整性复核失败')
    progress(100, '模型下载与完整性检查完成')

# Small text assets are prepared explicitly, never as an inference side effect.
_NLTK_REVISION = '550b6625bcef1f2abff2ff770a5a0d272c9c6b2a'
_NLTK_ASSETS = (
    ('corpora', 'cmudict', 'd07cca47fd72ad32ea9d8ad1219f85301eeaf4568f8b6b73747506a71fb5afd6'),
    ('taggers', 'averaged_perceptron_tagger', 'e1f13cf2532daadfd6f3bc481a49859f0b8ea6432ccdcd83e6a49a5f19008de9'),
    ('taggers', 'averaged_perceptron_tagger_eng', '6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b'),
)


def _sha256(path, cancel=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            _check(cancel)
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_nltk(source, cancel):
    target = source / '.nltk_data'
    for subdir, name, digest in _NLTK_ASSETS:
        archive = target / subdir / (name + '.zip')
        if not archive.is_file() or _sha256(archive, cancel) != digest:
            _fetch('https://raw.githubusercontent.com/nltk/nltk_data/' + _NLTK_REVISION +
                   '/packages/' + subdir + '/' + name + '.zip', archive, cancel, digest)
        _extract(archive, target / subdir)


def _auxiliary_inventory(source, cancel):
    rows = []
    for directory in ('GPT_SoVITS/text/G2PWModel', 'GPT_SoVITS/pretrained_models/fast_langdetect', '.nltk_data'):
        for path in sorted((source / directory).rglob('*')):
            if path.is_file():
                rows.append(dict(relative=path.relative_to(source).as_posix(), size=path.stat().st_size,
                                 sha256=_sha256(path, cancel)))
    return rows


def _verify_auxiliary(source, rows, cancel=None, full=True):
    source = source.resolve()
    names = {row['relative'] for row in rows}
    if not ('GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin' in names and
            any(name.lower().endswith('/g2pw.onnx') for name in names) and
            'GPT_SoVITS/text/G2PWModel/POLYPHONIC_CHARS.txt' in names and
            '.nltk_data/corpora/cmudict.zip' in names):
        return False
    for row in rows:
        _check(cancel)
        path = (source / row['relative']).resolve()
        if not path.is_relative_to(source) or not _file_matches(path, row, cancel, full):
            return False
    return True


def _prepare_gpt_assets(source, cancel, progress):
    if not (source / 'GPT_SoVITS').is_dir():
        raise RuntimeSetupError('请先安装 GPT-SoVITS 运行环境，再下载附属模型。')
    progress(96, '正在准备 GPT-SoVITS 文本前端资源')
    archive = source / '.assets' / 'G2PWModel.zip'
    expected = '46292be0374a49308069233cd5c147ae4c41806558e4781a2467a31a4d8099da'
    if not archive.is_file() or _sha256(archive, cancel) != expected:
        _fetch('https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/'
               '0c47645e02a7bc3688d7b263b0042c81e3cd82cd/G2PWModel.zip', archive, cancel, expected)
    # Extract a complete official-linked G2PW bundle including its config and dictionaries.
    with tempfile.TemporaryDirectory(prefix='.g2pw-', dir=source) as folder:
        temp = Path(folder)
        _extract(archive, temp)
        candidates = [temp / 'G2PWModel', temp / 'G2PWModel_1.1', temp]
        unpacked = next((p for p in candidates if (p / 'POLYPHONIC_CHARS.txt').is_file()), None)
        if unpacked is None:
            raise RuntimeSetupError('G2PW 模型压缩包结构不正确')
        target = source / 'GPT_SoVITS/text/G2PWModel'
        for path in unpacked.rglob('*'):
            _check(cancel)
            if path.is_file():
                destination = target / path.relative_to(unpacked)
                destination.parent.mkdir(parents=True, exist_ok=True)
                staged = destination.with_name(destination.name + '.install')
                shutil.copyfile(path, staged)
                os.replace(staged, destination)
    lid = source / 'GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin'
    # Official fastText binary has a stable object-version identity. Its downloaded
    # SHA256 is then included in the local integrity inventory.
    lid_url = ('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
               '?versionId=i0bvGpDpJiHbCgE2pdfmiT6e8rhcCJdD')
    lid_marker = lid.with_suffix('.json')
    try:
        recorded = json.loads(lid_marker.read_text('utf8'))
        reusable = (recorded['url'] == lid_url and lid.stat().st_size == 131266198 and
                    _sha256(lid, cancel) == recorded['sha256'])
    except (OSError, ValueError, KeyError):
        reusable = False
    if not reusable:
        digest = _fetch(lid_url, lid, cancel)
        if lid.stat().st_size != 131266198:
            raise RuntimeSetupError('fastText 模型长度不符合已核版本')
        lid_marker.write_text(json.dumps(dict(url=lid_url, sha256=digest)), 'utf8')
    _prepare_nltk(source, cancel)
    rows = _auxiliary_inventory(source, cancel)
    if not _verify_auxiliary(source, rows, cancel):
        raise RuntimeSetupError('GPT-SoVITS 附属资源不完整')
    return rows
