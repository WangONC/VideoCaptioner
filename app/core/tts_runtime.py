"""Managed IndexTTS2 setup. All downloads stay local; no GPU brand setting in UI.

Windows recipes are implementation candidates, NOT hardware qualification. A
runtime is usable only after its own GPU tensor probe succeeds. No driver edits.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
from app.core.tts_catalog import MODELS
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
    'more-itertools==11.1.0','munch==4.0.0','numba==0.63.0','numpy==2.2.6',
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
            stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
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
                process.terminate()
                try:process.wait(5)
                except subprocess.TimeoutExpired:process.kill();process.wait()


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
    paths=_paths(config)
    engine=MODELS[(config or {}).get('model_id','indextts2')]['engine']
    models=(paths['model_dir']/('config.yaml' if engine=='indextts2' else 'config.json')).is_file()
    result=dict(ready=False,models_ready=models,device=None,message='尚未准备运行环境')
    if not paths['python_path'].is_file() or not (paths['repo_path']/('indextts/infer_v2.py' if engine=='indextts2' else 'qwen_tts/__init__.py')).is_file():return result
    try:
        text=_command([paths['python_path'],'-c',_GPU_PROBE],cancel=cancel,timeout=45)
        info=json.loads(next(line[7:] for line in text.splitlines() if line.startswith('VC_GPU:')))
        return result|info|dict(ready=True,message='GPU 运行环境可用')
    except Exception as error:return result|dict(message=str(error))


def _patch(path,old,new):
    text=path.read_text('utf8').replace('\r\n','\n')
    if new in text:return
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


def prepare_runtime(progress,cancel,config=None):
    model_id=(config or {}).get('model_id','indextts2')
    engine=MODELS[model_id]['engine']
    paths=_paths(config);managed=_paths(dict(model_id=model_id))
    # Advanced settings are for reuse, never permission to overwrite an existing environment.
    if paths!=managed:
        state=inspect_runtime(config)
        if not state['ready']:raise RuntimeSetupError(state['message'])
        progress(100,state['message']);return
    backend=detect_backend();_check(cancel)
    if paths['python_path'].exists():
        state=inspect_runtime(config)
        if state['ready']:progress(100,state['message']);return
        raise RuntimeSetupError('已有环境未通过 GPU 检查，已保留原环境。请检查显卡驱动；不会覆盖已有环境。\n'+state['message'])
    root=paths['repo_path'].parent;root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free<15*1024**3:raise RuntimeSetupError('准备运行环境至少需要 15 GB 可用空间。')
    if paths['repo_path'].exists():raise RuntimeSetupError('目标源码目录已存在，未覆盖。为保护已有文件，本次安装已停止。')
    with tempfile.TemporaryDirectory(prefix='.setup-',dir=root) as folder:
        stage=Path(folder);runtime=stage/'runtime';source=stage/'source'
        progress(2,'正在下载独立 Python')
        _fetch(PYTHON_URL,stage/'python.zip',cancel,PYTHON_SHA256);_extract(stage/'python.zip',runtime)
        (runtime/'Lib'/'site-packages').mkdir(parents=True,exist_ok=True)
        (runtime/'python310._pth').write_text('python310.zip\n.\nLib\\site-packages\nimport site\n','ascii')
        python=runtime/'python.exe'
        metadata=_json_url('https://pypi.org/pypi/pip/25.0.1/json')
        wheel=next(row for row in metadata['urls'] if row['filename'].endswith('py3-none-any.whl'))
        _fetch(wheel['url'],stage/'pip.whl',cancel,wheel['digests']['sha256']);_extract(stage/'pip.whl',runtime/'Lib'/'site-packages')
        progress(10,'正在下载配音引擎')
        revision=SOURCE_REVISION if engine=='indextts2' else QWEN_SOURCE_REVISION
        project='index-tts/index-tts' if engine=='indextts2' else 'QwenLM/Qwen3-TTS'
        _fetch('https://codeload.github.com/'+project+'/zip/'+revision,stage/'source.zip',cancel)
        _extract(stage/'source.zip',stage/'unpacked')
        children=list((stage/'unpacked').iterdir())
        if len(children)!=1 or not children[0].is_dir():raise RuntimeSetupError('引擎安装包目录异常')
        shutil.move(str(children[0]),source)
        pip=[python,'-m','pip','install','--disable-pip-version-check']
        constraints=stage/'constraints.txt'
        if backend=='cuda':
            torch=['torch==2.8.0','torchaudio==2.8.0'];index='https://download.pytorch.org/whl/cu128'
        else:
            torch=['torch==2.11.0+rocm7.13.0','torchaudio==2.11.0+rocm7.13.0','torchvision==0.26.0+rocm7.13.0']
            index='https://repo.amd.com/rocm/whl/gfx120X-all'
        constraints.write_text('\n'.join(torch)+'\n','utf8')
        progress(20,'正在安装 GPU 运行时')
        _command(pip+['--index-url',index,'--extra-index-url','https://pypi.org/simple']+torch,cancel)
        progress(55,'正在安装配音依赖')
        if engine=='indextts2':
            _command(pip+['-c',constraints]+DEPENDENCIES,cancel)
            _command(pip+['--no-deps','openai-whisper==20250625'],cancel)
            if backend!='cuda':_amd_compat(runtime,source)
        else:
            _command(pip+['-c',constraints,'transformers==4.57.3','accelerate==1.12.0',
                'librosa==0.10.2.post1','soundfile==0.14.0','sox==1.5.0','onnxruntime==1.23.2','einops==0.8.2'],cancel)
        progress(90,'正在检查 GPU 和引擎导入')
        engine_import="from indextts.infer_v2 import IndexTTS2" if engine=="indextts2" else "from qwen_tts import Qwen3TTSModel"
        text=_command([python,'-c',_GPU_PROBE+"\nimport sys;sys.path.insert(0,"+repr(str(source))+");"+engine_import],cancel,timeout=90)
        info=json.loads(next(line[7:] for line in text.splitlines() if line.startswith('VC_GPU:')))
        if (backend=='cuda')!=(info['backend']=='cuda'):raise RuntimeSetupError('实际 GPU 后端与安装方案不一致。')
        _check(cancel)
        (source/'.videocaptioner-source.json').write_text(json.dumps(dict(revision=revision,backend=backend,engine=engine)),'utf8')
        shutil.move(str(source),paths['repo_path'])
        try:shutil.move(str(runtime),paths['python_path'].parent)
        except Exception:
            # Source is our newly published directory; keep it for diagnosis, never hide partial setup as ready.
            raise RuntimeSetupError('运行环境发布失败。已保留源码，尚未标记成功。')
    progress(100,'GPU 运行环境已准备；模型尚需单独下载')


# The upstream infer_v2 accepts these flat auxiliary locations. No network is
# permitted by the inference worker, so all auxiliary weights are prepared here.
MODEL_REPOS = [
 ('IndexTeam/IndexTTS-2','',None),
 ('facebook/w2v-bert-2.0','hf_cache/w2v-bert-2.0',{'config.json','preprocessor_config.json','model.safetensors'}),
 ('amphion/MaskGCT','hf_cache',{'semantic_codec/model.safetensors'}),
 ('funasr/campplus','hf_cache',{'campplus_cn_common.bin'}),
 ('nvidia/bigvgan_v2_22khz_80band_256x','hf_cache/bigvgan',{'config.json','bigvgan_generator.pt'})]


def download_models(progress,cancel,config=None):
    root=_paths(config)['model_dir'];root.mkdir(parents=True,exist_ok=True)
    inventory=[]
    progress(0,'正在获取模型清单')
    model_id=(config or {}).get('model_id','indextts2')
    repos=MODEL_REPOS if model_id=='indextts2' else [(MODELS[model_id]['repo'],'',None)]
    for repo,subfolder,selected in repos:
        _check(cancel);meta=_json_url('https://huggingface.co/api/models/'+repo+'?blobs=true')
        revision=meta['sha']
        if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):raise RuntimeSetupError('模型版本标识无效')
        for row in meta['siblings']:
            name=row['rfilename']
            if selected is not None and name not in selected:continue
            if selected is None and (name.startswith('.') or name.startswith('qwen') or name.endswith(('.png','.jpg','.mp4'))):continue
            relative='semantic_codec_model.safetensors' if name=='semantic_codec/model.safetensors' else name
            destination=(root/subfolder/relative).resolve()
            if not destination.is_relative_to(root.resolve()):raise RuntimeSetupError('模型清单包含非法路径')
            inventory.append(dict(repo=repo,revision=revision,name=name,path=str(destination),
                sha256=row.get('lfs',{}).get('sha256'),size=row.get('size',0)))
        if selected is not None and not selected.issubset({row['rfilename'] for row in meta['siblings']}):
            raise RuntimeSetupError('必要的附属模型文件不存在：'+repo)
    total=sum(row['size'] for row in inventory)
    if total and shutil.disk_usage(root).free<total:raise RuntimeSetupError('模型下载空间不足。')
    for index,row in enumerate(inventory):
        _check(cancel);path=Path(row['path'])
        if path.exists() and row['sha256']:
            with path.open('rb') as existing:
                checker=hashlib.sha256()
                for chunk in iter(lambda:existing.read(1024*1024),b''):checker.update(chunk)
                digest=checker.hexdigest()
            if digest==row['sha256']:continue
        progress(int(index*100/len(inventory)),'下载 '+row['name'])
        url='https://huggingface.co/'+row['repo']+'/resolve/'+row['revision']+'/'+urllib.parse.quote(row['name'],safe='/')
        def update(done,size):
            if size:progress(int((index+done/size)*100/len(inventory)),'下载 '+row['name'])
        _fetch(url,path,cancel,row['sha256'],update)
    _check(cancel)
    file=root/'videocaptioner-models.json';temporary=file.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(schema=1,files=inventory),indent=2),'utf8');os.replace(temporary,file)
    progress(100,'模型下载完成')
