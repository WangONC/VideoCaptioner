"""Pinned audio.cpp v0.9.2 Windows Vulkan setup, never a GPU qualification.

Contract source: https://github.com/0xShug0/audio.cpp/tree/0097d365a00b52a6991ad7618f0a04e8d2935e47
Model package filenames are from that release's model_specs, hashes from the
pinned official audio-cpp/audio.cpp-gguf snapshot. No model-manager subprocess.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]
RELEASE = 'v0.9.2'
SOURCE_REVISION = '0097d365a00b52a6991ad7618f0a04e8d2935e47'
ASSET_URL = 'https://github.com/0xShug0/audio.cpp/releases/download/v0.9.2/audio-v0.9.2-bin-windows-x64-vulkan-portable.zip'
ASSET_SHA256 = '4d11a82c560f7a1dc0934e6fef847d6e0b779cf79854aa7d90d8b28e4ad04d93'
MODEL_REPO = 'audio-cpp/audio.cpp-gguf'
MODEL_REVISION = 'ef1ab2598f6d4d101486e4c5b9e605c687f20642'
AUDIO_CPP_MODELS = {'indextts2': {'family': 'index_tts2', 'filename': 'index-tts2-q8_0.gguf', 'path': 'IndexTTS2-GGUF/index-tts2-q8_0.gguf', 'size': 3633895968, 'sha256': 'a56e09f49fd39a894dfe1b065ab94884e9ed3515a92f75aff06fb6e6714b6b2c'}, 'qwen3-0.6b': {'family': 'qwen3_tts', 'filename': 'qwen3-tts-12hz-0.6b-base-q8_0.gguf', 'path': 'Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0.gguf', 'size': 1991217728, 'sha256': 'c7d0e6c9485a25ae1f6a4c0796f0023230a0051c44cbd2bd05d73770f2cce637'}, 'qwen3-1.7b': {'family': 'qwen3_tts', 'filename': 'qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf', 'path': 'Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf', 'size': 2695186240, 'sha256': '92641a8022d521efd79a5db45e952589b3f2b781680f9abfcf26f075d37b86c6'}, 'qwen3-custom': {'family': 'qwen3_tts', 'filename': 'qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf', 'path': 'Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF/qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf', 'size': 2817055200, 'sha256': 'c8ac3f94b6e73f137c0e7ecc9270bc8faec7a476ff799b87e48844d1d0987ac3'}, 'qwen3-design': {'family': 'qwen3_tts', 'filename': 'qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf', 'path': 'Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF/qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf', 'size': 2817000064, 'sha256': 'b86e03d1d2077bbaed157ab7a7fc646c3e309e69afa52c5e1c86b622b4715057'}, 'dots-soar': {'family': 'dots_tts', 'filename': 'dots-tts-soar-q8_0.gguf', 'path': 'DotTTS-SOAR-GGUF/dots-tts-soar-q8_0.gguf', 'size': 2962751232, 'sha256': '2ac144eacc02ff47b37988a924a0480ed2fe3756790c51c1943348eba34dae27'}, 'dots-mf': {'family': 'dots_tts', 'filename': 'dots-tts-mf-q8_0.gguf', 'path': 'DotTTS-MF-GGUF/dots-tts-mf-q8_0.gguf', 'size': 2965377472, 'sha256': 'abc4c27db2a6750119d70f5e66e9c268c80c47ff4659867bbba124717f05ae7a'}}


def _helpers():
    from app.core.tts_runtime import RuntimeSetupError, _fetch, _extract, _command, _check
    return RuntimeSetupError, _fetch, _extract, _command, _check


def default_paths(model_id='indextts2'):
    Error, *_ = _helpers()
    if model_id not in AUDIO_CPP_MODELS:
        raise Error('此模型没有 audio.cpp 适配')
    base = ROOT / 'AppData' / 'tts' / 'audio.cpp' / RELEASE
    return dict(python_path=str(base / 'audiocpp_cli.exe'), repo_path=str(base),
                model_dir=str(ROOT / 'AppData' / 'models' / 'audio.cpp' / model_id))


def _paths(config=None):
    config = config or {}
    values = default_paths(config.get('model_id', 'indextts2'))
    values.update({key: str(config[key]) for key in values if config.get(key)})
    return {key: Path(value).expanduser().resolve() for key, value in values.items()}


_DIGEST_CACHE = {}

def _digest(path, cancel=None, force=False):
    _, _, _, _, check = _helpers()
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    check(cancel)
    if not force and key in _DIGEST_CACHE:
        return _DIGEST_CACHE[key]
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        while True:
            check(cancel)
            block = source.read(1024 * 1024)
            if not block:
                break
            result.update(block)
    digest = result.hexdigest()
    _DIGEST_CACHE[key] = digest
    return digest


def required_models(model_id):
    specs = [AUDIO_CPP_MODELS[model_id]]
    if model_id == 'qwen3-design':
        # Design once, then clone the resulting voice with Base across all cues.
        specs.append(AUDIO_CPP_MODELS['qwen3-1.7b'])
    return specs


def models_ready(config=None, cancel=None, verify_hash=False):
    """UI checks verified receipts and identity; explicit checks hash every file."""
    try:
        config = config or {}
        root = _paths(config)['model_dir']
        for spec in required_models(config.get('model_id', 'indextts2')):
            file = root / spec['filename']
            if not file.is_file() or file.stat().st_size != spec['size']:
                return False
            stat = file.stat()
            if verify_hash:
                if _digest(file, cancel, force=True) != spec['sha256']:
                    return False
                receipt = dict(revision=MODEL_REVISION, sha256=spec['sha256'], size=stat.st_size,
                               mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)
                temp = file.with_suffix('.verified.tmp')
                temp.write_text(json.dumps(receipt), 'utf-8')
                os.replace(temp, file.with_suffix('.verified.json'))
            else:
                receipt = json.loads(file.with_suffix('.verified.json').read_text('utf-8'))
                if not (receipt.get('revision') == MODEL_REVISION and receipt.get('sha256') == spec['sha256']
                        and receipt.get('size') == stat.st_size and receipt.get('mtime_ns') == stat.st_mtime_ns
                        and receipt.get('ctime_ns') == stat.st_ctime_ns):
                    return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def inspect_runtime(config=None, cancel=None):
    Error, _, _, command, check = _helpers()
    paths = _paths(config)
    result = dict(ready=False, installed=paths['python_path'].is_file(), models_ready=False,
                  message='尚未安装 audio.cpp 环境', device=None)
    if not result['installed']:
        if platform.system() != 'Windows':
            result['message'] = 'audio.cpp 自动环境仅提供 Windows x64 二进制；当前系统尚无可用环境（CPU 也不例外）'
        return result
    try:
        manifest = json.loads((paths['repo_path'] / 'videocaptioner-runtime.json').read_text('utf-8'))
        if manifest.get('release') != RELEASE or manifest.get('archive_sha256') != ASSET_SHA256:
            raise Error('audio.cpp 安装身份不匹配，请重新安装')
        files = manifest.get('files', {})
        if paths['python_path'].relative_to(paths['repo_path']).as_posix() not in files:
            raise Error('audio.cpp 安装清单缺少执行文件')
        for relative, expected in files.items():
            path = (paths['repo_path'] / relative).resolve()
            if not path.is_relative_to(paths['repo_path']) or not path.is_file() or _digest(path, cancel) != expected:
                raise Error('audio.cpp 安装文件不完整或已变更：' + relative)
        check(cancel)
        if (config or {}).get('device', 'gpu') == 'cpu':
            command([paths['python_path'], '--help'], cancel=cancel, cwd=paths['repo_path'], timeout=45)
            result.update(ready=True, device='cpu', message='audio.cpp CLI 检查通过，已明确选择 CPU（模型推理待实测）')
        else:
            devices = command([paths['python_path'], '--list-devices'], cancel=cancel, cwd=paths['repo_path'], timeout=45)
            vulkan = [line for line in devices.splitlines() if re.match(r'^vulkan:\d+\s', line, re.I)]
            if not vulkan:
                raise Error('未检测到 Vulkan 设备，请检查显卡驱动；不会自动回退 CPU')
            result.update(ready=True, device=vulkan[0], message='audio.cpp Vulkan 设备可用（模型推理待实测）')
        result['models_ready'] = models_ready(config, cancel)
    except Exception as error:
        # Cancellation must stop callers, not become a successful check result.
        check(cancel)
        result['message'] = str(error)
    return result


def prepare_runtime(progress, cancel, config=None):
    Error, fetch, extract, _, check = _helpers()
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise Error('audio.cpp 自动安装目前仅支持 Windows x64（CPU / Vulkan），不提供当前系统的构建')
    paths = _paths(config)
    if paths != _paths(dict(model_id=(config or {}).get('model_id', 'indextts2'))):
        raise Error('自动安装仅写入受管 audio.cpp 目录')
    root = paths['repo_path']
    if root.exists():
        state = inspect_runtime(config, cancel)
        if state['ready']:
            progress(100, state['message']); return
        raise Error('已有 audio.cpp 环境未通过检查，已保留原文件。' + state['message'])
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < 300 * 1024**2:
        raise Error('audio.cpp 安装空间不足')
    with tempfile.TemporaryDirectory(prefix='.audiocpp-setup-', dir=root.parent) as directory:
        stage = Path(directory)
        progress(0, '下载 audio.cpp Vulkan 环境')
        fetch(ASSET_URL, stage / 'runtime.zip', cancel, ASSET_SHA256,
              lambda done, total: progress(int(80 * done / total) if total else 0, '下载 audio.cpp Vulkan 环境'))
        extract(stage / 'runtime.zip', stage / 'unpacked')
        candidates = list((stage / 'unpacked').rglob('audiocpp_cli.exe'))
        if len(candidates) != 1:
            raise Error('audio.cpp 安装包缺少唯一 CLI')
        source = candidates[0].parent
        # Keep all sidecar DLL/resources from the official archive; no global install.
        files = {p.relative_to(source).as_posix(): _digest(p, cancel)
                 for p in source.rglob('*') if p.is_file()}
        if not any(name.lower().endswith('.dll') for name in files):
            raise Error('audio.cpp 安装包缺少运行库')
        (source / 'videocaptioner-runtime.json').write_text(json.dumps(dict(
            release=RELEASE, source_revision=SOURCE_REVISION, archive_sha256=ASSET_SHA256,
            files=files), ensure_ascii=False, indent=2), 'utf-8')
        check(cancel)
        os.rename(source, root)
    # Install can succeed while the driver is absent; report that truthfully.
    state = inspect_runtime(config, cancel)
    if not state['ready']:
        raise Error('audio.cpp 已下载，但环境检查未通过：' + state['message'])
    progress(100, state['message'])


def download_models(progress, cancel, config=None):
    Error, fetch, _, _, check = _helpers()
    config = config or {}
    model_id = config.get('model_id', 'indextts2')
    if model_id not in AUDIO_CPP_MODELS:
        raise Error('此模型没有 audio.cpp GGUF 下载适配')
    specs = required_models(model_id)
    root = _paths(config)['model_dir']
    root.mkdir(parents=True, exist_ok=True)
    if models_ready(config, cancel, verify_hash=True):
        progress(100, 'GGUF 模型校验通过'); return
    total = sum(spec['size'] for spec in specs)
    completed = 0
    for spec in specs:
        check(cancel)
        path = root / spec['filename']
        if path.is_file() and path.stat().st_size == spec['size'] and _digest(path, cancel, force=True) == spec['sha256']:
            completed += spec['size']
            continue
        if shutil.disk_usage(root).free < spec['size']:
            raise Error('GGUF 模型下载空间不足')
        url = 'https://huggingface.co/' + MODEL_REPO + '/resolve/' + MODEL_REVISION + '/' + urllib.parse.quote(spec['path'], safe='/')
        progress(int(100 * completed / total), '下载 ' + spec['filename'])
        fetch(url, path, cancel, spec['sha256'],
              lambda done, size: progress(min(99, int(100 * (completed + done) / total)), '下载 ' + spec['filename']))
        completed += spec['size']
    check(cancel)
    if not models_ready(config, cancel, verify_hash=True):
        raise Error('GGUF 模型完整性校验失败')
    progress(100, 'GGUF 模型下载完成并通过校验')
