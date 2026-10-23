"""Shared immutable job snapshots for single and batch dubbing."""
from dataclasses import asdict, dataclass
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import threading
from .pipeline import DubbingConfig, DubbingPipeline, DubbingError, parse_srt, _run, _check

# Selected backend is shared with the model dialog; unsupported backends fail closed.
backend_choices = {}
device_choices = {}
_execution_lock = threading.Lock()


def runtime_paths(model_id, backend=None, device=None):
    backend = backend or backend_choices.get(model_id, 'pytorch')
    if backend == 'audio.cpp':
        from app.core.audio_cpp_runtime import default_paths
        return default_paths(model_id)
    from app.core.tts_runtime import default_paths
    from app.config import APPDATA_PATH
    device = device or device_choices.get(model_id, 'gpu')
    paths = default_paths(model_id, device=device)
    name = 'dubbing-runtime.json' if model_id == 'indextts2' else 'dubbing-runtime-' + model_id + '.json'
    try:
        saved = json.loads((APPDATA_PATH / name).read_text('utf-8'))
        # Migrate only the tool's exact legacy default paths. Custom external
        # environments are retained; neither old files nor config are deleted.
        from app.core.tts_catalog import MODELS
        legacy = APPDATA_PATH / 'tts' / MODELS[model_id]['engine']
        old_defaults = dict(python_path=legacy / 'runtime' / 'python.exe', repo_path=legacy / 'source')
        migrate_legacy = all(not saved.get(key) or Path(saved[key]).absolute() == value.absolute()
                             for key, value in old_defaults.items())
        for key in paths:
            if not saved.get(key):
                continue
            if migrate_legacy and key in old_defaults:
                continue
            paths[key] = str(saved[key])
    except (OSError, ValueError, AttributeError):
        pass
    return paths


def next_output(video, mode):
    video = Path(video)
    suffix = '.wav' if mode == 'audio' else '.mp4'
    base = video.with_name(video.stem + '-中文配音' + suffix)
    result, number = base, 2
    while result.exists():
        result = base.with_name(base.stem + f'-{number}' + suffix)
        number += 1
    return str(result)


@dataclass(frozen=True)
class DubbingRequest:
    # Tuples, not mutable dictionaries: later UI edits cannot change queued jobs.
    config_items: tuple
    subtitle_text: str
    subtitle_mode: str = 'none'

    def config(self):
        return DubbingConfig(**dict(self.config_items))


def build_request(video, subtitle, output, settings, *, runtime=None, association=None,
                  output_mode=None, subtitle_mode='none', cache_dir=None, device=None):
    from .settings import normalized, executable, validation_error
    from app.config import CACHE_PATH
    value = normalized(settings)
    device = device or device_choices.get(value['model_id'], 'gpu')
    if device not in ('gpu', 'cpu'):
        raise DubbingError('无效的运行设备')
    association = dict(association or {})
    subtitle = association.get('subtitle') or subtitle
    if association.get('reference') and value['source'] in ('原视频声音', '参考音频'):
        value.update(source='参考音频', reference_audio=association['reference'],
                     reference_text=association.get('reference_text', ''))
    elif association.get('reference_text'):
        value['reference_text'] = association['reference_text']
    if not executable(value):
        raise DubbingError('所选模型或配音选项暂未接入，请使用已支持的配置')
    backend = backend_choices.get(value['model_id'], 'pytorch')
    if backend == 'audio.cpp':
        from app.core.audio_cpp_runtime import AUDIO_CPP_MODELS
        if value['model_id'] not in AUDIO_CPP_MODELS:
            raise DubbingError('此模型不支持 audio.cpp 后端')
    error = validation_error(value)
    if error:
        raise DubbingError(error)
    if not subtitle or not str(subtitle).strip():
        raise DubbingError('请选择对应字幕')
    video, subtitle = Path(video).expanduser().resolve(), Path(subtitle).expanduser().resolve()
    if not video.is_file() or not subtitle.is_file():
        raise DubbingError('视频或字幕文件不存在')
    text = subtitle.read_text('utf-8-sig')
    parse_srt(text, value['language'])
    mode = output_mode or ('audio' if value['output'] == '配音音频' else 'video')
    output = Path(output or next_output(video, mode)).expanduser().resolve()
    reference = value['reference_audio'] if value['source'] == '参考音频' else None
    protected = {video, subtitle}
    if reference:
        protected.add(Path(reference).expanduser().resolve())
    if output in protected or output.exists():
        raise DubbingError('输出文件已存在或与输入重名，请选择新的文件名，避免覆盖。')
    if output.suffix.lower() not in (('.wav',) if mode == 'audio' else ('.mp4', '.mkv', '.mov')):
        raise DubbingError('配音音频请使用 WAV，配音视频请使用 MP4、MKV 或 MOV')
    if subtitle_mode not in ('none', 'soft', 'hard') or (mode == 'audio' and subtitle_mode != 'none'):
        raise DubbingError('无效的字幕合成方式')
    config = DubbingConfig(video_path=str(video), srt_path=str(subtitle), output_path=str(output),
        cache_dir=str(cache_dir or CACHE_PATH / 'dubbing'), model_id=value['model_id'],
        reference_audio=reference, reference_text=value['reference_text'] or None,
        backend=backend, device=device, language=value['language'], voice=value['voice'], description=value['description'],
        seed=int(value['seed']) if value['model_id'] in ('dots-soar', 'dots-mf', 'gpt-sovits') else -1,
        alignment='natural' if value['alignment'] == '自然语速' else 'subtitle',
        overflow='keep' if value['overflow'] == '保留超限音频' else 'stop',
        max_speedup=float(value['max_speedup']), speed=float(value['speed']), output_mode=mode,
        original='separate' if value['original'] == '保留为独立音轨' else 'replace',
        **(runtime if runtime is not None else runtime_paths(value['model_id'], device=device)))
    return DubbingRequest(tuple(asdict(config).items()), text, subtitle_mode)


@contextmanager
def execution_lane(cancel_event=None):
    # Single/batch inference and runtime changes share one cancellable GPU lane.
    while not _execution_lock.acquire(timeout=.1):
        _check(cancel_event)
    try:
        _check(cancel_event)
        yield
    finally:
        _execution_lock.release()


def execute_request(request, on_progress=None, cancel_event=None, provider=None):
    with execution_lane(cancel_event):
        return _execute_request(request, on_progress, cancel_event, provider)


def _execute_request(request, on_progress=None, cancel_event=None, provider=None):
    """Use private subtitle bytes and atomic final publication for subtitle mux/burn."""
    config = request.config()
    progress = on_progress or (lambda done, total, message: None)
    output = Path(config.output_path)
    _check(cancel_event)
    if output.exists():
        raise DubbingError('输出文件已存在，未覆盖')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dubbing-job-', dir=output.parent) as directory:
        work = Path(directory)
        subtitles = work / 'captions.srt'
        subtitles.write_text(request.subtitle_text, encoding='utf-8')
        config.srt_path = str(subtitles)
        if request.subtitle_mode == 'none':
            return DubbingPipeline(config, provider).run(progress, cancel_event)
        config.output_path = str(work / ('dubbed' + output.suffix))
        def dubbing_progress(done, total, message):
            progress(int(80 * done / max(total, 1)), 100, message)
        DubbingPipeline(config, provider).run(dubbing_progress, cancel_event)
        progress(85, 100, '正在合成字幕')
        temporary = work / ('result' + output.suffix)
        args = [config.ffmpeg, '-v', 'error', '-y', '-i', config.output_path]
        if request.subtitle_mode == 'soft':
            args += ['-i', str(subtitles), '-map', '0:v:0', '-map', '0:a', '-map', '1:0',
                     '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'srt' if output.suffix == '.mkv' else 'mov_text']
        else:
            # Fixed basename in a private cwd avoids filter-language path escaping.
            args += ['-map', '0:v:0', '-map', '0:a', '-vf', 'subtitles=captions.srt',
                     '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-c:a', 'copy']
        _run(args + [str(temporary)], cancel_event, cwd=work)
        _check(cancel_event)
        try:
            if os.name == 'nt':
                os.rename(temporary, output)
            else:
                os.link(temporary, output)
        except FileExistsError as error:
            raise DubbingError('输出文件已被其他操作创建，未覆盖') from error
        progress(100, 100, '配音与合成完成')
        return str(output)
