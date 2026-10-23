"""Offline dubbing timeline, exported as WAV or muxed with the source video.

Original audio may be retained as a separate track; it is never source-separated.

Automatic reference selection is a text/duration heuristic, NOT speaker diarization.
It assumes a single speaker. Review the extracted reference before voice cloning.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
import wave


class DubbingError(RuntimeError):
    pass


class CancelledError(DubbingError):
    pass


@dataclass(frozen=True)
class Cue:
    index: int
    start: float
    end: float
    original_text: str
    speech_text: str
    english_text: str


def parse_srt(text, language="中文"):
    """Choose Chinese-bearing lines explicitly; never voice English-only subtitles.

    Bilingual input must put the translation on a separate line. Mixed-language
    Chinese lines preserve Latin names/numbers. Original blocks remain available.
    Overlapping, unordered, empty and malformed cues are rejected, not repaired.
    """
    cues = []
    stamp = r'(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})'
    def seconds(parts):
        h, m, s, ms = map(int, parts)
        if m >= 60 or s >= 60:
            raise DubbingError('Invalid subtitle timestamp')
        return h * 3600 + m * 60 + s + ms / 1000
    for block in re.split(r'\n\s*\n', text.lstrip('\ufeff').replace('\r\n', '\n').strip()):
        lines = block.splitlines()
        if len(lines) < 3 or not lines[0].strip().isdigit():
            raise DubbingError('Expected numbered SRT cues')
        match = re.fullmatch(stamp + r'\s*-->\s*' + stamp, lines[1].strip())
        if not match:
            raise DubbingError('Invalid subtitle timestamp line')
        start, end = seconds(match.groups()[:4]), seconds(match.groups()[4:])
        original = '\n'.join(lines[2:])
        clean = [re.sub(r'<[^>]+>|\{\\[^}]+\}', '', line).strip() for line in lines[2:]]
        japanese_or_korean = r'[\u3040-\u30ff\uac00-\ud7af]'
        chinese = [line for line in clean if re.search(r'[\u3400-\u9fff]', line)
                   and not re.search(japanese_or_korean, line)]
        english = [line for line in clean if re.search(r'[A-Za-z]', line)
                   and not re.search(r'[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]', line)]
        if language == '中文':
            selected = chinese
        elif language == '英语':
            selected = english
        elif language == '日语':
            selected = [line for line in clean if re.search(r'[\u3040-\u30ff]', line)]
            if not selected and not english:
                selected = chinese  # Kanji-only Japanese cues.
        elif language == '韩语':
            selected = [line for line in clean if re.search(r'[\uac00-\ud7af]', line)]
        else:
            raise DubbingError('Unsupported subtitle language')
        if not selected:
            raise DubbingError(f'Cue {lines[0]} has no {language} translation')
        if end <= start or (cues and start < cues[-1].end - 0.0001):
            raise DubbingError('Subtitle cues must have positive duration and must not overlap')
        cues.append(Cue(int(lines[0]), start, end, original, ' '.join(selected), ' '.join(english)))
    if not cues:
        raise DubbingError('No subtitle cues')
    return cues


def select_reference(cues):
    """Single-speaker timing/text heuristic, not acoustic analysis or diarization.

    Chinese-only translations use their original-video time spans. Neighbouring
    short cues can form a reference when gaps are at most 0.5 seconds.
    """
    candidates = []
    def clear(cue):
        return not re.search(r'(^|\s)[-–] |[♪♫]|\[[^]]+\]', cue.original_text)
    for i, first in enumerate(cues):
        if not clear(first):
            continue
        group = []
        for current in cues[i:]:
            if not clear(current) or (group and current.start - group[-1].end > .5):
                break
            if current.end - first.start > 10:
                break
            group.append(current)
            if current.end - first.start >= 3:
                candidates.append(Cue(first.index, first.start, current.end,
                    '\n'.join(c.original_text for c in group),
                    ' '.join(c.speech_text for c in group),
                    ' '.join(c.english_text for c in group if c.english_text)))
    if not candidates:
        raise DubbingError('No suitable 3–10 second reference interval. Supply a clean reference WAV.')
    return min(candidates, key=lambda c: (not bool(c.english_text),
               abs((c.end - c.start) - 6) - min(len(c.speech_text), 60) / 100))


@dataclass
class DubbingConfig:
    video_path: str
    srt_path: str
    output_path: str
    cache_dir: str
    model_id: str = 'indextts2'
    reference_audio: str = None
    reference_text: str = None
    python_path: str = None
    repo_path: str = None
    model_dir: str = None
    max_speedup: float = 1.35
    backend: str = 'pytorch'
    language: str = '中文'
    voice: str = ''
    description: str = ''
    seed: int = -1
    alignment: str = 'subtitle'
    overflow: str = 'stop'
    retries: int = 1
    ffmpeg: str = 'ffmpeg'
    ffprobe: str = 'ffprobe'
    output_mode: str = 'video'
    speed: float = 1.0
    original: str = 'replace'


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError('Dubbing cancelled; completed cue audio remains cached')


def _run(args, cancel=None, cwd=None):
    _check(cancel)
    with tempfile.TemporaryFile(mode='w+b') as out, tempfile.TemporaryFile(mode='w+b') as err:
        proc = subprocess.Popen([str(a) for a in args], stdout=out, stderr=err, cwd=cwd)
        try:
            while proc.poll() is None:
                _check(cancel)
                time.sleep(.05)
            out.seek(0); err.seek(0)
            stdout, stderr = out.read(), err.read()
            if proc.returncode:
                raise DubbingError(stderr.decode('utf-8', 'replace')[-4000:])
            return stdout
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()


def _hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _wav_duration(path):
    with wave.open(str(path), 'rb') as wav:
        if wav.getnframes() <= 0:
            raise DubbingError('Generated audio is empty')
        return wav.getnframes() / wav.getframerate()


class DubbingPipeline:
    def __init__(self, config, provider=None):
        self.config = config
        self.provider = provider

    def run(self, on_progress=None, cancel_event=None):
        try:
            return self._run_pipeline(on_progress, cancel_event)
        finally:
            if self.provider is not None:
                self.provider.close()

    def _run_pipeline(self, on_progress=None, cancel_event=None):
        c = self.config
        progress = on_progress or (lambda completed, total, message: None)
        if c.max_speedup < 1 or not math.isfinite(c.max_speedup) or c.retries < 0:
            raise DubbingError('Invalid speedup/retry limits')
        if not math.isfinite(c.speed) or not .5 <= c.speed <= 2:
            raise DubbingError('Speech speed must be between 0.5 and 2')
        if c.output_mode not in ('video', 'audio') or c.original not in ('replace', 'separate'):
            raise DubbingError('Unsupported dubbing output mode')
        video, source, output = map(lambda p: Path(p).expanduser().resolve(), (c.video_path, c.srt_path, c.output_path))
        if output in (video, source):
            raise DubbingError('Output must not overwrite the source video or subtitles')
        if output.exists():
            raise DubbingError('Output already exists; choose a new filename to preserve it')
        if not video.is_file() or not source.is_file():
            raise DubbingError('Source video or subtitles do not exist')
        if c.output_mode == 'audio':
            if output.suffix.lower() != '.wav':
                raise DubbingError('Audio output must be WAV')
        elif output.suffix.lower() not in ('.mp4', '.mkv', '.mov'):
            raise DubbingError('Output must be MP4, MKV or MOV')
        cues = parse_srt(source.read_text(encoding='utf-8-sig'), c.language)
        cache = Path(c.cache_dir).expanduser().resolve(); cache.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        _check(cancel_event)
        info = json.loads(_run([c.ffprobe, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', video], cancel_event))
        streams = [s for s in info['streams'] if s['codec_type'] == 'video']
        if not streams:
            raise DubbingError('Input has no video stream')
        duration = float(streams[0].get('duration') or info['format']['duration'])
        if not math.isfinite(duration) or duration <= 0 or cues[-1].end > duration + .02:
            raise DubbingError('Subtitles extend beyond the video duration')
        input_hash = _hash(video)
        if c.alignment not in ('subtitle', 'natural') or c.overflow not in ('stop', 'keep'):
            raise DubbingError('Unsupported timing settings')
        progress(0, len(cues), 'Preparing voice')
        with tempfile.TemporaryDirectory(prefix='dubbing-', dir=cache) as workdir:
            work = Path(workdir)
            reference = None
            reference_text, reference_hash = '', None
            needs_reference = c.model_id not in ('qwen3-custom', 'qwen3-design', 'edge-tts')
            if needs_reference:
                reference = work / 'reference.wav'
                if c.reference_audio:
                    reference_source = Path(c.reference_audio).expanduser().resolve()
                    if output == reference_source:
                        raise DubbingError('Output must not overwrite reference audio')
                    _run([c.ffmpeg, '-v', 'error', '-y', '-i', reference_source, '-vn', '-ac', '1', '-ar', '24000', '-c:a', 'pcm_s16le', reference], cancel_event)
                    reference_text = c.reference_text or ''
                else:
                    cue = select_reference(cues)
                    _run([c.ffmpeg, '-v', 'error', '-y', '-ss', cue.start, '-i', video, '-t', cue.end-cue.start,
                          '-vn', '-ac', '1', '-ar', '24000', '-c:a', 'pcm_s16le', reference], cancel_event)
                    # A bilingual subtitle does not prove which language the original
                    # speaker used. Never label translated text as an exact transcript.
                    reference_text = c.reference_text or ''
                if not 3 <= _wav_duration(reference) <= 10.02:
                    raise DubbingError('Reference must be between 3 and 10 seconds')
                reference_hash = _hash(reference)
            if self.provider is None:
                if c.backend == 'audio.cpp':
                    from .audio_cpp_provider import create_provider
                    self.provider = create_provider(c, cancel_event=cancel_event)
                else:
                    from .provider import create_provider
                    self.provider = create_provider(c.model_id, c.python_path, c.repo_path, c.model_dir,
                        language=c.language, voice=c.voice, description=c.description, seed=c.seed, ffmpeg=c.ffmpeg, cancel_event=cancel_event)
            if hasattr(self.provider,'reference_text'):self.provider.reference_text=reference_text
            backend = self.provider.fingerprint()
            timeline = work / 'timeline.wav'
            # Stream PCM segments and silence to disk: memory use does not grow with video length.
            rate = 24000
            position = 0
            with wave.open(str(timeline), 'wb') as dest:
                dest.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
                def silence(frames):
                    while frames > 0:
                        count = min(frames, rate)
                        dest.writeframesraw(b'\0\0' * count); frames -= count
                for i, cue in enumerate(cues):
                    _check(cancel_event)
                    identity = {'schema': 1, 'input': input_hash, 'reference': reference_hash,
                                'reference_text': reference_text, 'text': cue.speech_text, 'backend': backend}
                    key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    raw = cache / (key + '.wav')
                    try:
                        _wav_duration(raw)
                    except (OSError, EOFError, wave.Error, DubbingError):
                        raw.unlink(missing_ok=True)
                        for attempt in range(c.retries + 1):
                            partial = work / (key + '.partial.wav')
                            partial.unlink(missing_ok=True)
                            try:
                                self.provider.synthesize(cue.speech_text, str(reference) if reference else None, str(partial), cancel_event)
                                _check(cancel_event)
                                _wav_duration(partial)
                                os.replace(partial, raw)
                                break
                            except CancelledError:
                                raise
                            except Exception:
                                if attempt == c.retries:
                                    raise
                                progress(i, len(cues), f'Retrying cue {cue.index}')
                    slot = cue.end - cue.start
                    length = _wav_duration(raw)
                    speed = max(1.0, length / c.speed / slot) if c.alignment == 'subtitle' else 1.0
                    over_limit = speed > c.max_speedup + .0001
                    if over_limit and c.overflow == 'stop':
                        raise DubbingError(f'Cue {cue.index} needs {speed:.2f}x additional speed (limit {c.max_speedup:.2f}x); shorten the text or adjust timing')
                    # Natural/keep modes never discard speech: spillover shifts later cues.
                    preserve = c.alignment == 'natural' or over_limit
                    if over_limit:
                        speed = 1.0
                    fitted = work / 'fitted.wav'
                    filters = []
                    factor = c.speed * speed
                    while factor > 2:
                        filters.append('atempo=2'); factor /= 2
                    filters.append(f'atempo={factor:.9f}')
                    if not preserve:
                        filters.extend([f'apad=whole_dur={slot:.9f}', f'atrim=duration={slot:.9f}'])
                    _run([c.ffmpeg, '-v', 'error', '-y', '-i', raw, '-af', ','.join(filters), '-ac', '1', '-ar', rate,
                          '-c:a', 'pcm_s16le', fitted], cancel_event)
                    start = max(round(cue.start * rate), position)
                    frames = round(_wav_duration(fitted) * rate) if preserve else round(slot * rate)
                    silence(start - position)
                    with wave.open(str(fitted), 'rb') as wav:
                        remaining = frames
                        while remaining:
                            data = wav.readframes(min(remaining, rate))
                            if not data:
                                silence(remaining); break
                            dest.writeframesraw(data); remaining -= len(data) // 2
                    position = start + frames
                    progress(i+1, len(cues), f'Completed cue {cue.index}')
                silence(max(0, round(duration * rate) - position))
            temporary_output = output.with_name(output.stem + '.partial-' + work.name + output.suffix)
            try:
                if c.output_mode == 'audio':
                    # Copy to the output filesystem before exclusive publication.
                    with timeline.open('rb') as src, temporary_output.open('xb') as dst:
                        while True:
                            _check(cancel_event)
                            chunk = src.read(1024 * 1024)
                            if not chunk: break
                            dst.write(chunk)
                else:
                    args = [c.ffmpeg, '-v', 'error', '-y', '-i', video, '-i', timeline,
                            '-map', '0:v:0', '-map', '1:a:0']
                    if c.original == 'separate':
                        args += ['-map', '0:a:0?']
                    args += ['-map_metadata', '0', '-c:v', 'copy', '-c:a', 'aac',
                             '-metadata:s:a:0', 'title=Dubbing', '-disposition:a:0', 'default']
                    if c.original == 'separate':
                        args += ['-c:a:1', 'copy', '-metadata:s:a:1', 'title=Original', '-disposition:a:1', '0']
                    args += ['-t', max(duration, position / rate), temporary_output]
                    _run(args, cancel_event)
                _check(cancel_event)
                # Exclusive publication: never overwrite a file created while a long job ran.
                try:
                    if os.name=='nt':os.rename(temporary_output,output)
                    else:os.link(temporary_output, output)
                except FileExistsError as error:
                    raise DubbingError('Output was created by another operation; it was not overwritten') from error
            finally:
                temporary_output.unlink(missing_ok=True)
            progress(len(cues), len(cues), 'Dubbing complete')
            return str(output)
