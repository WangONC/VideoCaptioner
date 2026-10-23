"""Shared subtitle import for standalone and batch dubbing."""
from pathlib import Path
from app.core.entities import SupportedSubtitleFormats


def subtitle_file_filter():
    return '字幕 (' + ' '.join('*.' + item.value for item in SupportedSubtitleFormats) + ')'


def normalized_subtitles(path):
    from app.core.bk_asr.asr_data import ASRData
    from .pipeline import DubbingError
    path = Path(path)
    if path.suffix.lower().lstrip('.') not in {item.value for item in SupportedSubtitleFormats}:
        raise DubbingError('不支持的字幕格式')
    try:
        return ASRData.from_subtitle_file(str(path), strict=True).to_srt()
    except (ValueError, UnicodeError) as exc:
        raise DubbingError(str(exc)) from exc
