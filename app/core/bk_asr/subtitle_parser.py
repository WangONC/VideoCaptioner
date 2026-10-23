"""Lossless cue-text readers used by strict ASRData imports.

Presentation metadata is ignored; malformed spoken cues are never skipped.
"""
import re


def _time(value, ass=False):
    pattern = r'(\d+):(\d{2}):(\d{2})\.(\d{2})' if ass else r'(?:(\d{2,}):)?(\d{2}):(\d{2})[.,](\d{3})'
    match = re.fullmatch(pattern, value.strip())
    if not match:
        raise ValueError('字幕时间格式无效')
    h, m, s, fraction = match.groups()
    if int(m) >= 60 or int(s) >= 60:
        raise ValueError('字幕时间超出范围')
    return (int(h or 0) * 3600 + int(m) * 60 + int(s)) * 1000 + int(fraction) * (10 if ass else 1)


def read_cues(content, suffix):
    content = content.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    cues = []
    if suffix == '.ass':
        section, fields = '', None
        for line in content.splitlines():
            line = line.strip()
            if line.startswith('['):
                section = line.lower()
                continue
            if section != '[events]':
                continue
            if line.lower().startswith('format:'):
                fields = [part.strip().lower() for part in line.split(':', 1)[1].split(',')]
                if not {'start', 'end', 'text'}.issubset(fields) or fields[-1] != 'text' or len(set(fields)) != len(fields):
                    raise ValueError('ASS 字幕字段无效')
            elif line.lower().startswith('dialogue:'):
                if not fields:
                    raise ValueError('ASS 字幕缺少 Format')
                parts = line.split(':', 1)[1].lstrip().split(',', len(fields) - 1)
                if len(parts) != len(fields):
                    raise ValueError('ASS 对话字段不完整')
                item = dict(zip(fields, parts))
                text = item['text']
                if re.search(r'\\p[1-9]', text):
                    raise ValueError('ASS 绘图字幕不能用于配音')
                text = re.sub(r'\{[^}]*\}', '', text).replace('\\N', '\n').replace('\\n', '\n').replace('\\h', '\u00a0')
                if not text.strip():
                    raise ValueError('ASS 对话文本为空')
                cues.append((_time(item['start'], True), _time(item['end'], True), text.strip()))
            elif line and not line.startswith(';') and not line.lower().startswith('comment:'):
                raise ValueError('ASS 对话字段无效')
        # The app's bilingual ASS stores the two languages as separate events.
        merged = []
        for start, end, text in cues:
            if merged and (start, end) == merged[-1][:2]:
                previous = merged.pop()
                merged.append((start, end, previous[2] + '\n' + text))
            else:
                merged.append((start, end, text))
        cues = merged
    else:
        if suffix == '.vtt' and not re.match(r'WEBVTT(?:[ \t\n]|$)', content):
            raise ValueError('VTT 字幕缺少 WEBVTT 标头')
        for block in re.split(r'\n[ \t]*\n', content.strip()):
            lines = block.splitlines()
            if suffix == '.vtt' and re.match(r'^(WEBVTT|NOTE|STYLE|REGION)(?:[ \t]|$)', lines[0]):
                if any('-->' in line for line in lines):
                    raise ValueError('VTT 标头与字幕之间缺少空行')
                for line in lines[1:]:
                    if line.startswith('X-TIMESTAMP-MAP=') and line != 'X-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:0':
                        raise ValueError('请使用与视频时间轴一致的完整 VTT 字幕')
                continue
            position = 0 if '-->' in lines[0] else 1
            if len(lines) <= position + 1 or (suffix == '.srt' and (position != 1 or not lines[0].strip().isdigit())):
                raise ValueError('字幕条目不完整')
            match = re.fullmatch(r'(\S+)\s+-->\s+(\S+)(?:[ \t]+(.*))?', lines[position].strip())
            if not match or (suffix == '.srt' and match.group(3)):
                raise ValueError('字幕时间行无效')
            text = '\n'.join(lines[position + 1:])
            if '-->' in text or re.search(r'^WEBVTT(?:\s|$)', text, re.M):
                raise ValueError('字幕条目之间缺少空行')
            if suffix == '.vtt':
                # Inline timestamps and presentation tags are not speech text.
                text = re.sub(r'<(?:(?:\d{2,}:)?\d{2}:\d{2}\.\d{3}|/?(?:c(?:\.[^ >]+)?|v(?: [^>]+)?|lang(?: [^>]+)?))>', '', text)
            cues.append((_time(match.group(1)), _time(match.group(2)), text.strip()))
    if not cues:
        raise ValueError('字幕没有可用条目')
    previous = -1
    for start, end, text in cues:
        if not text or end <= start or start < previous:
            raise ValueError('字幕条目为空、时间无效或顺序错误')
        previous = start
    return cues
