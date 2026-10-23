"""UI capability catalog. Entries here do not register inference adapters."""
import json
from copy import deepcopy
from pathlib import Path

MODEL_OPTIONS = {
    'indextts2': dict(name='IndexTTS2', sources=('原视频声音', '参考音频'), style=True),
    'qwen3-0.6b': dict(name='Qwen3-TTS · 0.6B Base', sources=('原视频声音', '参考音频')),
    'qwen3-1.7b': dict(name='Qwen3-TTS · 1.7B Base', sources=('原视频声音', '参考音频')),
    'qwen3-custom': dict(name='Qwen3-TTS · 1.7B CustomVoice', sources=('预设音色',), voices=('Vivian', 'Serena', 'Ryan', 'Aiden', 'Dylan', 'Eric', 'Ono_Anna', 'Sohee', 'Uncle_Fu'), style=True),
    'qwen3-design': dict(name='Qwen3-TTS · 1.7B VoiceDesign', sources=('声音描述',)),
    'cosyvoice3': dict(name='CosyVoice3', sources=('原视频声音', '参考音频'), style=True),
    'gpt-sovits': dict(name='GPT-SoVITS', sources=('原视频声音', '参考音频'), seed=True),
    'dots-soar': dict(name='dots.tts · SOAR', sources=('原视频声音', '参考音频'), seed=True),
    'dots-mf': dict(name='dots.tts · MF', sources=('原视频声音', '参考音频'), seed=True),
    'edge-tts': dict(name='Edge TTS · 在线', sources=('预设音色',), voices=('zh-CN-XiaoxiaoNeural', 'zh-CN-YunxiNeural', 'en-US-JennyNeural')),
}
DEFAULT_SETTINGS = dict(model_id='indextts2', source='原视频声音', reference_source='原视频声音', reference_audio='', reference_text='', voice='', description='', language='中文', speed=1.0, style='', alignment='按字幕时间对齐', max_speedup=1.35, overflow='停止并提示', output='配音视频', original='替换原声', seed=-1)

def normalized(value):
    result = deepcopy(DEFAULT_SETTINGS)
    result.update({k: v for k, v in value.items() if k in result})
    if result['model_id'] not in MODEL_OPTIONS:
        result['model_id'] = 'indextts2'
    spec = MODEL_OPTIONS[result['model_id']]
    if result['source'] in ('原视频声音', '参考音频'):
        result['reference_source'] = result['source']
    if result['source'] not in spec['sources']:
        result['source'] = result['reference_source'] if result['reference_source'] in spec['sources'] else spec['sources'][0]
    if spec.get('voices') and result['voice'] not in spec['voices']:
        result['voice'] = spec['voices'][0]
    return result

def validation_error(value):
    if value['source'] == '参考音频' and not Path(value['reference_audio']).is_file():
        return '请选择参考音频'
    if value['source'] == '声音描述' and not value['description'].strip():
        return '请输入声音描述'
    return ''

def executable(value):
    """Explicit allow-list for the already implemented pipeline, not model promises."""
    from app.core.tts_catalog import MODELS
    return (value['model_id'] in MODELS and value['source'] in ('原视频声音', '参考音频')
            and value['language'] == '中文' and value['speed'] == 1.0
            and not (value['style'].strip() if MODEL_OPTIONS[value['model_id']].get('style') else '') and value['alignment'] == '按字幕时间对齐'
            and value['overflow'] == '停止并提示' and value['output'] == '配音视频'
            and value['original'] == '替换原声' and (not MODEL_OPTIONS[value['model_id']].get('seed') or value['seed'] == -1))

class PresetStore:
    def __init__(self, path):
        self.path = Path(path)
    def read(self):
        try:
            data = json.loads(self.path.read_text('utf-8'))
            return {k: normalized(v) for k, v in data.items() if isinstance(k, str) and isinstance(v, dict)}
        except (OSError, ValueError, AttributeError):
            return {}
    def save(self, name, value):
        if not name.strip():
            raise ValueError('请输入预设名称')
        data = self.read()
        data[name.strip()] = normalized(value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
        temporary.replace(self.path)


from PyQt5.QtCore import QObject, pyqtSignal

class SharedDubbingSettings(QObject):
    changed = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self._value = normalized({})
    def value(self):
        return deepcopy(self._value)
    def update(self, value):
        self._value = normalized(value)
        self.changed.emit(self.value())

shared_settings = SharedDubbingSettings()
