"""Download identities and adapter families, not claims of GPU qualification."""
MODELS = {
    'indextts2': dict(name='IndexTTS2', download_size='7.66 GB', engine='indextts2', folder='IndexTTS-2', repo='IndexTeam/IndexTTS-2'),
    'qwen3-0.6b': dict(name='Qwen3-TTS · 0.6B Base', download_size='2.52 GB', engine='qwen3', folder='Qwen3-TTS-12Hz-0.6B-Base', repo='Qwen/Qwen3-TTS-12Hz-0.6B-Base'),
    'qwen3-1.7b': dict(name='Qwen3-TTS · 1.7B Base', download_size='4.54 GB', engine='qwen3', folder='Qwen3-TTS-12Hz-1.7B-Base', repo='Qwen/Qwen3-TTS-12Hz-1.7B-Base'),
    'qwen3-custom': dict(name='Qwen3-TTS · 1.7B CustomVoice', download_size='按模型清单', engine='qwen3', folder='Qwen3-TTS-12Hz-1.7B-CustomVoice', repo='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'),
    'qwen3-design': dict(name='Qwen3-TTS · 1.7B VoiceDesign', download_size='按模型清单', engine='qwen3', folder='Qwen3-TTS-12Hz-1.7B-VoiceDesign', repo='Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'),
    'dots-soar': dict(name='dots.tts · SOAR', download_size='按模型清单', engine='dots', folder='dots.tts-soar', repo='dots-studio/dots.tts-soar'),
    'dots-mf': dict(name='dots.tts · MF', download_size='按模型清单', engine='dots', folder='dots.tts-mf', repo='dots-studio/dots.tts-mf'),
    'cosyvoice3': dict(name='CosyVoice3', download_size='按模型清单', engine='cosyvoice3', folder='Fun-CosyVoice3-0.5B-2512', repo='FunAudioLLM/Fun-CosyVoice3-0.5B-2512'),
    'gpt-sovits': dict(name='GPT-SoVITS · v2', download_size='按模型清单', engine='gpt-sovits', folder='GPT-SoVITS-v2', repo='lj1995/GPT-SoVITS'),
    'edge-tts': dict(name='Edge TTS · 在线', download_size='无需模型', engine='edge-tts', folder='edge-tts', repo=None),
}

# Reviewed official source identities. Installation never follows a mutable branch.
SOURCES = {
    'indextts2': ('index-tts/index-tts', '4f8792ff120cd3ea470dd511e997a17c86cddd10', 'indextts/infer_v2.py', 'from indextts.infer_v2 import IndexTTS2'),
    'qwen3': ('QwenLM/Qwen3-TTS', '022e286b98fbec7e1e916cb940cdf532cd9f488e', 'qwen_tts/__init__.py', 'from qwen_tts import Qwen3TTSModel'),
    'dots': ('studio-dots-ai/dots.tts', 'b65ca7149fb43b8fb8d52916aa12738deda7b188', 'src/dots_tts/runtime.py', 'from dots_tts.runtime import DotsTtsRuntime'),
    'cosyvoice3': ('QwenAudio/CosyVoice', '074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc', 'cosyvoice/cli/cosyvoice.py', 'from cosyvoice.cli.cosyvoice import AutoModel'),
    'gpt-sovits': ('RVC-Boss/GPT-SoVITS', 'b4570b876f5c086819ba84e3215472733c6a7b70', 'GPT_SoVITS/TTS_infer_pack/TTS.py', 'from TTS_infer_pack.TTS import TTS, TTS_Config'),
    'edge-tts': (None, '7.3.1', None, 'import edge_tts'),
}
MODEL_REVISIONS = {
    'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice': '0c0e3051f131929182e2c023b9537f8b1c68adfe',
    'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign': '5ecdb67327fd37bb2e042aab12ff7391903235d3',
    'dots-studio/dots.tts-soar': '2f9b3e18d70d670d4c701da2dc55ded5755815ce',
    'dots-studio/dots.tts-mf': 'c28105adc8228143392b4e346994ff613ee48a06',
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512': '29e01c4e8d000f4bcd70751be16fa94bf3d85a18',
    'lj1995/GPT-SoVITS': '8f8857563c040d542de108a2c18b0425abd6c9c5',
}
MODEL_REVISIONS.update({
    'IndexTeam/IndexTTS-2': '740dcaff396282ffb241903d150ac011cd4b1ede',
    'Qwen/Qwen3-TTS-12Hz-0.6B-Base': '5d83992436eae1d760afd27aff78a71d676296fc',
    'Qwen/Qwen3-TTS-12Hz-1.7B-Base': 'fd4b254389122332181a7c3db7f27e918eec64e3',
    'facebook/w2v-bert-2.0': 'da985ba0987f70aaeb84a80f2851cfac8c697a7b',
    'amphion/MaskGCT': '265c6cef07625665d0c28d2faafb1415562379dc',
    'funasr/campplus': 'e4b6ede7ce16997aff4ae69fbca1f0175e2afede',
    'nvidia/bigvgan_v2_22khz_80band_256x': '633ff708ed5b74903e86ff1298cf4a98e921c513',
})
