# Download totals verified against Hugging Face file metadata, 2026-10-23.
# Includes required auxiliary models, excludes Python/PyTorch runtime.
"""Implemented adapters; GPU qualification is separate from UI availability."""
MODELS = {
    'indextts2': dict(name='IndexTTS2', download_size='7.66 GB', engine='indextts2', folder='IndexTTS-2', repo='IndexTeam/IndexTTS-2'),
    'qwen3-0.6b': dict(name='Qwen3-TTS · 0.6B', download_size='2.52 GB', engine='qwen3', folder='Qwen3-TTS-12Hz-0.6B-Base', repo='Qwen/Qwen3-TTS-12Hz-0.6B-Base'),
    'qwen3-1.7b': dict(name='Qwen3-TTS · 1.7B', download_size='4.54 GB', engine='qwen3', folder='Qwen3-TTS-12Hz-1.7B-Base', repo='Qwen/Qwen3-TTS-12Hz-1.7B-Base'),
}
