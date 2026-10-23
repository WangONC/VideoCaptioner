# Local dubbing core

This module imports without Qt, torch or IndexTTS installed. `DubbingPipeline.run`
is blocking: call it from a background thread, pass a `threading.Event` for
cancellation, and marshal progress callbacks back to the GUI thread.

```python
from app.core.dubbing import DubbingConfig, DubbingPipeline
config = DubbingConfig(
    video_path='lesson.mp4', srt_path='lesson.zh.srt',
    output_path='lesson.dubbed.mp4', cache_dir='dubbing-cache',
    python_path='/path/to/indextts/.venv/bin/python',
    repo_path='/path/to/indextts', model_dir='/path/to/checkpoints',
)
output_path = DubbingPipeline(config).run(
    on_progress=lambda completed, total, message: print(message),
    cancel_event=None,
)
```

## Boundaries and behaviour

- Only Chinese-bearing subtitle lines are voiced. Separate bilingual lines are
  supported; the original subtitle file is never modified. Latin names embedded
  in Chinese lines remain intact. English-only cues fail explicitly.
- Timestamps must be ordered, positive-duration and non-overlapping. Invalid
  cues are rejected rather than silently shifted.
- Reference selection uses subtitle time spans and text heuristics, prefers
  English-bearing spans near six seconds and allows adjacent short cues with
  gaps up to 0.5 seconds. Chinese-only subtitles work too. **This does not detect
  speaker identity, music, noise or actual speech.** It assumes one speaker;
  review the source interval or supply a clean 3–10 second `reference_audio` WAV.
  `reference_text` is metadata for cache identity, not required by IndexTTS2.
- A dedicated Python subprocess loads IndexTTS2 once, then accepts sequential
  requests. The application interpreter never imports torch. The subprocess
  must have compatible dependencies and all required model/auxiliary assets.
  It refuses network connections and missing assets fail locally. No model
  downloads, package installation, external service or server is started.
- NVIDIA CUDA and AMD ROCm both use PyTorch's CUDA API. This is runtime detection,
  **not a promise that all AMD hardware or Windows AMD runtimes are supported**.
  Missing GPU support fails explicitly; there is no CPU fallback. CUDA-specific
  fused kernels and DeepSpeed are disabled. Actual GPU inference remains untested.
- Raw per-cue PCM WAVs are cached atomically using source-video content,
  reference content, selected Chinese text and model/source fingerprints.
  Retry defaults to one additional attempt; partial WAVs are not reused.
  Cancellation terminates the worker/process; successful cues remain reusable.
- Audio longer than its subtitle slot is tempo-adjusted only up to
  `max_speedup` (default 1.35). Larger mismatches stop with an actionable error.
  Short audio is silence-padded, not stretched to fill the slot.
- The final audio timeline is streamed to disk at mono 24 kHz. The video stream
  is copied without re-encoding; duration is bounded to the source video.
  **Original audio, music and ambience are replaced**, not source-separated or
  mixed. The first video stream is retained; extra audio/subtitle tracks are not.
  Output replacement is atomic and occurs only after successful muxing.

## Verification

`python -m unittest discover -s tests -p 'test_dubbing*.py' -v`

Tests use synthetic tones and fake TTS implementations. They exercise real
FFmpeg extraction/fitting/muxing, video-stream hash preservation, timeline
silence/sound placement, duration, cache reuse/invalidation, corrupted-cache
recovery, retry, cancellation/resume, GPU refusal and persistent JSON protocol.
**Passing tests do not establish voice-cloning quality or real model compatibility.**

The worker's API calls were checked against the official IndexTTS2 source:
https://github.com/index-tts/index-tts/blob/main/indextts/infer_v2.py
and https://github.com/index-tts/index-tts#python-api . No upstream source was
copied into this module. The host repository uses GPL-3.0; model licenses and
runtime redistribution require their own review before distribution.
