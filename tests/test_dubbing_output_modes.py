"""Real FFmpeg output contracts with synthetic TTS; no model quality claims."""
import json
import math
import shutil
import struct
import subprocess
import tempfile
import threading
import unittest
import wave
from pathlib import Path

from app.core.dubbing import DubbingConfig, DubbingError, DubbingPipeline, CancelledError
from test_dubbing_core import FakeProvider, SRT


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg required')
class OutputModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / 'source.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                        'color=c=blue:s=160x90:r=25:d=6', '-f', 'lavfi', '-i',
                        'sine=frequency=220:duration=6', '-c:v', 'mpeg4', '-c:a',
                        'aac', '-t', '6', str(self.video)], check=True)
        self.subtitle = self.root / 'source.srt'
        self.subtitle.write_text(SRT, encoding='utf-8')
        self.config = DubbingConfig(str(self.video), str(self.subtitle),
            str(self.root/'speech.wav'), str(self.root/'cache'), output_mode='audio')

    def tearDown(self):
        self.temp.cleanup()

    def stream_hash(self, path, mapping):
        return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
            '-map', mapping, '-c', 'copy', '-f', 'hash', '-'])

    def rms_at(self, path, second):
        with wave.open(str(path), 'rb') as wav:
            wav.setpos(round(second*wav.getframerate()))
            samples = struct.unpack('<2400h', wav.readframes(2400))
        return math.sqrt(sum(x*x for x in samples)/len(samples))

    def test_wav_timeline_speed_and_cache(self):
        provider = FakeProvider()
        output = DubbingPipeline(self.config, provider).run()
        with wave.open(output) as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 24000))
            self.assertEqual(wav.getnframes(), 6*24000)
        self.assertLess(self.rms_at(output, .1), 100)
        self.assertGreater(self.rms_at(output, 1), 1000)
        self.assertLess(self.rms_at(output, 4.1), 100)
        self.config.output_path = str(self.root/'faster.wav')
        self.config.speed = 2
        second = FakeProvider()
        faster = DubbingPipeline(self.config, second).run()
        self.assertEqual(second.calls, [])
        self.assertLess(self.rms_at(faster, 1), 100)
        self.assertTrue(second.closed)
        self.config.output_path = str(self.root/'slower.wav')
        self.config.speed = .5
        self.config.max_speedup = 1.5
        third = FakeProvider()
        slower = DubbingPipeline(self.config, third).run()
        self.assertEqual(third.calls, [])
        self.assertGreater(self.rms_at(slower, 1.4), 1000)

    def test_reference_free_and_english_selection(self):
        self.config.model_id = 'edge-tts'
        self.config.language = '英语'
        self.subtitle.write_text('1\n00:00:00,000 --> 00:00:01,000\nHello.\n你好。\n', encoding='utf-8')
        class NoReference(FakeProvider):
            def synthesize(inner, text, reference_audio, output_path, cancel_event=None):
                self.assertIsNone(reference_audio)
                return super(NoReference, inner).synthesize(text, reference_audio, output_path, cancel_event)
        provider = NoReference()
        DubbingPipeline(self.config, provider).run()
        self.assertEqual(provider.calls, ['Hello.'])

    def test_overflow_keep_preserves_whole_speech(self):
        self.config.model_id = 'edge-tts'
        self.config.overflow = 'keep'
        output = DubbingPipeline(self.config, FakeProvider(seconds=4)).run()
        with wave.open(output) as wav:
            self.assertGreater(wav.getnframes() / wav.getframerate(), 8)

    def test_natural_does_not_compress_to_slot(self):
        self.config.model_id = 'edge-tts'
        self.config.alignment = 'natural'
        output = DubbingPipeline(self.config, FakeProvider(seconds=4)).run()
        with wave.open(output) as wav:
            self.assertGreater(wav.getnframes() / wav.getframerate(), 8)

    def test_separate_track_preserves_original_bitstream(self):
        self.config.output_mode = 'video'
        self.config.output_path = str(self.root/'separate.mp4')
        self.config.original = 'separate'
        output = DubbingPipeline(self.config, FakeProvider()).run()
        streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
            '-show_streams', '-of', 'json', output]))['streams']
        self.assertEqual([s['codec_type'] for s in streams], ['video', 'audio', 'audio'])
        self.assertEqual(streams[1]['disposition']['default'], 1)
        self.assertEqual(streams[2]['disposition']['default'], 0)
        self.assertEqual(self.stream_hash(self.video, '0:v:0'), self.stream_hash(output, '0:v:0'))
        self.assertEqual(self.stream_hash(self.video, '0:a:0'), self.stream_hash(output, '0:a:1'))

    def test_slowdown_overflow_does_not_truncate(self):
        self.config.speed = .5
        with self.assertRaisesRegex(DubbingError, 'additional speed'):
            DubbingPipeline(self.config, FakeProvider()).run()
        self.assertFalse(Path(self.config.output_path).exists())
        self.assertFalse(list(self.root.glob('*.partial-*')))

    def test_cancellation_retains_cache_not_output(self):
        cancel = threading.Event()
        def progress(done, total, message):
            if done == 1: cancel.set()
        with self.assertRaises(CancelledError):
            DubbingPipeline(self.config, FakeProvider()).run(progress, cancel)
        self.assertFalse(Path(self.config.output_path).exists())
        self.assertEqual(len(list((self.root/'cache').glob('*.wav'))), 1)
        self.assertFalse(list((self.root/'cache').glob('dubbing-*')))
        resumed = FakeProvider()
        DubbingPipeline(self.config, resumed).run()
        self.assertEqual(len(resumed.calls), 1)

    def test_output_race_does_not_overwrite(self):
        def progress(done, total, message):
            if done == total: Path(self.config.output_path).write_bytes(b'other output')
        with self.assertRaisesRegex(DubbingError, 'not overwritten'):
            DubbingPipeline(self.config, FakeProvider()).run(progress)
        self.assertEqual(Path(self.config.output_path).read_bytes(), b'other output')
        self.assertFalse(list(self.root.glob('*.partial-*')))

    def test_invalid_settings_and_existing_output_close_provider(self):
        for field, value in [('speed', .4), ('speed', float('nan')),
                ('output_mode', 'other'), ('original', 'mix')]:
            with self.subTest(field=field, value=value):
                previous = getattr(self.config, field)
                setattr(self.config, field, value)
                provider = FakeProvider()
                with self.assertRaises(DubbingError): DubbingPipeline(self.config, provider).run()
                self.assertTrue(provider.closed)
                setattr(self.config, field, previous)
        Path(self.config.output_path).write_bytes(b'existing')
        with self.assertRaisesRegex(DubbingError, 'already exists'):
            DubbingPipeline(self.config, FakeProvider()).run()
        self.assertEqual(Path(self.config.output_path).read_bytes(), b'existing')

    def test_fingerprint_failure_closes_provider(self):
        class BrokenFingerprint(FakeProvider):
            def fingerprint(self): raise RuntimeError('fingerprint failure')
        provider = BrokenFingerprint()
        with self.assertRaisesRegex(RuntimeError, 'fingerprint failure'):
            DubbingPipeline(self.config, provider).run()
        self.assertTrue(provider.closed)
        self.assertFalse(list((self.root/'cache').glob('dubbing-*')))

    def test_audio_extension_and_source_validation(self):
        self.config.output_path = str(self.root/'bad.mp4')
        with self.assertRaisesRegex(DubbingError, 'WAV'):
            DubbingPipeline(self.config, FakeProvider()).run()
        self.config.output_path = str(self.subtitle)
        with self.assertRaisesRegex(DubbingError, 'overwrite'):
            DubbingPipeline(self.config, FakeProvider()).run()
        self.config.output_path = str(self.root/'out.wav')
        self.config.video_path = str(self.root/'missing.mp4')
        with self.assertRaisesRegex(DubbingError, 'do not exist'):
            DubbingPipeline(self.config, FakeProvider()).run()
