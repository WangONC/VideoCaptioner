"""Synthetic tones test mechanics, NOT cloned speech quality."""
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import threading
import unittest
import wave
from app.core.dubbing import CancelledError, DubbingConfig, DubbingError, DubbingPipeline, parse_srt, select_reference

SRT = '1\n00:00:00,500 --> 00:00:04,000\nHello, welcome.\n你好，欢迎观看。\n\n2\n00:00:04,500 --> 00:00:05,500\nThank you.\n谢谢。\n'

def tone(path, seconds):
    with wave.open(str(path), 'wb') as wav:
        wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        wav.writeframes(b''.join(struct.pack('<h', round(4000*math.sin(2*math.pi*440*n/24000))) for n in range(round(seconds*24000))))

class FakeProvider:
    def __init__(self, seconds=.7, version=1, fail_once=False):
        self.seconds, self.version, self.fail_once = seconds, version, fail_once
        self.calls = []; self.closed = False
    def fingerprint(self):
        return {'fake': self.version}
    def synthesize(self, text, reference_audio, output_path, cancel_event=None):
        self.calls.append(text)
        if self.fail_once:
            self.fail_once = False
            Path(output_path).write_bytes(b'partial')
            raise RuntimeError('Recoverable failure')
        tone(output_path, self.seconds)
    def close(self):
        self.closed = True

class SubtitleTests(unittest.TestCase):
    def test_selection_preserves_source(self):
        cues = parse_srt(SRT)
        self.assertEqual(cues[0].speech_text, '你好，欢迎观看。')
        self.assertIn('Hello', cues[0].original_text)
        self.assertTrue(select_reference(cues).english_text)
    def test_chinese_only_reference(self):
        cues = parse_srt('1\n00:00:00,000 --> 00:00:04,000\n你好\n')
        self.assertEqual(select_reference(cues), cues[0])
    def test_adjacent_short_reference(self):
        cues = parse_srt('1\n00:00:00,000 --> 00:00:02,000\n你好\n\n2\n00:00:02,100 --> 00:00:04,000\n世界\n')
        self.assertEqual(select_reference(cues).end, 4)
    def test_multilingual_lines_are_not_voiced_twice(self):
        prefix = '1\n00:00:00,000 --> 00:00:04,000\n'
        self.assertEqual(parse_srt(prefix + 'これは日本語です\n这是中文翻译')[0].speech_text, '这是中文翻译')
        self.assertEqual(parse_srt(prefix + 'GPUをテストします\nWe test the GPU.', '英语')[0].speech_text, 'We test the GPU.')
        self.assertEqual(parse_srt(prefix + 'GPU를 테스트합니다\n测试 GPU。')[0].speech_text, '测试 GPU。')
        self.assertEqual(parse_srt(prefix + 'GPUをテストします\n测试 GPU。', '日语')[0].speech_text, 'GPUをテストします')

    def test_invalid(self):
        for srt in ('', SRT.replace('00:00:04,500', '00:00:03,500'), SRT.replace('00:00:04,000', '00:00:00,000'), SRT.replace('00:00:00,500', '00:61:00,500'), '1\n00:00:00,000 --> 00:00:04,000\nEnglish only'):
            with self.subTest(srt=srt), self.assertRaises(DubbingError): parse_srt(srt)

@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg required')
class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.video = self.root / 'source.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=160x90:r=25:d=6','-f','lavfi','-i','sine=frequency=220:duration=6','-c:v','mpeg4','-c:a','aac','-t','6',str(self.video)], check=True)
        self.srt = self.root/'source.srt'; self.srt.write_text(SRT, encoding='utf-8')
        self.config = DubbingConfig(str(self.video),str(self.srt),str(self.root/'dubbed.mp4'),str(self.root/'cache'))
    def tearDown(self): self.temp.cleanup()
    def video_hash(self, path):
        return subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-c','copy','-f','hash','-'])
    def test_mux_cache_and_invalidation(self):
        original = self.srt.read_bytes(); provider = FakeProvider(); progress=[]
        output = DubbingPipeline(self.config,provider).run(lambda *args: progress.append(args))
        self.assertEqual(len(provider.calls),2); self.assertTrue(provider.closed)
        self.assertEqual(self.video_hash(self.video),self.video_hash(output))
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',output]))
        self.assertAlmostEqual(float(info['format']['duration']),6,delta=.05)
        self.assertEqual([s['codec_type'] for s in info['streams']],['video','audio'])
        pcm = subprocess.check_output(['ffmpeg','-v','error','-i',output,'-map','0:a:0','-f','s16le','-ac','1','-ar','24000','-'])
        def rms_at(second):
            start = round(second * 24000) * 2
            samples = struct.unpack('<2400h', pcm[start:start+4800])
            return math.sqrt(sum(x*x for x in samples) / len(samples))
        self.assertLess(rms_at(.1), 100)
        self.assertGreater(rms_at(.7), 1000)
        self.assertLess(rms_at(4.1), 100)
        self.assertGreater(rms_at(4.7), 1000)
        self.assertLess(rms_at(5.7), 100)
        self.assertEqual(self.srt.read_bytes(),original)
        self.config.output_path=str(self.root/'second.mp4')
        second=FakeProvider(); DubbingPipeline(self.config,second).run(); self.assertEqual(second.calls,[])
        self.config.output_path=str(self.root/'changed.mp4')
        changed=FakeProvider(version=2); DubbingPipeline(self.config,changed).run(); self.assertEqual(len(changed.calls),2)
        self.assertEqual(progress[-1],(2,2,'配音完成'))
    def test_speed_limit_preserves_existing_output(self):
        Path(self.config.output_path).write_bytes(b'existing')
        with self.assertRaisesRegex(DubbingError,'already exists'):
            DubbingPipeline(self.config,FakeProvider(seconds=8)).run()
        self.assertEqual(Path(self.config.output_path).read_bytes(),b'existing')
    def test_retry_corrupt_cache(self):
        provider=FakeProvider(fail_once=True); DubbingPipeline(self.config,provider).run(); self.assertEqual(len(provider.calls),3)
        for path in Path(self.config.cache_dir).glob('*.wav'): path.write_bytes(b'corrupt')
        self.config.output_path=str(self.root/'recovered.mp4')
        provider=FakeProvider(); DubbingPipeline(self.config,provider).run(); self.assertEqual(len(provider.calls),2)
    def test_cancel_resume(self):
        cancel=threading.Event(); provider=FakeProvider()
        def progress(done,total,message):
            if done==1: cancel.set()
        with self.assertRaises(CancelledError): DubbingPipeline(self.config,provider).run(progress,cancel)
        self.assertEqual(len(provider.calls),1); self.assertFalse(Path(self.config.output_path).exists())
        provider=FakeProvider(); DubbingPipeline(self.config,provider).run(); self.assertEqual(len(provider.calls),1)
    def test_speed_limit_without_output(self):
        with self.assertRaisesRegex(DubbingError,'shorten the text'):
            DubbingPipeline(self.config,FakeProvider(seconds=8)).run()
        self.assertFalse(Path(self.config.output_path).exists())
    def test_output_created_during_job_is_preserved(self):
        def progress(done,total,message):
            if done==total:Path(self.config.output_path).write_bytes(b'other writer')
        with self.assertRaisesRegex(DubbingError,'not overwritten'):
            DubbingPipeline(self.config,FakeProvider()).run(progress)
        self.assertEqual(Path(self.config.output_path).read_bytes(),b'other writer')
    def test_precancel_and_overwrite(self):
        cancel=threading.Event(); cancel.set()
        with self.assertRaises(CancelledError): DubbingPipeline(self.config,FakeProvider()).run(cancel_event=cancel)
        self.config.output_path=str(self.video)
        with self.assertRaisesRegex(DubbingError,'overwrite'): DubbingPipeline(self.config,FakeProvider()).run()

if __name__=='__main__': unittest.main()
