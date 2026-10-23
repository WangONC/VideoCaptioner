"""Authored fixtures: supported subtitle imports and source-safe execution."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from app.core.bk_asr.asr_data import ASRData
from app.core.dubbing.subtitles import normalized_subtitles, subtitle_file_filter
from app.core.dubbing.pipeline import parse_srt, DubbingError, DubbingConfig, DubbingPipeline
from app.core.dubbing.execution import build_request, execute_request
from app.core.dubbing.settings import normalized
from test_dubbing_core import FakeProvider

SRT = '1\n00:00:00,200 --> 00:00:03,500\nHello.\n你好。\n第三行。\n'
VTT = 'WEBVTT\n\nNOTE authored fixture\nignored\n\nSTYLE\n::cue { color: white }\n\nREGION\nid:main\n\nfirst\n00:00.200 --> 00:00:03.500 align:start\n<v Lecturer>Hello.</v>\n<c.green>你好。</c>\n第三行。\n'
ASS = '[Script Info]\nTitle: Authored fixture\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:00.20,0:00:03.50,Default,,0,0,0,,{\\b1}Hello.\\N你好。\\N第三行。\n'

class SubtitleImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def write(self, ext, text):
        path = self.root / ('source.' + ext)
        path.write_text(text, encoding='utf-8')
        return path
    def test_three_formats_keep_timing_and_every_text_line(self):
        for ext, text in [('srt', SRT), ('vtt', VTT), ('ass', ASS)]:
            with self.subTest(ext=ext):
                path = self.write(ext, text)
                before = path.read_bytes()
                canonical = normalized_subtitles(path)
                cue, = parse_srt(canonical)
                self.assertEqual((cue.start, cue.end), (.2, 3.5))
                self.assertEqual(cue.speech_text, '你好。 第三行。')
                self.assertEqual(cue.english_text, 'Hello.')
                self.assertEqual(before, path.read_bytes())
    def test_vtt_crlf_first_cue_identifier_settings_multiple_headers(self):
        text = 'WEBVTT\n\n00:00.100 --> 00:01.000\n你好\n\nWEBVTT - second segment\nX-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:0\n\n2\n00:00:01.000 --> 00:02.000 position:10%\n再见\n'
        data = ASRData.from_subtitle_file(str(self.write('vtt', text.replace('\n','\r\n'))), strict=True)
        self.assertEqual([s.text for s in data], ['你好', '再见'])
        self.assertEqual([s.start_time for s in data], [100, 1000])
        default = ASRData.from_vtt('WEBVTT\n\n00:00:00.100 --> 00:00:01.000\n你好\n')
        self.assertEqual(len(default), 1)
        self.assertEqual(len(ASRData.from_vtt('WEBVTT\n\nNOTE\n00:00:00.000 --> 00:00:01.000\nnot a cue\n')), 0)
    def test_ass_reordered_fields_commas_linebreaks_bilingual_events(self):
        text = '[Events]\nFormat: Start, End, Style, Text\nDialogue: 0:00:00.00,0:00:01.00,Secondary,Hello, world\nDialogue: 0:00:00.00,0:00:01.00,Default,{\\i1}你好\\h世界\\N再见\n'
        cue, = parse_srt(normalized_subtitles(self.write('ass', text)))
        self.assertEqual(cue.english_text, 'Hello, world')
        self.assertIn('再见', cue.speech_text)
    def test_malformed_entries_not_silently_skipped(self):
        cases = [('vtt', VTT + '\nsecond\nbroken --> 00:04.000\n你好\n'),
                 ('srt', SRT + '\n2\nBAD\n你好\n'),
                 ('ass', ASS + 'Dialogue: incomplete\n'),
                 ('vtt', VTT.replace('WEBVTT', 'WEBVTT\nX-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:900000')),
                 ('ass', ASS + 'Dialogue broken\n'),
                 ('ass', ASS + 'unexpected continuation\n'),
                 ('ass', ASS + 'Dialogue: 0,0:00:00.20,0:00:03.50,Default,,0,0,0,,{\\b1}\n'),
                 ('vtt', 'WEBVTT\n\n00:70.000 --> 01:20.000\n你好\n'),
                 ('vtt', 'WEBVTT\n\n00:01.000 --> 00:02.000\n你好\n\n00:00.000 --> 00:00.500\n再见\n'),
                 ('vtt', 'WEBVTT\n\n00:00.000 --> 00:01.000\n\n'),
                 ('ass', ASS.replace('Hello.', '{\\p1}m 0 0 l 1 1'))]
        for ext, text in cases:
            with self.subTest(text=text):
                with self.assertRaises(DubbingError):
                    normalized_subtitles(self.write(ext, text))
    def test_overlap_rejected_after_import(self):
        text = VTT + '\n00:02.000 --> 00:04.000\n再见\n'
        with self.assertRaises(DubbingError):
            parse_srt(normalized_subtitles(self.write('vtt', text)))
    def test_html_entities_are_speech_not_literal_entity_codes(self):
        cue, = parse_srt(normalized_subtitles(self.write('vtt', VTT.replace('你好。', '你好 &amp; 世界。'))))
        self.assertIn('你好 & 世界。', cue.speech_text)

    def test_filter_uses_supported_formats(self):
        from app.core.entities import SupportedSubtitleFormats
        for fmt in SupportedSubtitleFormats:
            self.assertIn('*.' + fmt.value, subtitle_file_filter())

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
    def test_each_format_actual_wav_snapshot_and_source_protection(self):
        video = self.root / 'video.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=s=160x90:d=4','-f','lavfi','-i','sine=duration=4','-c:v','mpeg4','-c:a','aac','-t','4',str(video)],check=True)
        for ext, text in [('srt',SRT),('vtt',VTT),('ass',ASS)]:
            with self.subTest(ext=ext):
                path = self.write(ext,text)
                settings = normalized({})
                request = build_request(video,path,self.root/(ext+'.wav'),settings,runtime={},output_mode='audio',cache_dir=self.root/('cache-'+ext))
                path.write_text('later edit', encoding='utf-8')
                provider = FakeProvider()
                output = execute_request(request,provider=provider)
                self.assertTrue(Path(output).is_file())
                self.assertEqual(provider.calls,['你好。 第三行。'])
                self.assertEqual(path.read_text('utf-8'),'later edit')
                path.write_text(text,encoding='utf-8')
                with self.assertRaises(DubbingError):
                    build_request(video,path,path,settings,runtime={},output_mode='audio')
                mux = build_request(video,path,self.root/(ext+'-soft.mp4'),settings,runtime={},output_mode='video',subtitle_mode='soft',cache_dir=self.root/('mux-cache-'+ext))
                mux_output = execute_request(mux,provider=FakeProvider())
                probe = subprocess.check_output(['ffprobe','-v','error','-select_streams','s','-show_entries','stream=codec_name','-of','csv=p=0',mux_output],text=True)
                self.assertIn('mov_text',probe)
                # Direct pipeline (without request builder) uses the same importer.
                config = DubbingConfig(str(video),str(path),str(self.root/(ext+'-direct.wav')),str(self.root/('direct-cache-'+ext)),output_mode='audio')
                DubbingPipeline(config,provider=FakeProvider()).run()
                self.assertEqual(path.read_text('utf-8'),text)

if __name__ == '__main__':
    unittest.main()
