"""Shared execution contracts: synthetic voices, real FFmpeg, no GPU claim."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import wave

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QCoreApplication, QEvent
from types import SimpleNamespace
from PyQt5.QtTest import QTest
from app.core.dubbing.execution import build_request, execute_request, backend_choices
from app.core.dubbing.pipeline import DubbingError, CancelledError
from app.core.dubbing.settings import normalized
from app.thread.dubbing_thread import DubbingThread
from app.thread.batch_process_thread import BatchProcessThread, BatchTask
from app.core.entities import BatchTaskType, BatchTaskStatus
from test_dubbing_core import SRT, FakeProvider, tone

APP = QApplication.instance() or QApplication([])


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class SharedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / 'source.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
            'color=c=blue:s=160x90:r=25:d=6', '-f', 'lavfi', '-i', 'sine=frequency=220:duration=6',
            '-c:v', 'mpeg4', '-c:a', 'aac', '-t', '6', str(self.video)], check=True)
        self.subtitle = self.root / 'source.srt'
        self.subtitle.write_text(SRT, encoding='utf-8')
        backend_choices.clear()

    def tearDown(self):
        backend_choices.clear()
        self.temp.cleanup()

    def request(self, mode='video', subtitle_mode='none', output=None, **kwargs):
        return build_request(self.video, self.subtitle,
            output or self.root / ('output.wav' if mode == 'audio' else 'output.mp4'),
            kwargs.pop('settings', normalized({})), runtime={}, output_mode=mode,
            subtitle_mode=subtitle_mode, cache_dir=self.root / 'cache', **kwargs)

    def streams(self, output):
        return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
            '-show_format', '-of', 'json', str(output)]))

    def test_request_snapshot_isolated_from_settings_runtime_and_subtitle_edits(self):
        settings = normalized({'speed': 1.15, 'reference_text': 'before'})
        runtime = {'python_path': '/before/python', 'repo_path': '/before/repo', 'model_dir': '/before/model'}
        request = build_request(self.video, self.subtitle, self.root / 'snapshot.wav', settings,
            runtime=runtime, output_mode='audio', cache_dir=self.root / 'cache')
        settings.update(speed=2, reference_text='after')
        runtime['python_path'] = '/after/python'
        self.subtitle.write_text('INVALID LATER EDIT', encoding='utf-8')
        self.assertEqual(request.config().speed, 1.15)
        self.assertEqual(request.config().python_path, '/before/python')
        self.assertEqual(request.subtitle_text, SRT)
        # Even mutation of the returned config copy cannot alter a queued request.
        request.config().speed = 2
        self.assertEqual(request.config().speed, 1.15)
        provider = FakeProvider()
        output = execute_request(request, provider=provider)
        self.assertEqual(provider.calls, ['你好，欢迎观看。', '谢谢。'])
        self.assertTrue(Path(output).is_file())
        self.assertEqual(self.subtitle.read_text('utf-8'), 'INVALID LATER EDIT')

    def test_association_overrides_reference_and_subtitle_without_mutating_common_settings(self):
        reference = self.root / 'reference.wav'; tone(reference, 4)
        subtitle = self.root / 'alternate.srt'; subtitle.write_text(SRT.replace('谢谢。', '再见。'), encoding='utf-8')
        settings = normalized({'reference_text': 'shared'})
        association = {'subtitle': str(subtitle), 'reference': str(reference), 'reference_text': 'associated'}
        request = self.request('audio', settings=settings, association=association)
        association.update(reference='/changed', reference_text='changed', subtitle='/changed')
        self.assertEqual(request.config().reference_audio, str(reference))
        self.assertEqual(request.config().reference_text, 'associated')
        self.assertIn('再见。', request.subtitle_text)
        self.assertEqual(settings['source'], '原视频声音')
        self.assertEqual(settings['reference_text'], 'shared')

    def test_audio_is_wav_without_video_mux(self):
        output = execute_request(self.request('audio'), provider=FakeProvider())
        with wave.open(output, 'rb') as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getframerate(), 24000)
            self.assertAlmostEqual(wav.getnframes() / wav.getframerate(), 6, places=2)
        self.assertEqual([s['codec_type'] for s in self.streams(output)['streams']], ['audio'])

    def test_video_subtitle_modes_are_distinct_and_keep_input(self):
        source_bytes = self.video.read_bytes(); subtitle_bytes = self.subtitle.read_bytes()
        for mode in ('none', 'soft', 'hard'):
            with self.subTest(mode=mode):
                output = execute_request(self.request(subtitle_mode=mode, output=self.root / (mode + '.mp4')),
                                         provider=FakeProvider())
                streams = self.streams(output)['streams']
                self.assertEqual(sum(s['codec_type'] == 'subtitle' for s in streams), int(mode == 'soft'))
                self.assertEqual(streams[0]['codec_name'], 'h264' if mode == 'hard' else 'mpeg4')
                self.assertAlmostEqual(float(self.streams(output)['format']['duration']), 6, delta=.1)
        self.assertEqual(self.video.read_bytes(), source_bytes)
        self.assertEqual(self.subtitle.read_bytes(), subtitle_bytes)

    def test_hard_subtitles_support_special_output_directory(self):
        directory = self.root / "teacher's [course]:第一课"
        directory.mkdir()
        output = execute_request(self.request(subtitle_mode='hard', output=directory / 'result.mp4'), provider=FakeProvider())
        self.assertTrue(Path(output).is_file())
        self.assertEqual(self.streams(output)['streams'][0]['codec_name'], 'h264')

    def test_cancel_during_synthesis_does_not_publish(self):
        cancel = threading.Event()
        def progress(done, total, message):
            if done == 85:
                cancel.set()
        request = self.request(subtitle_mode='soft')
        with self.assertRaises(CancelledError):
            execute_request(request, progress, cancel, FakeProvider())
        self.assertFalse(Path(request.config().output_path).exists())
        self.assertEqual(list(self.root.glob('.dubbing-job-*')), [])
        self.assertTrue(list((self.root / 'cache').glob('*.wav')))

    def test_conflicting_output_appears_during_synthesis_is_not_overwritten(self):
        request = self.request(subtitle_mode='soft')
        output = Path(request.config().output_path)
        def progress(done, total, message):
            if done == 85:
                output.write_bytes(b'external writer')
        with self.assertRaises(DubbingError):
            execute_request(request, progress, provider=FakeProvider())
        self.assertEqual(output.read_bytes(), b'external writer')
        self.assertEqual(list(self.root.glob('.dubbing-job-*')), [])

    def test_request_rejects_missing_and_protected_files_and_unsupported_model(self):
        for output in (self.video, self.subtitle):
            with self.subTest(output=output), self.assertRaises(DubbingError):
                self.request(output=output)
        with self.assertRaises(DubbingError):
            build_request(self.video, '', None, normalized({}), runtime={})
        request = self.request(settings=normalized({'model_id': 'edge-tts'}))
        self.assertIsNone(request.config().reference_audio)
        backend_choices['indextts2'] = 'audio.cpp'
        self.assertEqual(self.request().config().backend, 'audio.cpp')
        backend_choices['cosyvoice3'] = 'audio.cpp'
        with self.assertRaisesRegex(DubbingError, 'audio.cpp'):
            self.request(settings=normalized({'model_id': 'cosyvoice3'}))

    def test_single_ui_launches_immutable_request(self):
        from app.core.dubbing.settings import shared_settings
        from app.core.dubbing.execution import DubbingRequest
        from app.view.dubbing_interface import DubbingInterface
        shared_settings.update({})
        host = QWidget()
        try:
            page = DubbingInterface(host)
            shared_settings.update(normalized({'output': '配音音频', 'speed': 1.2}))
            page.set_inputs(self.video, self.subtitle)
            with patch.object(page, 'launch_job') as launch:
                page.start_dubbing()
                launch.assert_called_once()
                operation, request = launch.call_args.args
            self.assertEqual(operation, 'dub')
            self.assertIsInstance(request, DubbingRequest)
            self.assertEqual(request.config().output_mode, 'audio')
            self.assertEqual(Path(request.config().output_path).suffix, '.wav')
            self.assertEqual(request.config().speed, 1.2)
            shared_settings.update(normalized({'output': '配音视频', 'speed': 2}))
            self.assertEqual(request.config().speed, 1.2)
            self.assertEqual(request.config().output_mode, 'audio')
        finally:
            host.close(); host.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            APP.processEvents()
            shared_settings.update({})

    def test_batch_ui_snapshots_output_and_subtitle_synthesis_options(self):
        from app.core.dubbing.settings import shared_settings
        from app.view.batch_process_interface import BatchProcessInterface
        shared_settings.update({})
        host = QWidget()
        page = BatchProcessInterface(host)
        fake_cfg = SimpleNamespace(need_video=SimpleNamespace(value=True), soft_subtitle=SimpleNamespace(value=True))
        cases = ((BatchTaskType.DUBBING, True, True, 'audio', 'none'),
                 (BatchTaskType.DUBBING_SYNTHESIS, True, True, 'video', 'soft'),
                 (BatchTaskType.DUBBING_SYNTHESIS, True, False, 'video', 'hard'),
                 (BatchTaskType.DUBBING_SYNTHESIS, False, True, 'video', 'none'))
        requests = []
        try:
            for task_type, need_video, soft, output_mode, subtitle_mode in cases:
                with self.subTest(task_type=task_type, need_video=need_video, soft=soft):
                    page.clear_tasks()
                    page.task_type_combo.setCurrentText(str(task_type))
                    page.add_task_to_table(str(self.video))
                    page.dubbing_associations[str(self.video)] = {'subtitle': str(self.subtitle)}
                    fake_cfg.need_video.value, fake_cfg.soft_subtitle.value = need_video, soft
                    with patch('app.common.config.cfg', fake_cfg), patch.object(page.batch_thread, 'add_task') as add:
                        page.enqueue_task(0, str(self.video))
                        add.assert_called_once()
                        task = add.call_args.args[0]
                    self.assertEqual(task.task_type, task_type)
                    self.assertEqual(task.dubbing_request.config().output_mode, output_mode)
                    self.assertEqual(task.dubbing_request.subtitle_mode, subtitle_mode)
                    self.assertEqual(Path(task.dubbing_request.config().output_path).suffix, '.wav' if output_mode == 'audio' else '.mp4')
                    requests.append((task.dubbing_request, subtitle_mode))
            fake_cfg.need_video.value = False
            fake_cfg.soft_subtitle.value = False
            shared_settings.update(normalized({'speed': 1.8, 'output': '配音音频'}))
            for request, subtitle_mode in requests:
                self.assertEqual(request.subtitle_mode, subtitle_mode)
                self.assertEqual(request.config().speed, 1)
        finally:
            page.close(); host.close(); host.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            APP.processEvents()
            shared_settings.update({})

    def test_thread_success_failure_and_cancel_have_one_terminal_outcome(self):
        request = self.request('audio')
        for outcome in ('success', 'failure', 'cancel'):
            worker = DubbingThread(request)
            events = []
            worker.succeeded.connect(lambda result: events.append(('success', result)))
            worker.failed.connect(lambda error: events.append(('failure', error)))
            worker.cancelled.connect(lambda: events.append(('cancel', None)))
            effect = None if outcome == 'success' else (CancelledError('cancel') if outcome == 'cancel' else DubbingError('bad'))
            with patch('app.thread.dubbing_thread.execute_request', return_value='result.wav', side_effect=effect):
                worker.start()
                self.assertTrue(worker.wait(5000))
                APP.processEvents()
            self.assertEqual([kind for kind, _ in events], [outcome])
            worker.deleteLater()
            APP.processEvents()


class BatchLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.worker = BatchProcessThread()

    def tearDown(self):
        self.worker.stop_all()
        APP.processEvents()

    def test_cancel_queued_task_preserves_other_queue_entries(self):
        first = BatchTask('/first.mp4', BatchTaskType.DUBBING)
        second = BatchTask('/second.mp4', BatchTaskType.DUBBING)
        with patch.object(self.worker, 'start'):
            self.worker.add_task(first)
            self.worker.add_task(second)
        self.worker.stop_task(first.file_path)
        processed = []
        def process(task):
            processed.append(task.file_path)
            self.worker.is_running = False
        with patch.object(self.worker, '_process_task', side_effect=process):
            self.worker.run()
        self.assertTrue(first.cancelled)
        self.assertEqual(processed, [second.file_path])
        self.assertIs(self.worker.current_tasks[second.file_path], second)

    def test_failed_and_cancelled_tasks_cannot_be_marked_complete(self):
        completed = []; errors = []
        self.worker.task_completed.connect(completed.append)
        self.worker.task_error.connect(lambda *args: errors.append(args))
        task = BatchTask('/first.mp4', BatchTaskType.DUBBING)
        self.worker.current_tasks[task.file_path] = task
        task.status = BatchTaskStatus.RUNNING
        self.worker._on_error_wrapper(task, 'failure')
        self.worker._on_finished_wrapper(task)
        self.assertEqual(task.status, BatchTaskStatus.FAILED)
        self.assertEqual(completed, [])
        self.assertEqual(errors, [(task.file_path, 'failure')])
        task.status = BatchTaskStatus.RUNNING
        self.worker.stop_task(task.file_path)
        self.worker._on_dubbing_succeeded(task, 'result.wav')
        self.assertEqual(completed, [])
        self.assertIsNone(task.output_path)

    def test_old_callbacks_do_not_change_retried_task(self):
        old = BatchTask('/same.mp4', BatchTaskType.DUBBING)
        new = BatchTask('/same.mp4', BatchTaskType.DUBBING)
        old.status = BatchTaskStatus.FAILED
        self.worker.current_tasks[new.file_path] = new
        progress = []; errors = []; done = []
        self.worker.task_progress.connect(lambda *args: progress.append(args))
        self.worker.task_error.connect(lambda *args: errors.append(args))
        self.worker.task_completed.connect(done.append)
        self.worker._on_progress_wrapper(old, 90, 'stale')
        self.worker._on_error_wrapper(old, 'stale')
        self.worker._on_dubbing_succeeded(old, 'stale.wav')
        self.worker._on_finished_wrapper(old)
        self.assertEqual((progress, errors, done), ([], [], []))
        self.assertEqual(new.status, BatchTaskStatus.WAITING)

    def test_real_batch_workers_cancel_then_continue_without_stale_completion(self):
        from app.core.dubbing.execution import DubbingRequest
        entered = threading.Event()
        release = threading.Event()
        calls = []
        completed = []
        self.worker.task_completed.connect(completed.append)
        first_request = DubbingRequest((('output_path', '/first.wav'),), '')
        second_request = DubbingRequest((('output_path', '/second.wav'),), '')
        first = BatchTask('/first.mp4', BatchTaskType.DUBBING, first_request)
        second = BatchTask('/second.mp4', BatchTaskType.DUBBING, second_request)
        def execute(request, progress, cancel):
            path = dict(request.config_items)['output_path']
            calls.append(path)
            if path == '/first.wav':
                entered.set()
                if not cancel.wait(4):
                    raise AssertionError('Cancellation did not reach running worker')
                if not release.wait(4):
                    raise AssertionError('Test did not release cancelled worker')
                raise CancelledError('cancelled')
            progress(1, 1, 'done')
            return path
        with patch('app.thread.dubbing_thread.execute_request', side_effect=execute):
            self.worker.add_task(first)
            self.worker.add_task(second)
            deadline = time.monotonic() + 5
            while not entered.is_set() and time.monotonic() < deadline:
                QTest.qWait(20)
            self.assertTrue(entered.is_set())
            self.worker.stop_task(first.file_path)
            QTest.qWait(300)
            self.assertEqual(calls, ['/first.wav'], 'Next worker started before cancellation cleanup completed')
            release.set()
            deadline = time.monotonic() + 5
            while second.status != BatchTaskStatus.COMPLETED and time.monotonic() < deadline:
                QTest.qWait(20)
            self.assertEqual(second.status, BatchTaskStatus.COMPLETED)
            self.assertEqual(second.output_path, '/second.wav')
            self.assertEqual(completed, ['/second.mp4'])
            self.assertEqual(calls, ['/first.wav', '/second.wav'])
            self.worker.stop_all()
            APP.processEvents()

    def test_real_batch_failure_does_not_block_or_complete_failed_job(self):
        from app.core.dubbing.execution import DubbingRequest
        first = BatchTask('/failed.mp4', BatchTaskType.DUBBING, DubbingRequest((('output_path', '/failed.wav'),), ''))
        second = BatchTask('/good.mp4', BatchTaskType.DUBBING, DubbingRequest((('output_path', '/good.wav'),), ''))
        completed = []; errors = []
        self.worker.task_completed.connect(completed.append)
        self.worker.task_error.connect(lambda *args: errors.append(args))
        def execute(request, progress, cancel):
            output = dict(request.config_items)['output_path']
            if output == '/failed.wav':
                raise DubbingError('expected failure')
            return output
        with patch('app.thread.dubbing_thread.execute_request', side_effect=execute):
            self.worker.add_task(first); self.worker.add_task(second)
            deadline = time.monotonic() + 5
            while second.status != BatchTaskStatus.COMPLETED and time.monotonic() < deadline:
                QTest.qWait(20)
            self.assertEqual(first.status, BatchTaskStatus.FAILED)
            self.assertEqual(second.status, BatchTaskStatus.COMPLETED)
            self.assertEqual(completed, ['/good.mp4'])
            self.assertEqual(errors, [('/failed.mp4', 'expected failure')])
            self.worker.stop_all()
            APP.processEvents()

    def test_real_queued_cancellation_skips_only_cancelled_task(self):
        from app.core.dubbing.execution import DubbingRequest
        entered = threading.Event(); release = threading.Event()
        calls = []; completed = []
        self.worker.task_completed.connect(completed.append)
        tasks = [BatchTask('/' + name + '.mp4', BatchTaskType.DUBBING,
            DubbingRequest((('output_path', '/' + name + '.wav'),), ''))
            for name in ('first', 'cancelled', 'third')]
        def execute(request, progress, cancel):
            output = dict(request.config_items)['output_path']
            calls.append(output)
            if output == '/first.wav':
                entered.set()
                if not release.wait(4):
                    raise AssertionError('First job was not released')
            return output
        try:
            with patch('app.thread.dubbing_thread.execute_request', side_effect=execute):
                for task in tasks:
                    self.worker.add_task(task)
                deadline = time.monotonic() + 5
                while not entered.is_set() and time.monotonic() < deadline:
                    QTest.qWait(20)
                self.assertTrue(entered.is_set())
                self.worker.stop_task(tasks[1].file_path)
                release.set()
                deadline = time.monotonic() + 5
                while tasks[2].status != BatchTaskStatus.COMPLETED and time.monotonic() < deadline:
                    QTest.qWait(20)
                self.assertEqual(tasks[2].status, BatchTaskStatus.COMPLETED)
                self.assertEqual(calls, ['/first.wav', '/third.wav'])
                self.assertEqual(completed, ['/first.mp4', '/third.mp4'])
                self.assertTrue(tasks[1].cancelled)
                self.worker.stop_all()
                APP.processEvents()
        finally:
            release.set()

    def test_duplicate_active_task_not_enqueued(self):
        first = BatchTask('/same.mp4', BatchTaskType.DUBBING)
        second = BatchTask('/same.mp4', BatchTaskType.DUBBING)
        with patch.object(self.worker, 'start'):
            self.worker.add_task(first)
            self.assertFalse(self.worker.add_task(second))
        self.assertEqual(self.worker.task_queue.qsize(), 1)
        self.assertIs(self.worker.current_tasks[first.file_path], first)


if __name__ == '__main__':
    unittest.main()
