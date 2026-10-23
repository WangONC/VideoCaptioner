from re import S
from typing import List, Dict, Optional
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from pathlib import Path
import queue
import time
from functools import partial


from app.core.task_factory import TaskFactory
from app.core.entities import (
    TranscribeTask,
    SubtitleTask,
    TranscriptAndSubtitleTask,
    FullProcessTask,
)
from app.thread.transcript_thread import TranscriptThread
from app.thread.subtitle_thread import SubtitleThread
from app.thread.video_synthesis_thread import VideoSynthesisThread
from app.core.utils.logger import setup_logger
from app.core.entities import BatchTaskType, BatchTaskStatus
from app.common.config import cfg

logger = setup_logger("batch_process_thread")


class BatchTask:
    def __init__(self, file_path: str, task_type: BatchTaskType, dubbing_request=None):
        self.file_path = file_path
        self.task_type = task_type
        self.status = BatchTaskStatus.WAITING
        self.progress = 0
        self.error_message = ""
        self.current_thread: Optional[QThread] = None
        self.dubbing_request = dubbing_request
        self.cancelled = False
        self.output_path = None


class BatchProcessThread(QThread):
    # 信号定义
    task_progress = pyqtSignal(str, int, str)  # file_path, progress, status
    task_error = pyqtSignal(str, str)  # file_path, error_message
    task_completed = pyqtSignal(str)  # file_path

    def __init__(self):
        super().__init__()
        self.task_queue = queue.Queue()
        self.current_tasks: Dict[str, BatchTask] = {}
        self.max_concurrent_tasks = 1
        self.is_running = False
        self.factory = TaskFactory()
        self.threads = []  # 保存所有创建的线程

    def add_task(self, task: BatchTask):
        existing = self.current_tasks.get(task.file_path)
        if existing and existing.status in (BatchTaskStatus.WAITING, BatchTaskStatus.RUNNING) and not existing.cancelled:
            return False
        self.current_tasks[task.file_path] = task
        self.task_queue.put(task)
        if not self.isRunning():
            self.is_running = True
            self.start()

    def run(self):
        while self.is_running:
            # 检查是否有正在运行的任务数量是否达到上限
            running_tasks = sum(
                1
                for task in list(self.current_tasks.values())
                if task.status == BatchTaskStatus.RUNNING
            )

            # A cancelled task may be removed from the UI while its worker is
            # still stopping FFmpeg/model processes. Do not overlap the next GPU job.
            live_dubbing = sum(1 for thread in list(self.threads)
                               if getattr(thread, 'operation', None) == 'dub' and thread.isRunning())
            running_tasks = max(running_tasks, live_dubbing)
            if running_tasks < self.max_concurrent_tasks:
                try:
                    # 非阻塞方式获取任务
                    task = self.task_queue.get_nowait()
                    if not task.cancelled and self.current_tasks.get(task.file_path) is task:
                        self._process_task(task)
                except queue.Empty:
                    time.sleep(0.1)  # 避免CPU过度使用
            else:
                time.sleep(0.1)

    def _process_task(self, batch_task: BatchTask):
        try:
            batch_task.status = BatchTaskStatus.RUNNING
            self.task_progress.emit(
                batch_task.file_path, 0, str(BatchTaskStatus.RUNNING)
            )

            if batch_task.task_type == BatchTaskType.TRANSCRIBE:
                self._handle_transcribe_task(batch_task)
            elif batch_task.task_type == BatchTaskType.SUBTITLE:
                self._handle_subtitle_task(batch_task)
            elif batch_task.task_type == BatchTaskType.TRANS_SUB:
                self._handle_trans_sub_task(batch_task)
            elif batch_task.task_type == BatchTaskType.TRANS_SPLIT:
                self._handle_trans_split_task(batch_task)
            elif batch_task.task_type == BatchTaskType.FULL_PROCESS:
                self._handle_full_process_task(batch_task)
            elif batch_task.task_type == BatchTaskType.SYNTHESIS:
                self._handle_synthesis_task(batch_task)
            elif batch_task.task_type in (BatchTaskType.DUBBING, BatchTaskType.DUBBING_SYNTHESIS):
                self._handle_dubbing_task(batch_task)

        except Exception as e:
            logger.exception(f"处理任务失败: {str(e)}")
            batch_task.status = BatchTaskStatus.FAILED
            batch_task.error_message = str(e)
            self.task_error.emit(batch_task.file_path, str(e))

    def _handle_dubbing_task(self, batch_task):
        from app.thread.dubbing_thread import DubbingThread
        if batch_task.dubbing_request is None:
            raise ValueError("缺少配音任务配置，请选择字幕后重试")
        thread = DubbingThread(batch_task.dubbing_request)
        thread.moveToThread(self.thread())
        batch_task.current_thread = thread
        self.threads.append(thread)
        thread.progress.connect(partial(self._on_progress_wrapper, batch_task), Qt.QueuedConnection)
        thread.failed.connect(partial(self._on_error_wrapper, batch_task), Qt.QueuedConnection)
        thread.cancelled.connect(partial(self._on_error_wrapper, batch_task, "已取消，完成的句子已缓存"), Qt.QueuedConnection)
        thread.succeeded.connect(partial(self._on_dubbing_succeeded, batch_task), Qt.QueuedConnection)
        thread.finished.connect(partial(self._release_thread, thread), Qt.QueuedConnection)
        if batch_task.cancelled:
            thread.stop()
        thread.start()

    def _release_thread(self, thread):
        if thread in self.threads:
            self.threads.remove(thread)
        thread.deleteLater()

    def _on_dubbing_succeeded(self, batch_task, output):
        if batch_task.cancelled or self.current_tasks.get(batch_task.file_path) is not batch_task:
            return
        batch_task.output_path = output
        self._on_finished_wrapper(batch_task)

    def _on_progress_wrapper(self, batch_task: BatchTask, progress: int, message: str):
        """进度信号包装器"""
        if batch_task.cancelled or batch_task.status == BatchTaskStatus.FAILED or self.current_tasks.get(batch_task.file_path) is not batch_task:
            return
        self.task_progress.emit(batch_task.file_path, progress, message)

    def _on_error_wrapper(self, batch_task: BatchTask, error: str):
        """错误信号包装器"""
        if batch_task.cancelled or self.current_tasks.get(batch_task.file_path) is not batch_task:
            return
        batch_task.status = BatchTaskStatus.FAILED
        batch_task.error_message = error
        self.task_error.emit(batch_task.file_path, error)

    def _on_finished_wrapper(self, batch_task: BatchTask, task=None):
        """完成信号包装器"""
        if batch_task.cancelled or batch_task.status == BatchTaskStatus.FAILED or self.current_tasks.get(batch_task.file_path) is not batch_task:
            return
        batch_task.status = BatchTaskStatus.COMPLETED
        batch_task.progress = 100
        self.task_completed.emit(batch_task.file_path)
        if batch_task.dubbing_request is None and batch_task.current_thread in self.threads:
            self.threads.remove(batch_task.current_thread)

    def _handle_transcribe_task(self, batch_task: BatchTask):
        # self.max_concurrent_tasks = 3
        task = self.factory.create_transcribe_task(batch_task.file_path)
        thread = TranscriptThread(task)
        batch_task.current_thread = thread

        # 保存线程引用
        self.threads.append(thread)

        thread.progress.connect(
            partial(self._on_progress_wrapper, batch_task), Qt.QueuedConnection
        )
        thread.error.connect(
            partial(self._on_error_wrapper, batch_task), Qt.QueuedConnection
        )
        thread.finished.connect(
            partial(self._on_finished_wrapper, batch_task), Qt.QueuedConnection
        )

        thread.start()

    def _handle_subtitle_task(self, batch_task: BatchTask):
        logger.info(f"开始处理字幕任务: {batch_task.file_path}")

        task = self.factory.create_subtitle_task(batch_task.file_path)
        thread = SubtitleThread(task)
        batch_task.current_thread = thread

        # 保存线程引用
        self.threads.append(thread)

        thread.progress.connect(
            partial(self._on_progress_wrapper, batch_task), Qt.QueuedConnection
        )
        thread.error.connect(
            partial(self._on_error_wrapper, batch_task), Qt.QueuedConnection
        )
        thread.finished.connect(
            partial(self._on_finished_wrapper, batch_task), Qt.QueuedConnection
        )

        thread.start()

    def _handle_trans_split_task(self, batch_task: BatchTask):
        trans_task = self.factory.create_transcribe_task(
            batch_task.file_path, need_next_task=True
        )
        trans_task.transcribe_config.need_word_time_stamp = True
        thread = TranscriptThread(trans_task)
        batch_task.current_thread = thread
        self.current_tasks[batch_task.file_path] = batch_task

        # 保存线程引用
        self.threads.append(thread)

        thread.progress.connect(
            partial(self._on_trans_sub_progress_wrapper, batch_task),
            Qt.QueuedConnection,
        )
        thread.error.connect(
            partial(self._on_error_wrapper, batch_task), Qt.QueuedConnection
        )
        thread.finished.connect(
            partial(self._on_trans_split_finished_wrapper, batch_task),
            Qt.QueuedConnection,
        )

        thread.start()

    def _on_trans_split_finished_wrapper(
        self, batch_task: BatchTask, task: TranscribeTask
    ):
        if batch_task.current_thread in self.threads:
            self.threads.remove(batch_task.current_thread)

        subtitle_task = self.factory.create_subtitle_split_task(
            task.output_path, batch_task.file_path, need_next_task=True
        )
        video_path = Path(batch_task.file_path)
        subtitle_task.output_path = str(
            video_path.parent / f"【断句字幕】{video_path.stem}.srt"
        )
        thread = SubtitleThread(subtitle_task)
        batch_task.current_thread = thread
        self.current_tasks[batch_task.file_path] = batch_task

        # 保存线程引用
        self.threads.append(thread)

        thread.progress.connect(
            partial(self._on_trans_sub_subtitle_progress_wrapper, batch_task),
            Qt.QueuedConnection,
        )
        thread.error.connect(
            partial(self._on_error_wrapper, batch_task), Qt.QueuedConnection
        )
        thread.finished.connect(
            partial(self._on_finished_wrapper, batch_task), Qt.QueuedConnection
        )

        thread.start()

    def _handle_trans_sub_task(self, batch_task: BatchTask):
        task = self.factory.create_transcript_and_subtitle_task(batch_task.file_path)
        trans_task = self.factory.create_transcribe_task(
            batch_task.file_path, need_next_task=True
        )
        thread = TranscriptThread(trans_task)
        batch_task.current_thread = thread
        self.current_tasks[batch_task.file_path] = batch_task

        # 保存线程引用
        self.threads.append(thread)

        thread.progress.connect(
            partial(self._on_trans_sub_progress_wrapper, batch_task),
            Qt.QueuedConnection,
        )
        thread.error.connect(
            partial(self._on_error_wrapper, batch_task), Qt.QueuedConnection
        )
        thread.finished.connect(
            partial(self._on_trans_sub_finished_wrapper, batch_task),
            Qt.QueuedConnection,
        )

        thread.start()

    def _on_trans_sub_progress_wrapper(
        self, batch_task: BatchTask, progress: int, message: str
    ):
        """转录+字幕任务进度包装器"""
        if batch_task.status == BatchTaskStatus.FAILED:
            return
        progress = progress // 2  # 转录占50%进度
        self.task_progress.emit(batch_task.file_path, progress, message)

    def _on_trans_sub_finished_wrapper(
        self, batch_task: BatchTask, task: TranscribeTask
    ):
        """转录+字幕任务转录完成包装器"""
        if batch_task.current_thread in self.threads:
            self.threads.remove(batch_task.current_thread)

        # 创建字幕任务
        subtitle_task = self.factory.create_subtitle_task(
            task.output_path, batch_task.file_path, need_next_task=True
        )
        thread = SubtitleThread(subtitle_task)
        batch_task.current_thread = thread
        self.current_tasks[batch_task.file_path] = batch_task

        # 保存线程引用
        self.threads.append(thread)

        from functools import partial

        thread.progress.connect(
            partial(self._on_trans_sub_subtitle_progress_wrapper, batch_task),
            Qt.QueuedConnection,
        )
        thread.error.connect(
            partial(self._on_error_wrapper, batch_task), Qt.QueuedConnection
        )
        thread.finished.connect(
            partial(self._on_finished_wrapper, batch_task), Qt.QueuedConnection
        )

        thread.start()

    def _on_trans_sub_subtitle_progress_wrapper(
        self, batch_task: BatchTask, progress: int, message: str
    ):
        """转录+字幕任务字幕进度包装器"""
        if batch_task.status == BatchTaskStatus.FAILED:
            return
        progress = 50 + progress // 2  # 字幕处理占后50%进度
        self.task_progress.emit(batch_task.file_path, progress, message)

    def _handle_synthesis_task(self, batch_task: BatchTask):
        """处理视频合成任务"""

        # 根据配置寻找命名的字幕文件
        video_name = Path(batch_task.file_path).stem
        suffix = (
            f"-{cfg.translator_service.value.value}" if cfg.need_translate.value else ""
        )
        subtitle_path = str(
                Path(batch_task.file_path).parent / f"【字幕】{video_name}{suffix}.srt"
            )

        
        # 创建视频合成任务
        synthesis_task = self.factory.create_synthesis_task(batch_task.file_path, subtitle_path)
        thread = VideoSynthesisThread(synthesis_task)
        batch_task.current_thread = thread

        # 保存线程引用
        self.threads.append(thread)

        thread.progress.connect(
            partial(self._on_progress_wrapper, batch_task), Qt.QueuedConnection
        )
        thread.error.connect(
            partial(self._on_error_wrapper, batch_task), Qt.QueuedConnection
        )
        thread.finished.connect(
            partial(self._on_finished_wrapper, batch_task), Qt.QueuedConnection
        )

        thread.start()

    def _handle_full_process_task(self, batch_task: BatchTask):
        task = self.factory.create_full_process_task(batch_task.file_path)
        # 首先创建转录任务
        trans_task = self.factory.create_transcribe_task(
            batch_task.file_path, need_next_task=True
        )
        thread = TranscriptThread(trans_task)
        batch_task.current_thread = thread

        # 保存线程引用
        self.threads.append(thread)

        thread.progress.connect(
            partial(self.on_full_process_progress, batch_task), Qt.QueuedConnection
        )
        thread.error.connect(
            partial(self._on_error_wrapper, batch_task), Qt.QueuedConnection
        )
        thread.finished.connect(
            partial(self.on_full_process_finished, batch_task), Qt.QueuedConnection
        )

        thread.start()

    def on_full_process_progress(
        self, batch_task: BatchTask, progress: int, message: str
    ):
        """处理全流程任务的转录进度"""
        if batch_task.status == BatchTaskStatus.RUNNING:
            progress_value = progress // 3  # 转录占33%进度
            self.task_progress.emit(batch_task.file_path, progress_value, message)

    def on_full_process_finished(self, batch_task: BatchTask, task: TranscribeTask):
        """处理转录完成后开始字幕任务"""
        if batch_task.current_thread in self.threads:
            self.threads.remove(batch_task.current_thread)

        # 转录完成后创建字幕任务
        subtitle_task = self.factory.create_subtitle_task(
            task.output_path,
            batch_task.file_path,
            need_next_task=True,
        )
        thread = SubtitleThread(subtitle_task)
        batch_task.current_thread = thread

        # 保存线程引用
        self.threads.append(thread)

        thread.progress.connect(
            partial(self.on_full_process_subtitle_progress, batch_task),
            Qt.QueuedConnection,
        )
        thread.error.connect(
            partial(self._on_error_wrapper, batch_task), Qt.QueuedConnection
        )
        thread.finished.connect(
            partial(self.on_full_process_subtitle_finished, batch_task),
            Qt.QueuedConnection,
        )

        thread.start()

    def on_full_process_subtitle_progress(
        self, batch_task: BatchTask, progress: int, message: str
    ):
        """处理全流程任务中字幕部分的进度"""
        if batch_task.status == BatchTaskStatus.RUNNING:
            progress_value = 33 + progress // 3  # 字幕处理占中间33%进度
            self.task_progress.emit(batch_task.file_path, progress_value, message)

    def on_full_process_subtitle_finished(
        self, batch_task: BatchTask, video_path: str, subtitle_path: str
    ):
        """处理字幕完成后开始视频合成任务"""
        if batch_task.current_thread in self.threads:
            self.threads.remove(batch_task.current_thread)

        # 字幕完成后创建视频合成任务
        synthesis_task = self.factory.create_synthesis_task(video_path, subtitle_path)
        thread = VideoSynthesisThread(synthesis_task)
        batch_task.current_thread = thread

        # 保存线程引用
        self.threads.append(thread)

        thread.progress.connect(
            partial(self.on_full_process_synthesis_progress, batch_task),
            Qt.QueuedConnection,
        )
        thread.error.connect(
            partial(self._on_error_wrapper, batch_task), Qt.QueuedConnection
        )
        thread.finished.connect(
            partial(self._on_finished_wrapper, batch_task), Qt.QueuedConnection
        )

        thread.start()

    def on_full_process_synthesis_progress(
        self, batch_task: BatchTask, progress: int, message: str
    ):
        """处理全流程任务中视频合成部分的进度"""
        if batch_task.status == BatchTaskStatus.RUNNING:
            progress_value = 66 + progress // 3  # 视频合成占最后34%进度
            self.task_progress.emit(batch_task.file_path, progress_value, message)

    def stop_task(self, file_path: str):
        task = self.current_tasks.pop(file_path, None)
        if task:
            task.cancelled = True
            if task.current_thread and hasattr(task.current_thread, 'stop'):
                task.current_thread.stop()
        # Queued entries are skipped by identity/cancel checks; do not drop other tasks.

    def stop_all(self):
        self.is_running = False
        for task in list(self.current_tasks.values()):
            task.cancelled = True
        # Stop queue dispatch before collecting workers, avoiding a late-start race.
        if self.isRunning() and QThread.currentThread() is not self:
            self.wait()
        for thread in list(self.threads):
            if hasattr(thread, 'stop'):
                thread.stop()
        for thread in list(self.threads):
            thread.wait()
        self.threads.clear()
        self.current_tasks.clear()
        with self.task_queue.mutex:
            self.task_queue.queue.clear()
