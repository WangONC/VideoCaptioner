"""Native Qt dubbing page; inference runs in a dedicated, cancellable worker."""
import json
import threading
from pathlib import Path
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap, QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QDialog, QFormLayout, QTableWidgetItem, QHeaderView
from qfluentwidgets import (BodyLabel, CaptionLabel, SubtitleLabel, CardWidget, LineEdit,
    ComboBox, PrimaryPushButton, PushButton, ProgressBar, ProgressRing, CheckBox, TextEdit, InfoBar)
from app.core.tts_catalog import MODELS
from app.config import APPDATA_PATH, MODEL_PATH, CACHE_PATH, RESOURCE_PATH
from qfluentwidgets import SingleDirectionScrollArea, CommandBar, Action, FluentIcon, TransparentDropDownPushButton, RoundMenu, HyperlinkCard, MessageBoxBase, TableWidget, HyperlinkButton


class DubbingJob(QThread):
    progress = pyqtSignal(int, str)
    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str)
    def __init__(self, operation, config, parent=None):
        super().__init__(parent)
        self.operation, self.config = operation, config
        self.cancel_event = threading.Event()
        self.runtime_ready = False
    def run(self):
        from app.core.dubbing.execution import execution_lane
        try:
            with execution_lane(self.cancel_event):
                self.run_setup()
        except Exception as error:
            self.failed.emit(str(error))

    def run_setup(self):
        try:
            if self.operation not in ('check_runtime', 'check_models', 'runtime', 'models'):
                raise ValueError('未知环境操作')
            if self.config.get('backend') == 'audio.cpp':
                from app.core import audio_cpp_runtime as tts_runtime
            else:
                from app.core import tts_runtime
            if self.operation == 'check_runtime':
                state = tts_runtime.inspect_runtime(self.config, cancel=self.cancel_event)
                if not state['ready']:
                    raise RuntimeError(state['message'])
                self.runtime_ready = True
                result = state['message']
            elif self.operation == 'check_models':
                self.progress.emit(0, '正在校验本地模型文件…')
                if self.config.get('backend') == 'audio.cpp':
                    ready = tts_runtime.models_ready(self.config, cancel=self.cancel_event, verify_hash=True)
                else:
                    ready = tts_runtime.verify_models(self.config['model_dir'], self.config['model_id'],
                        cancel=self.cancel_event, full=True, repo_path=self.config['repo_path'])
                if not ready:
                    raise RuntimeError('模型文件缺失或校验失败，请重新下载')
                result = '模型文件校验通过'
            else:
                function = tts_runtime.prepare_runtime if self.operation == 'runtime' else tts_runtime.download_models
                function(self.progress.emit, self.cancel_event, config=self.config)
                self.runtime_ready = self.operation == 'runtime'
                result = '运行环境已准备' if self.operation == 'runtime' else '模型下载完成'
            self.succeeded.emit(str(result))
        except Exception as error:
            self.failed.emit(str(error))


class DubbingInterface(QWidget):
    finished = pyqtSignal(str)
    setup_finished = pyqtSignal()
    model_size_updated = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('DubbingInterface')
        self.setAcceptDrops(True)
        self.job = None
        self.last_output = None
        self.environment_checks = {}
        self._model_size_refresh_started = False
        self.model_id = "indextts2"
        from app.core.dubbing.execution import backend_choices, device_choices
        self.backend_choices = backend_choices
        self.device_choices = device_choices
        self.runtime_config = self.paths_for_model(self.model_id)
        from app.core.dubbing.settings import shared_settings
        self.dubbing_settings = shared_settings.value()
        self.setup_ui()
        shared_settings.changed.connect(self.apply_shared_settings)
        self.apply_shared_settings(self.dubbing_settings)

    @staticmethod
    def load_runtime_config():
        root = APPDATA_PATH / 'tts' / 'indextts2'
        paths = dict(python_path=str(root/'runtime'/'python.exe'), repo_path=str(root/'source'), model_dir=str(MODEL_PATH/'IndexTTS-2'))
        file = APPDATA_PATH/'dubbing-runtime.json'
        try:
            if file.exists():
                value = json.loads(file.read_text('utf8'))
                paths.update({key: str(value[key]) for key in paths if value.get(key)})
        except (OSError, ValueError):
            pass
        return paths

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = SingleDirectionScrollArea(orient=Qt.Vertical, parent=self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet('QScrollArea{background: transparent; border: none}')
        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet('QWidget{background: transparent}')
        layout = QVBoxLayout(self.scroll_content)
        layout.setSpacing(20)
        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area)
        self.command_bar = CommandBar(self)
        self.command_bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.command_bar.setFixedHeight(40)
        action = Action(FluentIcon.FOLDER, self.tr('打开视频'), self)
        action.triggered.connect(self.choose_video)
        self.command_bar.addAction(action)
        self.command_bar.addSeparator()
        model = self.model_button = TransparentDropDownPushButton('IndexTTS2', self, FluentIcon.SETTING)
        model.setMinimumWidth(180)
        model.setFixedHeight(34)
        menu = RoundMenu(parent=model)
        from app.core.dubbing.settings import MODEL_OPTIONS
        for key,spec in MODEL_OPTIONS.items():
            item=Action(FluentIcon.ROBOT,spec['name'],menu)
            item.triggered.connect(lambda checked=False,model_id=key:self.select_model(model_id))
            menu.addAction(item)
        model.setMenu(menu)
        self.command_bar.addWidget(model)
        layout.addWidget(self.command_bar)

        media = CardWidget(self)
        media.setFixedHeight(150)
        row = QHBoxLayout(media)
        row.setContentsMargins(20, 15, 20, 15)
        row.setSpacing(20)
        self.thumbnail = BodyLabel('', media)
        self.thumbnail.setFixedSize(208, 117)
        self.thumbnail.setAlignment(Qt.AlignCenter)
        self.thumbnail.setStyleSheet('background-color:#1E1F22;')
        pixmap = QPixmap(str(RESOURCE_PATH/'assets'/'default_thumbnail.jpg'))
        self.thumbnail.setPixmap(pixmap.scaled(208,117,Qt.KeepAspectRatio,Qt.SmoothTransformation))
        row.addWidget(self.thumbnail)
        info = QVBoxLayout()
        self.video_title = BodyLabel('请拖入视频文件', media)
        self.video_title.setWordWrap(True)
        self.video_title.setFont(QFont('Microsoft YaHei',14,QFont.Bold))
        self.video_detail = BodyLabel('', media)
        info.addWidget(self.video_title)
        details_row = QHBoxLayout()
        details_row.setSpacing(15)
        details_row.addWidget(self.video_detail)
        self.progress_bar = ProgressRing(media)
        self.progress_bar.setFixedSize(20, 20)
        self.progress_bar.setStrokeWidth(4)
        self.progress_bar.hide()
        details_row.addWidget(self.progress_bar)
        details_row.addStretch()
        info.addLayout(details_row)
        row.addLayout(info, 1)
        buttons = QVBoxLayout()
        self.open_button = PushButton('打开文件夹', media)
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_output)
        self.start_button = PrimaryPushButton('开始配音', media)
        self.start_button.clicked.connect(self.start_dubbing)
        self.cancel_button = PushButton('取消', media)
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(self.cancel_job)
        for button in (self.open_button,self.start_button,self.cancel_button):
            button.setFixedWidth(130)
            buttons.addWidget(button)
        row.addLayout(buttons)
        layout.addWidget(media)
        # The video path is represented by the existing media card, not a duplicate row.
        self.video_input = LineEdit(self)
        self.video_input.hide()

        inputs = self.input_card = CardWidget(self)
        fields = QVBoxLayout(inputs)
        fields.setContentsMargins(20,20,20,20)
        fields.setSpacing(15)
        self.subtitle_input = self.file_row(fields, '字幕文件', '选择或拖入中文字幕', self.choose_subtitle)
        self.subtitle_input.setToolTip('选择配音字幕')
        self.output_input = self.file_row(fields, '输出文件', '选择保存位置', self.choose_output)
        layout.addWidget(inputs)

        self.manage_model_card=HyperlinkCard('', '管理模型', FluentIcon.DOWNLOAD,
            '模型管理', '下载或配置配音模型', self)
        self.manage_model_card.linkButton.clicked.connect(self.show_model_manager)
        layout.addWidget(self.manage_model_card)
        # Setup actions live in the same model-manager pattern as transcription.
        self.download_button=self.manage_model_card.linkButton
        self.runtime_button=PushButton(self)
        self.runtime_button.hide()
        self.runtime_status=CaptionLabel('',self)
        self.runtime_status.hide()
        self.status_label=BodyLabel('',self)
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        self.details=TextEdit(self)
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(95)
        self.details.hide()
        layout.addWidget(self.details)
        from app.components.dubbing_settings import DubbingSettingsWidget
        from app.core.dubbing.settings import shared_settings
        self.settings_widget = DubbingSettingsWidget(self.dubbing_settings, self, inline=True)
        # The command bar owns model selection; common settings appear only once below.
        self.settings_widget.cards['model_id'].hide()
        self.settings_widget.changed.connect(shared_settings.update)
        layout.addWidget(self.settings_widget)
        layout.addStretch()
        self.video_input.textChanged.connect(self.update_ready)
        self.subtitle_input.textChanged.connect(self.update_ready)
        self.output_input.textChanged.connect(self.update_ready)
        self.update_ready()

    def file_row(self, layout, label, placeholder, handler):
        row = QHBoxLayout()
        name = BodyLabel(label, self); name.setFixedWidth(78)
        row.addWidget(name)
        field = LineEdit(self); field.setPlaceholderText(placeholder)
        row.addWidget(field, 1)
        button = PushButton('浏览', self); button.clicked.connect(handler)
        row.addWidget(button)
        layout.addLayout(row)
        return field

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择原始视频', '', '视频 (*.mp4 *.mkv *.mov *.avi *.webm *.m4v)')
        if path: self.set_video(path)

    def set_video(self, path):
        self.video_input.setText(path)
        self.video_title.setText(Path(path).name)
        self.video_detail.setText(f'{Path(path).stat().st_size / 1024 / 1024:.1f} MB' if Path(path).is_file() else '')
        from app.core.dubbing.execution import next_output
        mode = 'audio' if self.current_settings()['output'] == '配音音频' else 'video'
        self.output_input.setText(next_output(path, mode))

    def choose_subtitle(self):
        from app.core.dubbing.subtitles import subtitle_file_filter
        path, _ = QFileDialog.getOpenFileName(self, '选择中文字幕', '', subtitle_file_filter())
        if path: self.subtitle_input.setText(path)

    def choose_output(self):
        audio = self.current_settings()['output'] == '配音音频'
        suffix = '.wav' if audio else '.mp4'
        path, _ = QFileDialog.getSaveFileName(self, '保存配音', self.output_input.text(), 'WAV 音频 (*.wav)' if audio else 'MP4 视频 (*.mp4)')
        if path: self.output_input.setText(path if path.lower().endswith(suffix) else path + suffix)

    def set_inputs(self, video_path, subtitle_path):
        self.set_video(str(video_path))
        self.subtitle_input.setText(str(subtitle_path))

    def update_ready(self):
        running = self.job is not None
        valid = Path(self.video_input.text()).is_file() and Path(self.subtitle_input.text()).is_file() and bool(self.output_input.text().strip())
        from app.core.dubbing.settings import executable
        available = executable(self.current_settings())
        self.start_button.setText("开始配音" if available else "暂不可用")
        self.start_button.setEnabled(valid and not running and available)

    def current_settings(self):
        return dict(self.dubbing_settings)

    def show_dubbing_settings(self):
        self.scroll_area.verticalScrollBar().setValue(self.settings_widget.y())
        self.settings_widget.controls['source'].setFocus(Qt.OtherFocusReason)

    def apply_shared_settings(self, value):
        from app.core.dubbing.settings import MODEL_OPTIONS
        old_output = self.dubbing_settings.get('output')
        self.dubbing_settings = dict(value)
        if old_output != value['output'] and self.output_input.text():
            target = Path(self.output_input.text())
            self.output_input.setText(str(target.with_suffix('.wav' if value['output'] == '配音音频' else '.mp4')))
        self.model_id = value['model_id']
        self.model_button.setText(MODEL_OPTIONS[self.model_id]['name'])
        self.model_button.setFixedWidth(max(180, self.model_button.fontMetrics().horizontalAdvance(self.model_button.text()) + 80))
        self.command_bar.updateGeometry()
        if self.model_id in MODELS:
            self.runtime_config = self.paths_for_model(self.model_id)
        if self.settings_widget.value() != value:
            self.settings_widget.blockSignals(True)
            self.settings_widget.set_value(value)
            self.settings_widget.blockSignals(False)
        self.update_ready()

    def backend_for_model(self, model_id):
        return self.backend_choices.get(model_id, 'pytorch')

    def device_for_model(self, model_id):
        return self.device_choices.get(model_id, 'gpu')

    def set_device(self, model_id, device):
        if self.job is not None or device not in ('gpu', 'cpu'):
            return
        self.device_choices[model_id] = device
        if model_id == self.model_id:
            self.runtime_config = self.paths_for_model(model_id)
        self.setup_finished.emit()

    def set_backend(self, model_id, backend):
        if self.job is not None or backend not in ('pytorch', 'audio.cpp'):
            return
        if backend == 'audio.cpp':
            from app.core.audio_cpp_runtime import AUDIO_CPP_MODELS
            if model_id not in AUDIO_CPP_MODELS:
                return
        self.backend_choices[model_id] = backend
        if model_id == self.model_id:
            self.runtime_config = self.paths_for_model(model_id)
        self.setup_finished.emit()

    def environment_key(self, model_id):
        paths = self.paths_for_model(model_id)
        if self.backend_for_model(model_id) == 'audio.cpp':
            return ('audio.cpp', paths['python_path'], self.device_for_model(model_id))
        return ('pytorch', MODELS[model_id]['engine'], paths['python_path'], paths['repo_path'])

    def environment_state(self, model_id):
        paths = self.paths_for_model(model_id)
        backend = self.backend_for_model(model_id)
        if backend == 'audio.cpp':
            installed = Path(paths['python_path']).is_file()
        else:
            from app.core.tts_runtime import runtime_installed
            installed = runtime_installed(dict(paths, model_id=model_id, device=self.device_for_model(model_id)))
        check = self.environment_checks.get(self.environment_key(model_id))
        if check:
            ready, message = check
            return dict(installed=installed, ready=ready,
                        label='可用' if ready else '异常', message=message)
        return dict(installed=installed, ready=False,
                    label='待检查' if installed else '未安装',
                    message='环境已存在，尚未检查' if installed else '尚未下载运行环境')

    def model_downloaded(self, model_id):
        paths = self.paths_for_model(model_id)
        if self.backend_for_model(model_id) == 'audio.cpp':
            from app.core.audio_cpp_runtime import models_ready
        else:
            from app.core.tts_runtime import models_ready
        return models_ready(dict(paths, model_id=model_id))

    def model_download_size(self, model_id):
        if self.backend_for_model(model_id) == 'audio.cpp':
            from app.core.audio_cpp_runtime import required_models
            total = sum(spec['size'] for spec in required_models(model_id))
            return f'{total / 1e9:.2f} GB'
        if model_id == 'edge-tts':
            return '无需模型'
        from app.core.tts_model_metadata import get_cached_model_bytes, format_model_size
        total = get_cached_model_bytes(model_id)
        return format_model_size(total) if total is not None else '获取中'

    def refresh_model_sizes(self):
        if self._model_size_refresh_started:
            return
        self._model_size_refresh_started = True
        # Metadata is independent of model installation; never block Qt's UI thread.
        # Daemon workers cannot keep the application alive or destroy a running QThread.
        def refresh():
            from app.core.tts_model_metadata import fetch_model_bytes
            from app.core.utils.logger import setup_logger
            for model_id in MODELS:
                if model_id == 'edge-tts':
                    continue
                try:
                    fetch_model_bytes(model_id)
                except Exception as error:
                    setup_logger('dubbing').warning('Model size metadata %s: %s', model_id, error)
                try:
                    self.model_size_updated.emit(model_id)
                except RuntimeError:
                    return  # Qt page has been destroyed during shutdown.
        threading.Thread(target=refresh, name='dubbing-model-metadata', daemon=True).start()

    def edit_runtime(self):
        if self.model_id not in MODELS:
            self.status_label.setText("暂不可用")
            return
        from app.components.dubbing_model_dialog import DubbingEnvironmentDialog
        self.environment_dialog = DubbingEnvironmentDialog(self.window(), self, self.model_id)
        self.environment_dialog.show()

    def paths_for_model(self,model_id):
        from app.core.dubbing.execution import runtime_paths
        return runtime_paths(model_id)

    def select_model(self,model_id):
        if self.job is not None:return
        from app.core.dubbing.settings import shared_settings
        value = self.current_settings()
        value['model_id'] = model_id
        shared_settings.update(value)

    def show_model_manager(self):
        from app.components.dubbing_model_dialog import DubbingModelDialog
        self.model_dialog=DubbingModelDialog(self.window(),self)
        self.model_dialog.show()
        self.refresh_model_sizes()

    def start_setup(self, operation, model_id=None):
        target=model_id or self.model_id
        if target not in MODELS:
            self.status_label.setText("暂不可用")
            return
        if self.job is None: self.launch_job(operation, dict(self.paths_for_model(target),model_id=target,backend=self.backend_for_model(target),device=self.device_for_model(target)))

    def start_dubbing(self):
        if self.job is not None: return
        from app.core.dubbing.execution import build_request
        try:
            request = build_request(self.video_input.text(), self.subtitle_input.text(),
                self.output_input.text(), self.current_settings(), runtime=self.runtime_config)
            self.launch_job('dub', request)
        except Exception as error:
            self.on_failure(str(error))

    def launch_job(self, operation, config):
        self.details.clear();self.details.hide();self.progress_bar.setValue(0)
        self.progress_bar.setVisible(operation == 'dub')
        self._completion_text = None
        if operation == 'dub':
            from app.thread.dubbing_thread import DubbingThread
            self.job = DubbingThread(config, self)
            self.job.cancelled.connect(lambda: self.on_failure('已取消，完成的句子已缓存'))
        else:
            self.job = DubbingJob(operation, config, self)
        self.job.progress.connect(self.on_progress)
        self.job.succeeded.connect(self.on_success)
        self.job.failed.connect(self.on_failure)
        self.job.finished.connect(self.job_finished)
        self.download_button.setEnabled(False);self.runtime_button.setEnabled(False)
        self.command_bar.setEnabled(False);self.input_card.setEnabled(False)
        self.settings_widget.setEnabled(False)
        for field in (self.video_input,self.subtitle_input,self.output_input):field.setEnabled(False)
        self.cancel_button.show();self.cancel_button.setEnabled(True)
        self.status_label.setText('正在检查并准备任务…')
        self.update_ready()
        if operation == 'dub': self.start_button.setText('正在配音')
        self.job.start()

    def on_progress(self, value, message):
        self.progress_bar.setValue(max(0,min(value,100)))
        self.status_label.setText(message)
        if self.job and self.job.operation == 'dub':
            self.start_button.setText(message)

    def on_success(self, result):
        self.progress_bar.setValue(100)
        if self.job and self.job.operation=='dub':
            self.last_output=result;self.open_button.setEnabled(True)
            self.status_label.setText('配音完成')
            self._completion_text = '配音完成'
            self.start_button.setText('配音完成')
            self.finished.emit(result)
        else:
            if self.job and self.job.operation in ('runtime', 'check_runtime'):
                key = self.environment_key(self.job.config['model_id'])
                self.environment_checks[key] = (True, '运行环境可用')
            self.status_label.setText(result);self.runtime_status.setText(result)

    def on_failure(self, message):
        canceled=self.job is not None and self.job.cancel_event.is_set()
        if self.job and self.job.operation in ('runtime', 'check_runtime') and not canceled:
            ready = self.job.runtime_ready
            self.environment_checks[self.environment_key(self.job.config['model_id'])] = (ready, '运行环境可用' if ready else message)
        if canceled:
            self._completion_text = '开始配音'
            self.status_label.setText('已取消')
            InfoBar.info('已取消', '', duration=3000, parent=self)
        else:
            self._completion_text = '重新配音'
            self.status_label.setText('配音失败')
            self.status_label.setToolTip(message)
            from app.core.utils.logger import setup_logger
            logger = setup_logger('dubbing')
            logger.error('配音任务未完成: %s', message)
            summary = message.splitlines()[0]
            if not any('\u4e00' <= char <= '\u9fff' for char in summary):
                summary = '处理失败，请查看日志'
            InfoBar.error('配音失败' if self.job and self.job.operation == 'dub' else '操作失败', summary[:120], duration=5000, parent=self)
        self.details.hide()

    def job_finished(self):
        job=self.job;self.job=None
        if job:job.deleteLater()
        self.cancel_button.hide()
        self.download_button.setEnabled(True);self.runtime_button.setEnabled(True)
        self.command_bar.setEnabled(True);self.input_card.setEnabled(True)
        self.settings_widget.setEnabled(True)
        for field in (self.video_input,self.subtitle_input,self.output_input):field.setEnabled(True)
        self.update_ready()
        if job and job.operation == 'dub' and getattr(self, '_completion_text', None):
            self.start_button.setText(self._completion_text)
        self.setup_finished.emit()

    def cancel_job(self):
        if self.job:
            self.job.cancel_event.set();self.cancel_button.setEnabled(False)
            self.status_label.setText('正在取消')
            if self.job.operation == 'dub': self.start_button.setText('正在取消')

    def open_output(self):
        if self.last_output: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_output).parent)))

    def dragEnterEvent(self, event):
        if self.job is None and event.mimeData().hasUrls():event.acceptProposedAction()

    def dropEvent(self, event):
        if self.job is not None:return
        for url in event.mimeData().urls():
            path=url.toLocalFile()
            from app.core.entities import SupportedSubtitleFormats
            if Path(path).suffix.lower().lstrip('.') in {f.value for f in SupportedSubtitleFormats}:self.subtitle_input.setText(path)
            elif Path(path).suffix.lower() in ('.mp4','.mkv','.mov','.avi','.webm','.m4v'):self.set_video(path)
        event.acceptProposedAction()

    def closeEvent(self,event):
        if self.job is not None:
            self.cancel_job();event.ignore()
        else:super().closeEvent(event)
