"""Native Qt dubbing page; inference runs in a dedicated, cancellable worker."""
import json
import threading
from pathlib import Path
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap, QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QDialog, QFormLayout, QTableWidgetItem, QHeaderView
from qfluentwidgets import (BodyLabel, CaptionLabel, SubtitleLabel, CardWidget, LineEdit,
    ComboBox, PrimaryPushButton, PushButton, ProgressBar, CheckBox, TextEdit, InfoBar)
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
        try:
            if self.operation == 'dub':
                from app.core.dubbing import DubbingConfig, DubbingPipeline
                def progress(done, total, message):
                    self.progress.emit(int(100*done/max(total, 1)), message)
                DubbingPipeline(DubbingConfig(**self.config)).run(on_progress=progress, cancel_event=self.cancel_event)
                result = self.config['output_path']
            else:
                from app.core import tts_runtime
                if self.operation == 'check_runtime':
                    state = tts_runtime.inspect_runtime(self.config, cancel=self.cancel_event)
                    if not state['ready']:
                        raise tts_runtime.RuntimeSetupError(state['message'])
                    self.runtime_ready = True
                    result = state['message']
                elif self.operation == 'prepare':
                    tts_runtime.prepare_runtime(lambda v,m:self.progress.emit(v//2,m), self.cancel_event, config=self.config)
                    self.runtime_ready = True
                    tts_runtime.download_models(lambda v,m:self.progress.emit(50+v//2,m), self.cancel_event, config=self.config)
                    result = '环境与模型已准备'
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('DubbingInterface')
        self.setAcceptDrops(True)
        self.job = None
        self.last_output = None
        self.environment_checks = {}
        self.model_id = "indextts2"
        self.backend_choices = {}  # UI preview only; do not change inference configuration.
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
        info.addWidget(self.video_detail)
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
        self.subtitle_input.setToolTip('支持中文 SRT 或中英分行的双语 SRT。只合成中文行，原字幕不改动。')
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
        layout.addWidget(self.status_label)
        self.progress_bar=ProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
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
        target = Path(path).with_name(Path(path).stem+'-中文配音.mp4')
        if target.exists():
            number=2
            while target.exists():
                target=Path(path).with_name(Path(path).stem+f'-中文配音-{number}.mp4'); number+=1
        self.output_input.setText(str(target))

    def choose_subtitle(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择中文字幕', '', 'SRT 字幕 (*.srt)')
        if path: self.subtitle_input.setText(path)

    def choose_output(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存配音视频', self.output_input.text(), 'MP4 视频 (*.mp4)')
        if path: self.output_input.setText(path if path.lower().endswith('.mp4') else path+'.mp4')

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
        self.dubbing_settings = dict(value)
        self.model_id = value['model_id']
        self.model_button.setText(MODEL_OPTIONS[self.model_id]['name'])
        if self.model_id in MODELS:
            self.runtime_config = self.paths_for_model(self.model_id)
        if self.settings_widget.value() != value:
            self.settings_widget.blockSignals(True)
            self.settings_widget.set_value(value)
            self.settings_widget.blockSignals(False)
        self.update_ready()

    def backend_for_model(self, model_id):
        return self.backend_choices.get(model_id, 'pytorch')

    def set_backend(self, model_id, backend):
        if self.job is not None or backend not in ('pytorch', 'audio.cpp'):
            return
        self.backend_choices[model_id] = backend
        self.setup_finished.emit()

    def environment_key(self, model_id):
        paths = self.paths_for_model(model_id)
        if self.backend_for_model(model_id) == 'audio.cpp':
            return ('audio.cpp',)
        return ('pytorch', MODELS[model_id]['engine'], paths['python_path'], paths['repo_path'])

    def environment_state(self, model_id):
        paths = self.paths_for_model(model_id)
        backend = self.backend_for_model(model_id)
        if backend == 'audio.cpp':
            return dict(installed=False, ready=False, label='未接入',
                        message='暂不可用')
        else:
            installed = Path(paths['python_path']).is_file() or Path(paths['repo_path']).exists()
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
        return (Path(paths['model_dir'])/'videocaptioner-models.json').is_file()

    def model_download_size(self, model_id):
        return MODELS[model_id].get('download_size', '待获取')

    def edit_runtime(self):
        if self.model_id not in MODELS:
            self.status_label.setText("暂不可用")
            return
        from app.components.dubbing_model_dialog import DubbingEnvironmentDialog
        self.environment_dialog = DubbingEnvironmentDialog(self.window(), self, self.model_id)
        self.environment_dialog.show()

    def paths_for_model(self,model_id):
        from app.core.tts_runtime import default_paths
        paths=default_paths(model_id)
        if model_id=='indextts2':paths.update(self.load_runtime_config())
        else:
            file=APPDATA_PATH/('dubbing-runtime-'+model_id+'.json')
            try:
                if file.exists():
                    saved=json.loads(file.read_text('utf8'))
                    paths.update({key: str(saved[key]) for key in paths if saved.get(key)})
            except (ValueError,OSError):pass
        return paths

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

    def start_setup(self, operation, model_id=None):
        target=model_id or self.model_id
        if target not in MODELS:
            self.status_label.setText("暂不可用")
            return
        if self.backend_for_model(target) == 'audio.cpp':
            self.status_label.setText('暂不可用')
            return
        if self.job is None: self.launch_job(operation, dict(self.paths_for_model(target),model_id=target))

    def start_dubbing(self):
        if self.job is not None: return
        if self.backend_for_model(self.model_id) == 'audio.cpp':
            self.status_label.setText('暂不可用')
            return
        from app.core.dubbing.settings import executable, validation_error
        settings = self.current_settings()
        if not executable(settings):
            self.status_label.setText('暂不可用')
            return
        from app.core.dubbing import parse_srt
        try:
            error = validation_error(settings)
            if error: raise ValueError(error)
            video = Path(self.video_input.text()).resolve()
            subtitles = Path(self.subtitle_input.text()).resolve()
            output = Path(self.output_input.text()).resolve()
            if output in (video, subtitles) or output.exists():
                raise ValueError('输出文件已存在或与输入重名，请选择新的文件名，避免覆盖。')
            parse_srt(subtitles.read_text('utf-8-sig'))
            reference = settings['reference_audio'].strip() if settings['source'] == '参考音频' else None
            if settings['source'] == '参考音频' and not reference:
                raise ValueError('请选择参考音频，或启用自动提取。')
            config=dict(video_path=str(video),srt_path=str(subtitles),output_path=str(output),
                cache_dir=str(CACHE_PATH/'dubbing'),reference_audio=reference,reference_text=settings['reference_text'] or None,
                max_speedup=settings['max_speedup'],model_id=self.model_id,**self.runtime_config)
            self.launch_job('dub', config)
        except Exception as error:
            self.on_failure(str(error))

    def launch_job(self, operation, config):
        self.details.clear();self.details.hide();self.progress_bar.setValue(0);self.progress_bar.show()
        self.job=DubbingJob(operation,config,self)
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
        self.update_ready();self.job.start()

    def on_progress(self, value, message):
        self.progress_bar.setValue(max(0,min(value,100)))
        self.status_label.setText(message)

    def on_success(self, result):
        self.progress_bar.setValue(100)
        if self.job and self.job.operation=='dub':
            self.last_output=result;self.open_button.setEnabled(True)
            self.status_label.setText('配音完成，原视频和字幕保持不变')
            self.finished.emit(result)
        else:
            if self.job and self.job.operation in ('runtime', 'prepare', 'check_runtime'):
                key = self.environment_key(self.job.config['model_id'])
                self.environment_checks[key] = (True, 'GPU 运行环境可用')
            self.status_label.setText(result);self.runtime_status.setText(result)

    def on_failure(self, message):
        canceled=self.job is not None and self.job.cancel_event.is_set()
        if self.job and self.job.operation in ('runtime', 'prepare', 'check_runtime') and not canceled:
            ready = self.job.runtime_ready
            self.environment_checks[self.environment_key(self.job.config['model_id'])] = (ready, 'GPU 运行环境可用' if ready else message)
        self.status_label.setText('已取消，完成的句子已缓存，可再次开始继续' if canceled else '任务未完成，请查看原因')
        self.details.setPlainText(message);self.details.show()

    def job_finished(self):
        job=self.job;self.job=None
        if job:job.deleteLater()
        self.cancel_button.hide()
        self.download_button.setEnabled(True);self.runtime_button.setEnabled(True)
        self.command_bar.setEnabled(True);self.input_card.setEnabled(True)
        self.settings_widget.setEnabled(True)
        for field in (self.video_input,self.subtitle_input,self.output_input):field.setEnabled(True)
        self.update_ready()
        self.setup_finished.emit()

    def cancel_job(self):
        if self.job:
            self.job.cancel_event.set();self.cancel_button.setEnabled(False)
            self.status_label.setText('正在取消，保留已完成的句子…')

    def open_output(self):
        if self.last_output: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_output).parent)))

    def dragEnterEvent(self, event):
        if self.job is None and event.mimeData().hasUrls():event.acceptProposedAction()

    def dropEvent(self, event):
        if self.job is not None:return
        for url in event.mimeData().urls():
            path=url.toLocalFile()
            if path.lower().endswith('.srt'):self.subtitle_input.setText(path)
            elif Path(path).suffix.lower() in ('.mp4','.mkv','.mov','.avi','.webm','.m4v'):self.set_video(path)
        event.acceptProposedAction()

    def closeEvent(self,event):
        if self.job is not None:
            self.cancel_job();event.ignore()
        else:super().closeEvent(event)
