"""Dubbing model/environment management using the transcription dialog widgets."""
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QTableWidgetItem, QWidget, QHeaderView
from qfluentwidgets import (BodyLabel, ComboBox, FluentIcon, HyperlinkButton,
                            MessageBoxBase, ProgressBar, PushButton, SubtitleLabel)
from app.components.WhisperCppSettingWidget import WhisperCppDownloadDialog
from app.core.tts_catalog import MODELS


class DubbingEnvironmentDialog(MessageBoxBase):
    """The same status / backend / download layout as FasterWhisper's program section."""
    def __init__(self, parent, page, model_id):
        super().__init__(parent)
        self.page, self.model_id = page, model_id
        self.widget.setMinimumWidth(600)
        layout = QVBoxLayout()
        layout.addWidget(SubtitleLabel(MODELS[model_id]['name'] + ' · 运行环境', self))
        layout.addSpacing(8)
        self.status_label = BodyLabel('', self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        controls = QHBoxLayout()
        self.program_combo = ComboBox(self)
        self.program_combo.setFixedWidth(300)
        self.program_combo.addItem('Edge TTS（在线）' if model_id == 'edge-tts' else '官方 PyTorch（自动适配 GPU）', userData='pytorch')
        from app.core.audio_cpp_runtime import AUDIO_CPP_MODELS
        if model_id in AUDIO_CPP_MODELS:
            self.program_combo.addItem('audio.cpp', userData='audio.cpp')
        self.program_combo.setCurrentIndex(self.program_combo.findData(page.backend_for_model(model_id)))
        self.program_combo.currentIndexChanged.connect(self.backend_changed)
        self.program_combo.setToolTip('选择推理后端')
        self.program_download_btn = PushButton('下载环境', self)
        self.program_download_btn.clicked.connect(self.start_operation)
        controls.addWidget(self.program_combo)
        controls.addWidget(self.program_download_btn)
        controls.addStretch()
        layout.addLayout(controls)
        self.progress_bar = ProgressBar(self)
        self.progress_label = BodyLabel('', self)
        self.progress_label.setWordWrap(True)
        self.progress_bar.hide()
        self.progress_label.hide()
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        self.viewLayout.addLayout(layout)
        self.yesButton.hide()
        self.cancelButton.setText('关闭')
        self.page.setup_finished.connect(self.refresh_status)
        self.refresh_status()

    def backend_changed(self):
        self.page.set_backend(self.model_id, self.program_combo.currentData())
        self.refresh_status()

    def refresh_status(self):
        state = self.page.environment_state(self.model_id)
        self.status_label.setText(state['message'])
        self.program_download_btn.setText('检查环境' if state['installed'] else '下载环境')
        self.program_download_btn.setEnabled(self.page.job is None)
        self.program_combo.setEnabled(self.page.job is None)

    def start_operation(self):
        if self.page.job is not None:
            return
        state = self.page.environment_state(self.model_id)
        self.page.start_setup('check_runtime' if state['installed'] else 'runtime', self.model_id)
        job = self.page.job
        if job is None:
            return
        self.progress_bar.show()
        self.progress_label.show()
        self.progress_label.setText('正在检查环境…' if state['installed'] else '正在准备运行环境…')
        self.program_download_btn.setEnabled(False)
        self.program_combo.setEnabled(False)
        job.progress.connect(self.update_progress)
        job.succeeded.connect(self.progress_label.setText)
        job.failed.connect(self.progress_label.setText)

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def reject(self):
        if self.page.job and self.page.job.operation in ('runtime', 'check_runtime'):
            self.page.cancel_job()
        super().reject()


class DubbingModelDialog(WhisperCppDownloadDialog):
    def __init__(self, parent, page):
        self.page = page
        super().__init__(parent, setting_widget=page)
        self.widget.setMinimumWidth(740)
        self.page.setup_finished.connect(self.operation_finished)

    def _setup_ui(self):
        layout = QVBoxLayout()
        self._setup_model_section(layout)
        self._setup_progress_section(layout)
        self.progress_label.setWordWrap(True)
        self.viewLayout.addLayout(layout)
        self.cancelButton.setText('关闭')
        self.yesButton.hide()

    def _create_model_table(self):
        table = super()._create_model_table()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(['模型名称', '大小', '状态', '操作', '环境'])
        # Preserve the original readable size/status columns and row/delegate styling.
        # The added environment column needs extra width, not wrapped cells.
        for column, width in ((1, 100), (2, 80), (3, 90), (4, 140)):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Fixed)
            table.setColumnWidth(column, width)
        table.setWordWrap(False)
        return table

    def _populate_model_table(self):
        self.model_table.setRowCount(len(MODELS))
        for row, (model_id, model) in enumerate(MODELS.items()):
            downloaded = self.page.model_downloaded(model_id)
            for column, text in enumerate((model['name'], self.page.model_download_size(model_id), '在线' if model_id == 'edge-tts' else '已下载' if downloaded else '未下载')):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.model_table.setItem(row, column, item)
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(4, 4, 4, 4)
            button = HyperlinkButton('', '检查' if downloaded else '下载', self)
            button.setIcon(FluentIcon.DOWNLOAD)
            button.clicked.connect(lambda checked=False, key=model_id: self.start_operation('models', key))
            layout.addStretch()
            layout.addWidget(button)
            layout.addStretch()
            self.model_table.setCellWidget(row, 3, container)
            environment = QWidget()
            env_layout = QHBoxLayout(environment)
            env_layout.setContentsMargins(4, 4, 4, 4)
            env_layout.setSpacing(0)
            state = self.page.environment_state(model_id)
            label = ('audio.cpp' if self.page.backend_for_model(model_id) == 'audio.cpp' else
                     '安装环境' if not state['installed'] else
                     '检查环境' if not state['ready'] else
                     '在线' if model_id == 'edge-tts' else 'PyTorch')
            env_layout.addStretch()
            settings = HyperlinkButton('', label, environment)
            settings.setToolTip(state['message'])
            settings.clicked.connect(lambda checked=False, key=model_id: self.open_environment(key))
            env_layout.addWidget(settings)
            env_layout.addStretch()
            self.model_table.setCellWidget(row, 4, environment)
        self._set_all_download_buttons_enabled(self.page.job is None)

    def open_environment(self, model_id):
        if self.page.job is not None:
            return
        self.environment_dialog = DubbingEnvironmentDialog(self.window(), self.page, model_id)
        self.environment_dialog.show()

    def start_operation(self, operation, model_id=None):
        if self.page.job is not None:
            return
        self.page.start_setup('prepare', model_id=model_id)
        if self.page.job is None:
            return
        self.progress_bar.show()
        self.progress_label.show()
        self.progress_label.setText('正在准备…')
        self._set_all_download_buttons_enabled(False)
        self.page.job.progress.connect(self.update_progress)
        self.page.job.failed.connect(self.progress_label.setText)
        self.page.job.succeeded.connect(self.progress_label.setText)

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def _set_all_download_buttons_enabled(self, enabled):
        for row in range(self.model_table.rowCount()):
            for column in (3, 4):
                container = self.model_table.cellWidget(row, column)
                if container:
                    for button in container.findChildren(HyperlinkButton):
                        button.setEnabled(enabled)

    def operation_finished(self):
        self._populate_model_table()

    def reject(self):
        if self.page.job and self.page.job.operation != 'dub':
            self.page.cancel_job()
        super(WhisperCppDownloadDialog, self).reject()
