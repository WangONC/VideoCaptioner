from pathlib import Path
from PyQt5.QtWidgets import QHBoxLayout, QFileDialog
from qfluentwidgets import MessageBoxBase, SubtitleLabel, BodyLabel, LineEdit, PushButton

class DubbingAssociationDialog(MessageBoxBase):
    def __init__(self, video, value, parent=None, generated_subtitles=False):
        super().__init__(parent)
        self.generated_subtitles = generated_subtitles
        self.widget.setMinimumWidth(640)
        self.viewLayout.addWidget(SubtitleLabel(Path(video).name, self))
        self.subtitle = self.file_row('字幕', value.get('subtitle', ''), 'SRT 字幕 (*.srt)')
        self.subtitle.setPlaceholderText('流程生成字幕' if generated_subtitles else '选择对应字幕')
        self.reference = self.file_row('参考声音', value.get('reference', ''), '音频 (*.wav *.mp3 *.flac *.m4a)')
        self.reference.setPlaceholderText('使用配音设置')
        self.reference_text = LineEdit(self)
        self.reference_text.setPlaceholderText('参考音频对应文本')
        self.reference_text.setText(value.get('reference_text', ''))
        self.viewLayout.addWidget(self.reference_text)
        self.error = BodyLabel('', self)
        self.viewLayout.addWidget(self.error)
        self.yesButton.setText('确定'); self.cancelButton.setText('取消')

    def file_row(self, title, value, file_filter):
        row = QHBoxLayout()
        row.addWidget(BodyLabel(title, self))
        field = LineEdit(self); field.setText(value)
        row.addWidget(field, 1)
        button = PushButton('浏览', self)
        def browse():
            path, _ = QFileDialog.getOpenFileName(self, title, '', file_filter)
            if path: field.setText(path)
        button.clicked.connect(browse)
        row.addWidget(button); self.viewLayout.addLayout(row)
        return field

    def value(self):
        return dict(subtitle=self.subtitle.text().strip(), reference=self.reference.text().strip(), reference_text=self.reference_text.text().strip())

    def validate(self):
        value = self.value()
        if not value['subtitle'] and not self.generated_subtitles:
            self.error.setText('请选择对应字幕'); return False
        for key in ('subtitle', 'reference'):
            if value[key] and not Path(value[key]).is_file():
                self.error.setText('文件不存在'); return False
        return True
