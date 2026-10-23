"""Dubbing settings reuse the transcription SettingCardGroup/scroll layout."""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from qfluentwidgets import (SettingCard, SettingCardGroup, SingleDirectionScrollArea,
    ComboBox, LineEdit, PushButton, DoubleSpinBox, SpinBox, BodyLabel,
    MessageBoxBase, SubtitleLabel, FluentIcon as FIF)
from app.core.dubbing.settings import MODEL_OPTIONS, normalized, validation_error


class DubbingSettingsWidget(QWidget):
    changed = pyqtSignal(dict)

    def __init__(self, value=None, parent=None, *, inline=False):
        super().__init__(parent)
        self._updating = False
        self.controls, self.cards = {}, {}
        self.voice_choices = {}
        self.previous_model = None
        self.reference_source = "原视频声音"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if inline:
            self.scrollArea = None
            self.containerLayout = layout
        else:
            self.scrollArea = SingleDirectionScrollArea(orient=Qt.Vertical, parent=self)
            self.scrollArea.setWidgetResizable(True)
            self.scrollArea.setStyleSheet('QScrollArea{background: transparent; border: none}')
            container = QWidget()
            container.setStyleSheet('QWidget{background: transparent}')
            self.containerLayout = QVBoxLayout(container)
            self.scrollArea.setWidget(container)
            layout.addWidget(self.scrollArea)
        group = self.group('声音')
        self.combo(group, 'model_id', '模型', [s['name'] for s in MODEL_OPTIONS.values()])
        self.combo(group, 'source', '声音来源', [])
        self.text(group, 'reference_audio', '参考音频')
        browse = PushButton('浏览', self)
        self.cards['reference_audio'].hBoxLayout.addWidget(browse)
        self.cards['reference_audio'].hBoxLayout.addSpacing(16)
        browse.clicked.connect(self.browse_reference)
        self.text(group, 'reference_text', '参考文本')
        self.combo(group, 'voice', '音色', [])
        self.text(group, 'description', '声音描述')
        self.combo(group, 'language', '输出语言', ['中文', '英语', '日语', '韩语'])
        self.number(group, 'speed', '语速', .5, 2., .05)
        self.text(group, 'style', '风格指令')
        timing = self.group('时间与输出')
        self.combo(timing, 'alignment', '时间对齐', ['按字幕时间对齐', '自然语速'])
        self.number(timing, 'max_speedup', '最大加速倍数', 1., 2., .05)
        self.combo(timing, 'overflow', '时长超限', ['停止并提示', '保留超限音频'])
        self.combo(timing, 'output', '输出', ['配音视频', '配音音频'])
        self.combo(timing, 'original', '原声', ['替换原声', '保留为独立音轨'])
        self.advanced_button = PushButton('高级设置', self)
        self.advanced_button.setCheckable(True)
        self.containerLayout.addWidget(self.advanced_button)
        self.advanced = self.group('生成参数')
        seed = SpinBox(self)
        seed.setRange(-1, 2147483647)
        seed.setSpecialValueText('随机')
        self.add_control(self.advanced, 'seed', '随机种子', seed)
        self.advanced.hide()
        self.advanced_button.toggled.connect(self.advanced.setVisible)
        self.error = BodyLabel('', self)
        self.containerLayout.addWidget(self.error)
        self.containerLayout.addStretch()
        self.controls['model_id'].currentIndexChanged.connect(self.model_changed)
        self.controls['source'].currentTextChanged.connect(self.source_changed)
        self.set_value(value or {})
        for key, control in self.controls.items():
            signal = (control.currentTextChanged if isinstance(control, ComboBox)
                      else control.valueChanged if isinstance(control, (SpinBox, DoubleSpinBox))
                      else control.textChanged)
            signal.connect(self.notify_changed)

    def notify_changed(self):
        if not self._updating:
            self.changed.emit(self.value())

    def group(self, title):
        group = SettingCardGroup(title, self)
        self.containerLayout.addWidget(group)
        return group

    def add_control(self, group, key, title, control):
        icons = {
            'model_id': FIF.ROBOT, 'source': FIF.MICROPHONE,
            'reference_audio': FIF.MUSIC, 'reference_text': FIF.DOCUMENT,
            'voice': FIF.PEOPLE, 'description': FIF.CHAT,
            'language': FIF.LANGUAGE, 'speed': FIF.SPEED_MEDIUM,
            'style': FIF.BRUSH, 'alignment': FIF.ALIGNMENT,
            'max_speedup': FIF.SPEED_HIGH, 'overflow': FIF.STOP_WATCH,
            'output': FIF.VIDEO, 'original': FIF.VOLUME, 'seed': FIF.UNIT,
        }
        card = SettingCard(icons[key], title, '', parent=group)
        control.setFixedWidth(260)
        card.hBoxLayout.addWidget(control, 0, Qt.AlignRight)
        card.hBoxLayout.addSpacing(16)
        group.addSettingCard(card)
        self.controls[key], self.cards[key] = control, card
        return control

    def combo(self, group, key, title, options):
        widget = ComboBox(self)
        widget.addItems(options)
        return self.add_control(group, key, title, widget)

    def text(self, group, key, title):
        return self.add_control(group, key, title, LineEdit(self))

    def number(self, group, key, title, minimum, maximum, step):
        widget = DoubleSpinBox(self)
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setDecimals(2)
        return self.add_control(group, key, title, widget)

    def model_id(self):
        return list(MODEL_OPTIONS)[self.controls['model_id'].currentIndex()]

    def model_changed(self):
        was_updating = self._updating
        self._updating = True
        spec = MODEL_OPTIONS[self.model_id()]
        if self.previous_model is not None:
            self.voice_choices[self.previous_model] = self.controls['voice'].currentText()
        self.previous_model = self.model_id()
        source = self.controls['source']
        old = source.currentText()
        source.blockSignals(True)
        source.clear(); source.addItems(list(spec['sources']))
        if old in spec['sources']: source.setCurrentText(old)
        elif self.reference_source in spec['sources']: source.setCurrentText(self.reference_source)
        source.blockSignals(False)
        voice = self.controls['voice']
        voice.clear(); voice.addItems(list(spec.get('voices', ())))
        previous_voice = self.voice_choices.get(self.model_id())
        if previous_voice in spec.get('voices', ()):
            voice.setCurrentText(previous_voice)
        # Keep existing style state for round-trips, without exposing this optional control.
        self.cards['style'].hide()
        self.advanced_button.setVisible(bool(spec.get('seed')))
        self.advanced.setVisible(bool(spec.get('seed')) and self.advanced_button.isChecked())
        self.source_changed()
        self._updating = was_updating

    def source_changed(self):
        source = self.controls['source'].currentText()
        if source in ('原视频声音', '参考音频'):
            self.reference_source = source
        for key in ('reference_audio', 'reference_text'):
            self.cards[key].setVisible(source == '参考音频')
        self.cards['voice'].setVisible(source == '预设音色')
        self.cards['description'].setVisible(source == '声音描述')

    def value(self):
        result = {}
        for key, control in self.controls.items():
            result[key] = (control.currentText() if isinstance(control, ComboBox)
                else control.value() if isinstance(control, (SpinBox, DoubleSpinBox)) else control.text())
        result['model_id'] = self.model_id()
        result['reference_source'] = self.reference_source
        spec = MODEL_OPTIONS[result['model_id']]
        return normalized(result)

    def set_value(self, value):
        was_updating = self._updating
        self._updating = True
        value = normalized(value)
        self.reference_source = value['reference_source']
        self.controls['model_id'].setCurrentIndex(list(MODEL_OPTIONS).index(value['model_id']))
        self.model_changed()
        for key, content in value.items():
            if key in ('model_id', 'reference_source'): continue
            widget = self.controls[key]
            if isinstance(widget, ComboBox): widget.setCurrentText(content)
            elif isinstance(widget, (SpinBox, DoubleSpinBox)): widget.setValue(content)
            else: widget.setText(content)
        self.source_changed()
        self._updating = was_updating
        self.notify_changed()

    def browse_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择参考音频', '', '音频 (*.wav *.mp3 *.flac *.m4a)')
        if path: self.controls['reference_audio'].setText(path)



class DubbingSettingsDialog(MessageBoxBase):
    def __init__(self, value=None, parent=None):
        super().__init__(parent)
        self.widget.setMinimumWidth(720)
        self.viewLayout.addWidget(SubtitleLabel('配音设置', self))
        self.settings = DubbingSettingsWidget(value, self)
        self.settings.setMinimumHeight(420)
        self.viewLayout.addWidget(self.settings)
        self.yesButton.setText('确定')
        self.cancelButton.setText('取消')

    def validate(self):
        error = validation_error(self.settings.value())
        self.settings.error.setText(error)
        return not error
