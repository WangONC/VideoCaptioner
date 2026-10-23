import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtTest import QTest
from app.core.dubbing.settings import MODEL_OPTIONS, normalized, executable, validation_error, PresetStore, shared_settings
from app.components.dubbing_settings import DubbingSettingsDialog
from app.components.dubbing_batch_dialog import DubbingAssociationDialog
from app.view.dubbing_interface import DubbingInterface
from app.view.batch_process_interface import BatchProcessInterface
from app.core.entities import BatchTaskType, BatchTaskStatus
from app.thread.batch_process_thread import BatchProcessThread, BatchTask
APP = QApplication.instance() or QApplication([])

class SettingsTests(unittest.TestCase):
    def setUp(self):
        shared_settings.update({})
        self.host = QWidget(); self.host.resize(1050, 800)
    def tearDown(self):
        QTest.qWait(250)
        self.host.close(); self.host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        APP.processEvents()
    def test_capability_controls_and_roundtrip(self):
        dialog = DubbingSettingsDialog(parent=self.host)
        widget = dialog.settings
        for model, spec in MODEL_OPTIONS.items():
            value = normalized(dict(model_id=model))
            widget.set_value(value)
            self.assertEqual(widget.value(), value)
            self.assertEqual(widget.controls['source'].count(), len(spec['sources']))
            self.assertEqual(widget.cards['voice'].isHidden(), value['source'] != '预设音色')
            self.assertEqual(widget.cards['description'].isHidden(), value['source'] != '声音描述')
            self.assertEqual(widget.advanced_button.isHidden(), not bool(spec.get('seed')))
        self.assertNotIn('dots.tts.edit', str(MODEL_OPTIONS))
    def test_validation(self):
        self.assertTrue(validation_error(normalized(dict(source='参考音频'))))
        self.assertTrue(validation_error(normalized(dict(model_id='qwen3-design'))))
        self.assertFalse(validation_error(normalized(dict(model_id='edge-tts'))))
    def test_presets_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            store = PresetStore(Path(root)/'presets.json')
            value = normalized(dict(model_id='dots-mf', reference_text='课程讲解', seed=42))
            store.save('讲师', value)
            self.assertEqual(store.read()['讲师'], value)
            with self.assertRaises(ValueError): store.save(' ', value)
    def test_execution_allowlist(self):
        self.assertTrue(executable(normalized({})))
        for key, value in dict(model_id='edge-tts', speed=1.1, language='英语', style='轻松', alignment='自然语速', output='配音音频', original='保留为独立音轨', overflow='保留超限音频').items():
            self.assertFalse(executable(normalized({key:value})), key)
        page = DubbingInterface(self.host)
        page.dubbing_settings['speed'] = 1.1
        with patch.object(page, 'launch_job') as launch:
            page.start_dubbing(); launch.assert_not_called()
    def test_batch_dubbing_does_not_enqueue(self):
        page = BatchProcessInterface(self.host)
        page.task_type_combo.setCurrentText(str(BatchTaskType.DUBBING))
        page.add_task_to_table('/tmp/course.mp4')
        self.assertFalse(page.start_all_btn.isEnabled())
        self.assertIn('未选择字幕', page.task_table.item(0,0).text())
        with patch.object(page.batch_thread, 'add_task') as add:
            page.start_all_tasks(); page.start_task('/tmp/course.mp4'); add.assert_not_called()
        page.task_type_combo.setCurrentText(str(BatchTaskType.DUBBING_SYNTHESIS))
        self.assertFalse(page.start_all_btn.isEnabled())
        with patch.object(page.batch_thread, 'add_task') as add:
            page.start_all_tasks(); page.start_task('/tmp/course.mp4'); add.assert_not_called()
        page.task_type_combo.setCurrentText(str(BatchTaskType.FULL_PROCESS))
        self.assertTrue(page.start_all_btn.isEnabled())
        self.assertFalse(page.dubbing_requested())
        page.task_type_combo.setCurrentText(str(BatchTaskType.TRANSCRIBE))
        self.assertFalse(page.dubbing_requested())
        page.close()
    def test_worker_rejects_dubbing(self):
        worker = BatchProcessThread()
        for kind in (BatchTaskType.DUBBING, BatchTaskType.DUBBING_SYNTHESIS):
            task = BatchTask('/tmp/course.mp4', kind)
            worker._process_task(task)
            self.assertEqual(task.status, BatchTaskStatus.FAILED)
            self.assertEqual(task.error_message, '暂不可用')
    def test_association_validation(self):
        dialog = DubbingAssociationDialog('/tmp/course.mp4', {}, self.host)
        self.assertFalse(dialog.validate())
        dialog.generated_subtitles=True
        self.assertTrue(dialog.validate())
        dialog.reference.setText('/missing/reference.wav')
        self.assertFalse(dialog.validate())

    def test_switch_back_preserves_common_and_hidden_values(self):
        widget = DubbingSettingsDialog(parent=self.host).settings
        original = normalized(dict(model_id='dots-soar', source='参考音频', reference_audio='/tmp/narrator.wav', reference_text='讲师声音', speed=1.2, language='英语', max_speedup=1.5, seed=123, style='平静'))
        widget.set_value(original)
        widget.controls['model_id'].setCurrentIndex(list(MODEL_OPTIONS).index('edge-tts'))
        hidden = widget.value()
        for key in ('reference_audio', 'reference_text', 'speed', 'language', 'max_speedup', 'seed', 'style'):
            self.assertEqual(hidden[key], original[key])
        widget.controls['model_id'].setCurrentIndex(list(MODEL_OPTIONS).index('dots-soar'))
        restored = widget.value()
        self.assertEqual(restored['source'], '参考音频')
        for key in ('reference_audio', 'reference_text', 'speed', 'language', 'max_speedup', 'seed', 'style'):
            self.assertEqual(restored[key], original[key])
        with tempfile.TemporaryDirectory() as root:
            store = PresetStore(Path(root)/'presets.json')
            store.save('shared', hidden)
            self.assertEqual(store.read()['shared']['seed'], 123)
            self.assertEqual(store.read()['shared']['style'], '平静')

    def test_single_and_batch_share_same_profile(self):
        single = DubbingInterface(self.host)
        batch = BatchProcessInterface(self.host)
        value = normalized(dict(model_id='dots-mf', reference_text='共同讲师', speed=1.2, seed=5))
        shared_settings.update(value)
        self.assertEqual(single.dubbing_settings, batch.dubbing_settings)
        single.select_model('qwen3-1.7b')
        self.assertEqual(batch.dubbing_settings['model_id'], 'qwen3-1.7b')
        self.assertEqual(batch.dubbing_settings['reference_text'], '共同讲师')
        self.assertEqual(batch.dubbing_settings['speed'], 1.2)
        self.assertEqual(batch.dubbing_settings['seed'], 5)
        batch.close()

    def test_inline_edits_apply_immediately_without_duplicate_reference_controls(self):
        single = DubbingInterface(self.host)
        batch = BatchProcessInterface(self.host)
        self.assertIsNone(single.settings_widget.scrollArea)
        self.assertIs(single.scroll_area.widget(), single.scroll_content)
        self.assertFalse(hasattr(single, 'auto_reference'))
        self.assertTrue(single.settings_widget.cards['model_id'].isHidden())
        single.settings_widget.controls['reference_text'].setText('实时同步')
        single.settings_widget.controls['speed'].setValue(1.15)
        self.assertEqual(batch.dubbing_settings['reference_text'], '实时同步')
        self.assertEqual(shared_settings.value()['speed'], 1.15)
        self.assertEqual(single.current_settings(), batch.dubbing_settings)
        single.settings_widget.controls['source'].setCurrentText('参考音频')
        single.settings_widget.controls['reference_audio'].setText('/tmp/voice.wav')
        single.select_model('edge-tts')
        single.select_model('dots-soar')
        self.assertEqual(single.current_settings()['source'], '参考音频')
        self.assertEqual(single.current_settings()['reference_audio'], '/tmp/voice.wav')
        self.assertEqual(single.current_settings()['reference_text'], '实时同步')
        self.assertEqual(single.current_settings()['speed'], 1.15)
        batch.close()

    def test_inline_loading_is_atomic_and_updates_shared_state(self):
        single = DubbingInterface(self.host)
        value = normalized(dict(model_id='gpt-sovits', source='参考音频', reference_audio='/tmp/voice.wav', reference_text='预设', speed=1.25, seed=71))
        events = []
        single.settings_widget.changed.connect(events.append)
        single.settings_widget.set_value(value)
        self.assertEqual(events, [value])
        self.assertEqual(shared_settings.value(), value)
        self.assertEqual(single.current_settings(), value)

    def test_batch_toolbar_and_task_order(self):
        batch = BatchProcessInterface(self.host)
        self.assertFalse(hasattr(batch, 'include_dubbing'))
        self.assertFalse(hasattr(batch, 'dubbing_settings_btn'))
        self.assertEqual([batch.task_type_combo.itemText(i) for i in range(batch.task_type_combo.count())],
            ['批量转录', '批量字幕', '批量配音', '批量合成', '转录+字幕', '转录+断句', '配音+合成', '全流程处理'])
        self.assertEqual(batch.layout().itemAt(0).layout().count(), 5)
        batch.close()

    def test_simplified_settings_cards(self):
        widget = DubbingSettingsDialog(parent=self.host).settings
        self.assertNotIn('preset', widget.controls)
        self.assertNotIn('preset_name', widget.controls)
        self.assertFalse(hasattr(widget, 'store'))
        for model in MODEL_OPTIONS:
            widget.set_value(dict(model_id=model, style='保留的旧值'))
            self.assertTrue(widget.cards['style'].isHidden())
            self.assertEqual(widget.value()['style'], '保留的旧值')
        icons = [widget.cards[key].iconLabel._icon for key in ('source', 'language', 'speed', 'alignment', 'output')]
        self.assertEqual(len(set(icons)), len(icons))
