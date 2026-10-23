"""Headless Qt contract tests: no model downloads or GPU inference."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import threading
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtTest import QTest
from qfluentwidgets import LineEdit, HyperlinkButton
from app.view.dubbing_interface import DubbingInterface
from app.components.dubbing_model_dialog import DubbingModelDialog, DubbingEnvironmentDialog
from app.components.WhisperCppSettingWidget import WhisperCppDownloadDialog

APP = QApplication.instance() or QApplication([])


class DubbingUiTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.host = QWidget()
        self.host.resize(1050, 800)
        self.page = DubbingInterface(self.host)
        self.paths = patch.object(self.page, 'paths_for_model', side_effect=self.model_paths)
        self.paths.start()

    def tearDown(self):
        self.paths.stop()
        # Fluent modal close animates asynchronously. Finish it before deleting
        # the host so one test cannot leave a live animation in the next test.
        QTest.qWait(250)
        self.host.close()
        self.host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        APP.processEvents()
        self.root.cleanup()

    def model_paths(self, model_id):
        engine = 'qwen3' if model_id.startswith('qwen3') else 'indextts2'
        root = Path(self.root.name)
        return dict(python_path=str(root/engine/'python.exe'),
                    repo_path=str(root/engine/'source'), model_dir=str(root/model_id))

    def test_model_rows_reuse_transcription_table_and_have_environment(self):
        original = WhisperCppDownloadDialog(self.host)
        dialog = DubbingModelDialog(self.host, self.page)
        self.assertEqual(dialog.widget.minimumWidth(), 740)
        self.assertEqual(dialog.model_table.columnWidth(1), original.model_table.columnWidth(1))
        self.assertEqual(dialog.model_table.columnWidth(2), original.model_table.columnWidth(2))
        self.assertFalse(dialog.model_table.wordWrap())
        self.assertEqual(dialog.model_table.height(), original.model_table.height())
        self.assertEqual(dialog.model_table.verticalHeader().defaultSectionSize(), 45)
        self.assertEqual(dialog.model_table.columnCount(), 5)
        from app.core.tts_catalog import MODELS
        self.assertEqual(dialog.model_table.rowCount(), len(MODELS))
        for row in range(3):
            self.assertEqual(dialog.model_table.cellWidget(row, 4).findChild(HyperlinkButton).text(), '安装环境')
        self.assertFalse(any(b.text() == '高级设置' for b in dialog.findChildren(HyperlinkButton)))

    def test_environment_has_backend_and_download_not_path_form(self):
        dialog = DubbingEnvironmentDialog(self.host, self.page, 'qwen3-0.6b')
        self.assertEqual(dialog.program_combo.count(), 2)
        self.assertEqual(dialog.program_combo.currentData(), 'pytorch')
        self.assertEqual(dialog.program_combo.width(), 300)
        self.assertEqual(dialog.program_download_btn.text(), '下载环境')
        self.assertEqual(dialog.findChildren(LineEdit), [])
        with patch.object(self.page, 'start_setup') as start:
            dialog.program_download_btn.click()
            start.assert_called_once_with('runtime', 'qwen3-0.6b')
        self.assertEqual(self.page.model_id, 'indextts2')

    def test_audiocpp_download_requires_explicit_click(self):
        dialog = DubbingEnvironmentDialog(self.host, self.page, 'qwen3-0.6b')
        dialog.program_combo.setCurrentIndex(dialog.program_combo.findData('audio.cpp'))
        self.assertEqual(self.page.backend_for_model('qwen3-0.6b'), 'audio.cpp')
        self.assertTrue(dialog.program_download_btn.isEnabled())
        self.assertIn('尚未下载', dialog.status_label.text())
        self.assertEqual(dialog.program_combo.toolTip(), '选择推理后端')
        self.assertEqual(self.page.model_id, 'indextts2')
        with patch.object(self.page, 'launch_job') as launch:
            dialog.start_operation()
            self.page.start_setup('runtime', 'qwen3-0.6b')
            self.assertEqual(launch.call_count, 2)
            self.assertEqual(launch.call_args.args[1]['backend'], 'audio.cpp')
        manager = DubbingModelDialog(self.host, self.page)
        button = manager.model_table.cellWidget(1, 4).findChild(HyperlinkButton)
        self.assertEqual(button.text(), 'audio.cpp')
        self.assertIn('尚未下载', button.toolTip())
        self.assertEqual(len(manager.model_table.cellWidget(1, 4).findChildren(HyperlinkButton)), 1)
        dialog.program_combo.setCurrentIndex(dialog.program_combo.findData('pytorch'))
        self.assertTrue(dialog.program_download_btn.isEnabled())

    def test_existing_environment_gets_check_not_unconditional_ready(self):
        python = Path(self.model_paths('qwen3-0.6b')['python_path'])
        python.parent.mkdir(parents=True)
        python.touch()
        dialog = DubbingEnvironmentDialog(self.host, self.page, 'qwen3-0.6b')
        self.assertEqual(self.page.environment_state('qwen3-0.6b')['label'], '待检查')
        self.assertEqual(dialog.program_download_btn.text(), '检查环境')
        with patch.object(self.page, 'start_setup') as start:
            dialog.program_download_btn.click()
            start.assert_called_once_with('check_runtime', 'qwen3-0.6b')

    def test_shared_qwen_environment_status(self):
        key = self.page.environment_key('qwen3-0.6b')
        self.page.environment_checks[key] = (True, 'GPU 运行环境可用')
        self.assertEqual(key, self.page.environment_key('qwen3-1.7b'))
        self.assertEqual(self.page.environment_state('qwen3-1.7b')['label'], '可用')
        self.assertEqual(self.page.environment_state('indextts2')['label'], '未安装')

    def test_manager_download_does_not_switch_active_model(self):
        dialog = DubbingModelDialog(self.host, self.page)
        with patch.object(self.page, 'start_setup') as start:
            dialog.start_operation('models', 'qwen3-1.7b')
            start.assert_called_once_with('models', model_id='qwen3-1.7b')
        self.assertEqual(self.page.model_id, 'indextts2')

    def test_busy_environment_does_not_start_duplicate(self):
        dialog = DubbingEnvironmentDialog(self.host, self.page, 'indextts2')
        self.page.job = object()
        with patch.object(self.page, 'start_setup') as start:
            dialog.start_operation()
            start.assert_not_called()
        self.page.job = None

    def test_environment_close_cancels_only_environment_job(self):
        dialog = DubbingEnvironmentDialog(self.host, self.page, 'indextts2')
        self.page.job = SimpleNamespace(operation='check_runtime')
        with patch.object(self.page, 'cancel_job') as cancel:
            dialog.reject()
            cancel.assert_called_once()
        self.page.job = None

    def test_model_download_failure_does_not_mark_working_environment_broken(self):
        self.page.environment_checks[self.page.environment_key('qwen3-0.6b')] = (True, '运行环境可用')
        self.page.job = SimpleNamespace(operation='models', runtime_ready=False,
            cancel_event=threading.Event(), config={'model_id': 'qwen3-0.6b'})
        self.page.on_failure('模型下载网络错误')
        self.assertEqual(self.page.environment_state('qwen3-1.7b')['label'], '可用')
        self.page.job = None

    def test_failed_runtime_check_is_not_ready(self):
        self.page.job = SimpleNamespace(operation='check_runtime', runtime_ready=False,
            cancel_event=threading.Event(), config={'model_id': 'qwen3-0.6b'})
        self.page.on_failure('GPU 检查失败')
        self.assertEqual(self.page.environment_state('qwen3-1.7b')['label'], '异常')
        self.assertFalse(self.page.environment_state('qwen3-1.7b')['ready'])
        self.page.job = None

if __name__ == '__main__':
    unittest.main()
