"""Button-to-worker regression tests, using real Qt signals without downloads."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication, QWidget
from qfluentwidgets import HyperlinkButton
from app.view.dubbing_interface import DubbingInterface, DubbingJob
from app.components.dubbing_model_dialog import DubbingModelDialog
from app.core.tts_catalog import MODELS

APP = QApplication.instance() or QApplication([])

class SetupButtonsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.host = QWidget()
        self.page = DubbingInterface(self.host)
        self.paths = patch.object(self.page, 'paths_for_model', side_effect=lambda key: dict(
            model_dir=str(Path(self.tmp.name)/'models'/key),
            python_path=str(Path(self.tmp.name)/'runtime'/'python.exe'), repo_path=str(Path(self.tmp.name)/'source')))
        self.paths.start()
    def tearDown(self):
        self.page.job = None
        self.host.deleteLater()
        APP.processEvents()
        self.paths.stop()
        self.tmp.cleanup()
    def test_actual_download_buttons_never_prepare_environment(self):
        with patch.object(self.page, 'model_downloaded', return_value=False):
            dialog = DubbingModelDialog(self.host, self.page)
        with patch.object(self.page, 'start_setup') as start:
            dialog.model_table.cellWidget(0, 3).findChild(HyperlinkButton).click()
            start.assert_called_once_with('models', model_id='indextts2')
    def test_actual_check_buttons_never_download(self):
        with patch.object(self.page, 'model_downloaded', return_value=True):
            dialog = DubbingModelDialog(self.host, self.page)
        with patch.object(self.page, 'start_setup') as start:
            dialog.model_table.cellWidget(0, 3).findChild(HyperlinkButton).click()
            start.assert_called_once_with('check_models', model_id='indextts2')
        with patch.object(self.page, 'start_setup') as start:
            row = list(MODELS).index('edge-tts')
            dialog.model_table.cellWidget(row, 3).findChild(HyperlinkButton).click()
            start.assert_called_once_with('check_runtime', model_id='edge-tts')
    def test_missing_model_check_is_local_failure_not_install(self):
        config = dict(self.page.paths_for_model('indextts2'), model_id='indextts2')
        job = DubbingJob('check_models', config)
        errors = []; job.failed.connect(errors.append)
        with patch('app.core.tts_runtime.prepare_runtime') as install, patch('app.core.tts_runtime.download_models') as download, patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
            job.run_setup()
        self.assertTrue(errors)
        install.assert_not_called(); download.assert_not_called()
    def test_folder_uses_selected_dubbing_model(self):
        dialog = DubbingModelDialog(self.host, self.page)
        row = list(MODELS).index('dots-mf')
        dialog.model_table.selectRow(row)
        with patch('app.components.dubbing_model_dialog.QDesktopServices.openUrl') as opened:
            dialog._open_model_folder()
        self.assertEqual(Path(opened.call_args.args[0].toLocalFile()), Path(self.tmp.name)/'models'/'dots-mf')
    def test_cancel_uses_toast_not_error_textedit(self):
        event = threading.Event(); event.set()
        self.page.job = SimpleNamespace(operation='runtime', cancel_event=event)
        with patch('app.view.dubbing_interface.InfoBar.info') as toast:
            self.page.on_failure('操作已取消')
        toast.assert_called_once()
        self.assertTrue(self.page.details.isHidden())
        self.assertNotIn('句子', self.page.status_label.text())
    def test_legacy_paths_migrate_but_custom_paths_do_not(self):
        import json
        from app.core.dubbing.execution import runtime_paths
        from app.core.tts_runtime import default_paths
        root = Path(self.tmp.name)
        legacy = root / 'tts' / 'indextts2'
        saved = dict(python_path=str(legacy/'runtime'/'python.exe'), repo_path=str(legacy/'source'))
        file = root/'dubbing-runtime.json'
        file.write_text(json.dumps(saved))
        with patch('app.config.APPDATA_PATH', root):
            result = runtime_paths('indextts2', 'pytorch')
            self.assertEqual(result['python_path'], default_paths('indextts2')['python_path'])
            saved['python_path'] = str(root/'custom'/'python.exe')
            file.write_text(json.dumps(saved))
            result = runtime_paths('indextts2', 'pytorch')
            self.assertEqual(result['python_path'], saved['python_path'])
            self.assertEqual(result['repo_path'], saved['repo_path'])

    def test_explicit_cpu_selection_changes_only_selected_model(self):
        from app.components.dubbing_model_dialog import DubbingEnvironmentDialog
        from app.core.dubbing.execution import device_choices
        try:
            dialog = DubbingEnvironmentDialog(self.host, self.page, 'qwen3-0.6b')
            dialog.device_combo.setCurrentIndex(dialog.device_combo.findData('cpu'))
            self.assertEqual(self.page.device_for_model('qwen3-0.6b'), 'cpu')
            with patch.object(self.page, 'launch_job') as launch:
                self.page.start_setup('runtime', 'qwen3-0.6b')
            self.assertEqual(launch.call_args.args[1]['device'], 'cpu')
            self.assertEqual(self.page.device_for_model('indextts2'), 'gpu')
        finally:
            device_choices.clear()

    def test_environment_selection_clears_previous_operation_status(self):
        from app.components.dubbing_model_dialog import DubbingEnvironmentDialog
        from app.core.dubbing.execution import device_choices, backend_choices
        try:
            dialog = DubbingEnvironmentDialog(self.host, self.page, 'qwen3-0.6b')
            dialog.program_combo.setCurrentIndex(dialog.program_combo.findData('audio.cpp'))
            dialog.progress_label.setText('正在准备运行环境…')
            dialog.progress_label.show()
            dialog.progress_bar.setValue(30)
            dialog.progress_bar.show()
            dialog.program_combo.setCurrentIndex(dialog.program_combo.findData('pytorch'))
            self.assertEqual(dialog.progress_label.text(), '')
            self.assertTrue(dialog.progress_label.isHidden())
            self.assertTrue(dialog.progress_bar.isHidden())
            self.assertEqual(dialog.progress_bar.value(), 0)
            dialog.progress_label.setText('先前 GPU 检查失败')
            dialog.progress_label.show()
            dialog.device_combo.setCurrentIndex(dialog.device_combo.findData('cpu'))
            self.assertEqual(dialog.progress_label.text(), '')
            self.assertTrue(dialog.progress_label.isHidden())
            self.assertTrue(dialog.program_download_btn.isEnabled())
        finally:
            device_choices.clear()
            backend_choices.clear()

    def test_progress_reuses_media_card_ring_and_concise_completion(self):
        from qfluentwidgets import ProgressRing
        self.assertIsInstance(self.page.progress_bar, ProgressRing)
        self.assertEqual(self.page.progress_bar.width(), 20)
        self.assertTrue(self.page.status_label.isHidden())
        self.page.job = SimpleNamespace(operation='dub')
        self.page.on_success(str(Path(self.tmp.name)/'result.mp4'))
        self.assertEqual(self.page.start_button.text(), '配音完成')
        self.assertEqual(self.page.status_label.text(), '配音完成')

    def test_long_model_button_fits_text_and_dropdown(self):
        from app.core.dubbing.settings import MODEL_OPTIONS
        value = self.page.current_settings()
        for key in MODEL_OPTIONS:
            value['model_id'] = key
            self.page.apply_shared_settings(value)
            button = self.page.model_button
            self.assertGreaterEqual(button.width(), button.fontMetrics().horizontalAdvance(button.text()) + 70)

    def test_setup_raw_english_goes_to_log_not_progress_copy(self):
        from app.components.dubbing_model_dialog import concise_status
        self.assertEqual(concise_status('Collecting torch==2.8.0'), '正在安装依赖')
        self.assertEqual(concise_status('下载 model.bin 10 / 20 MB'), '下载 model.bin 10 / 20 MB')

    def test_unknown_size_is_explicit(self):
        self.assertEqual(self.page.model_download_size('dots-mf'), '未知')

if __name__ == '__main__': unittest.main()
