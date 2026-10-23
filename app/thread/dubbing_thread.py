"""The same cancellable execution worker is used by the single and batch UI."""
import threading
from PyQt5.QtCore import QThread, pyqtSignal
from app.core.dubbing.execution import execute_request
from app.core.dubbing.pipeline import CancelledError


class DubbingThread(QThread):
    progress = pyqtSignal(int, str)
    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, request, parent=None):
        super().__init__(parent)
        self.request = request
        self.operation = 'dub'
        self.config = dict(request.config_items)
        self.cancel_event = threading.Event()
        self.output_path = None

    def run(self):
        try:
            result = execute_request(self.request,
                lambda done, total, message: self.progress.emit(int(100 * done / max(total, 1)), message),
                self.cancel_event)
            # Publication is the success boundary; a later cancel does not erase a completed result.
            self.output_path = result
            self.succeeded.emit(result)
        except CancelledError:
            self.cancelled.emit()
        except Exception as error:
            self.failed.emit(str(error))

    def stop(self):
        self.cancel_event.set()
