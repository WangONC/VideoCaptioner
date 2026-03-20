# app/core/storage/database.py
import os
import logging
from contextlib import contextmanager
import threading
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from .models import Base
from .constants import CACHE_CONFIG

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理类，负责数据库连接和会话管理"""

    _registry_lock = threading.Lock()
    _engines = {}
    _session_makers = {}
    _write_locks = {}

    def __init__(self, app_data_path: str):
        self.db_path = os.path.join(app_data_path, CACHE_CONFIG["db_filename"])
        self.db_url = f"sqlite:///{self.db_path}"
        self._engine = None
        self._session_maker = None
        self._write_lock = None
        self.init_db()

    def init_db(self):
        """初始化数据库连接和表结构"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            with self.__class__._registry_lock:
                if self.db_path in self.__class__._engines:
                    self._engine = self.__class__._engines[self.db_path]
                    self._session_maker = self.__class__._session_makers[self.db_path]
                    self._write_lock = self.__class__._write_locks[self.db_path]
                    return

                self._engine = create_engine(
                    self.db_url,
                    connect_args={"check_same_thread": False, "timeout": 30},
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                    pool_recycle=3600,
                )

                @event.listens_for(self._engine, "connect")
                def _set_sqlite_pragma(dbapi_connection, _connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute("PRAGMA busy_timeout=30000")
                    cursor.close()

                Base.metadata.create_all(self._engine)
                self._session_maker = sessionmaker(
                    bind=self._engine, expire_on_commit=False
                )
                self._write_lock = threading.RLock()

                self.__class__._engines[self.db_path] = self._engine
                self.__class__._session_makers[self.db_path] = self._session_maker
                self.__class__._write_locks[self.db_path] = self._write_lock
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_maker = None

    @contextmanager
    def get_session(self, write: bool = False):
        """获取数据库会话的上下文管理器"""
        if not self._engine or not self._session_maker:
            self.init_db()

        lock = self._write_lock if write else None
        if lock:
            lock.acquire()

        session = self._session_maker()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()
            if lock:
                lock.release()
