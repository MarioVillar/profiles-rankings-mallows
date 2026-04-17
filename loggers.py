import logging
import traceback
import sys
import os


def check_create_dir(path: str) -> None:
    """
    Check if the directory exists. If it does not, it is created.

    :param path: path to the directory.
    :type path: str
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logging.warning(f"Directory {path} did not exist. It has been automatically created.")


class _MaxLevelFilter(logging.Filter):
    """Allow only records up to max_level (inclusive)."""

    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


class _MinMaxLevelFilter(logging.Filter):
    """Allow only records between min_level and max_level (inclusive)."""

    def __init__(self, min_level: int, max_level: int):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return self.min_level <= record.levelno <= self.max_level


def setup_script_logging() -> tuple[logging.Logger, logging.Logger, logging.Logger]:
    """Configure three file loggers: errors, warnings, and info."""
    log_dir = "localDB/logging"
    check_create_dir(log_dir)

    script_name = os.path.basename(__file__)
    log_format = f"%(asctime)s | %(levelname)s | script={script_name} | logger_file=%(filename)s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    info_logger = logging.getLogger(f"{script_name}.info")
    info_logger.setLevel(logging.NOTSET)
    info_logger.propagate = False
    info_logger.handlers.clear()

    info_handler = logging.FileHandler(f"{log_dir}/info.log", encoding="utf-8")
    info_handler.setLevel(logging.NOTSET)
    info_handler.addFilter(_MaxLevelFilter(logging.INFO))
    info_handler.setFormatter(formatter)
    info_logger.addHandler(info_handler)

    warnings_logger = logging.getLogger(f"{script_name}.warnings")
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.propagate = False
    warnings_logger.handlers.clear()

    warnings_handler = logging.FileHandler(f"{log_dir}/warnings.log", encoding="utf-8")
    warnings_handler.setLevel(logging.WARNING)
    warnings_handler.addFilter(_MinMaxLevelFilter(logging.WARNING, logging.WARNING))
    warnings_handler.setFormatter(formatter)
    warnings_logger.addHandler(warnings_handler)

    errors_logger = logging.getLogger(f"{script_name}.errors")
    errors_logger.setLevel(logging.ERROR)
    errors_logger.propagate = False
    errors_logger.handlers.clear()

    errors_handler = logging.FileHandler(f"{log_dir}/errors.log", encoding="utf-8")
    errors_handler.setLevel(logging.ERROR)
    errors_handler.setFormatter(formatter)
    errors_logger.addHandler(errors_handler)

    def _handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # capture_locals=True adds local variable values for traceback frames.
        tb_exc = traceback.TracebackException(exc_type, exc_value, exc_traceback, capture_locals=True)
        full_trace = "".join(tb_exc.format())
        errors_logger.error("Unhandled exception:\n%s", full_trace)
        warnings_logger.warning("Unhandled exception captured. See errors.log for full traceback.")

    sys.excepthook = _handle_uncaught_exception

    return errors_logger, warnings_logger, info_logger
