import logging.config

from neoassist.utils import abs_path


def setup_logging(log_level):
    log_dir = abs_path("../", __file__) / "logs"
    log_path = log_dir / "app.log"

    log_dir.mkdir(parents=True, exist_ok=True)

    console_formatter = {
        "class": "colorlog.ColoredFormatter",
        "format": "%(log_color)s%(asctime)s [%(levelname)s] [%(name)s] %(filename)s:%(lineno)d - %(message)s",
        "log_colors": {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    }

    json_formatter = {
        "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
        "format": "%(asctime)s [%(levelname)s] [%(name)s] %(filename)s:%(lineno)d - %(message)s",
    }

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"json": json_formatter, "console": console_formatter},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "console",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": log_path,
                "maxBytes": 10000000,
                "backupCount": 3,
                "encoding": "utf-8",
            },
        },
        "root": {"handlers": ["console", "file"], "level": log_level},
    }

    logging.config.dictConfig(config)
