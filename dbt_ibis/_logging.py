import logging


def configure_logging(logger: logging.Logger) -> None:
    log_level = logging.INFO
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Imitate dbt's log format but add dbt-ibis before the log message
    formatter = logging.Formatter(
        "%(asctime)s  dbt-ibis: %(message)s", datefmt="%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
