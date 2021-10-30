# Целевые переменные
TARGET = ['TST', 'C']
# Категориальные признаки, для которых применяется smoothed target encoding.
CATEGORICAL_STE_FEATURES = ['plavka_ST_FURM']
# Категориальные признаки, для которых применяется one-hot-encoding.
CATEGORICAL_OHE_FEATURES = []
# Численные признаки.
NUM_FEATURES = ['O2', 'N2','T', 'H2', 'CO2', 'CO', 'AR']
# Суффикс для обозначения категориальных переменных.
CAT_PREFIX = "#CAT#_"
# Параметры модели.
MODEL_PARAMS = dict(
    n_estimators=1500,
    loss_function='MultiRMSE',
    learning_rate=0.01,
    min_child_samples=5,
    random_seed=42,
    verbose=0
        )
# Параметры логирования.
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"},
    },
    "handlers": {
        "file_handler": {
            "level": "INFO",
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": 'train.log',
            "mode": "a",
        },
        "print_to_console": {
            "level": "INFO",
            "formatter": "default",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {"handlers": ["file_handler", "print_to_console"], "level": "INFO", "propagate": False},
    },
}