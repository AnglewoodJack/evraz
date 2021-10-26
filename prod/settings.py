# TODO: добавить списки признаков и параметры модели
# Целевая переменная.
TARGET = ''
# Категориальные признаки, для которых применяется smoothed target encoding.
CATEGORICAL_STE_FEATURES = []
# Категориальные признаки, для которых применяется one-hot-encoding.
CATEGORICAL_OHE_FEATURES = []
# Численные признаки.
NUM_FEATURES = []
# Параметры модели.
MODEL_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.01,
    reg_alpha=1,
    num_leaves=40,
    min_child_samples=5,
    importance_type="gain",
    n_jobs=6,
    random_state=563,
    categorical_feature= CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES,
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
    },
    "loggers": {
        "": {"handlers": ["file_handler"], "level": "INFO", "propagate": False},
    },
}
