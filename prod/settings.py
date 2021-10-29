# TODO: добавить списки признаков и параметры модели
# cols_names = ['SEC','RAS','POL','VDL','NML','VES','plavka_VR_NACH', 'plavka_VR_KON','plavka_NMZ','plavka_NAPR_ZAD',
#              'plavka_STFUT','plavka_TIPE_FUR','plavka_ST_FURM','plavka_TIPE_GOL','plavka_ST_GOL',
#              'VDSYP','NMSYP','VSSYP','DAT_OTD','TYPE_OPER','NOP','VR_NACH','VR_KON','O2','VES','T','SI',
#              'MN','S','P','CR','NI','CU','O2_pressure','T фурмы 2','T фурмы 1','AR','CO','CO2','H2','N2',
#              'O2','T','V','NPLV','Time','DATA_ZAMERA','TI','V',]
# Целевая переменная.
TARGET = ['C', 'TST']
# Категориальные признаки, для которых применяется smoothed target encoding.
CATEGORICAL_STE_FEATURES = ['44']
# Категориальные признаки, для которых применяется one-hot-encoding.
CATEGORICAL_OHE_FEATURES = []
# Численные признаки.
NUM_FEATURES = [str(num) for num in range(43)]
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
