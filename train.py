import argparse
import pandas as pd
import logging.config
from datetime import datetime
from traceback import format_exc

from prod.model import PredictionModel
from prod.settings import (MODEL_PARAMS,
                           LOGGING_CONFIG,
                           NUM_FEATURES,
                           CATEGORICAL_OHE_FEATURES,
                           CATEGORICAL_STE_FEATURES,
                           TARGET,
                           CAT_PREFIX)
from prod.pipeline import pipeline_mapper
from prod.metrics import metrics_stat, evraz_metric


logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        Модель предсказания содержания углерода и температуры чугуна 
        во время процесса продувки металла для хакатона EVRAZ.
        Скрипт для обучения модели.
        Пример запуска:
            python3 train.py --train_data /path/to/train/data --model_path /path/to/model
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train_data", "-d", type=str, dest="d", required=True,
                        help="Путь до обучающего датасета")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True,
                        help="Куда сохранить обученную ML модель")

    return parser.parse_args()


if __name__ == "__main__":

    try:
        # Загрузка обучающего датасета.
        start = datetime.now()
        logger.info('START train.py')
        args = vars(parse_args())
        logger.info('Load train dataframe')
        train_df = pd.read_csv(args['d'])
        logger.info(f'Input shape: {train_df.shape}')
        # Деление на признаковые и целевые датафреймы.
        X_train = train_df[NUM_FEATURES + CATEGORICAL_STE_FEATURES + CATEGORICAL_OHE_FEATURES]
        y_train = train_df[TARGET]
        logger.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
        # Создание пайплайна.
        pipeline = pipeline_mapper(numerical=NUM_FEATURES,
                                   ohe_categorical=CATEGORICAL_OHE_FEATURES,
                                   ste_categorical=CATEGORICAL_STE_FEATURES,
                                   targets=TARGET,
                                   prefix=CAT_PREFIX)
        logger.info('Pipeline created')
        # Обучение модели.
        model = PredictionModel(mapper=pipeline, model_params=MODEL_PARAMS)
        model.fit(X_train, y_train)
        logger.info('Model fitting completed')
        # Сохранение модели.
        logger.info('Save model')
        model.save(args['mp'])
        # Предсказание.
        predictions = model.predict(X_train)
        metrics = metrics_stat(y_train.values, predictions)
        logger.info(f'General metrics stat: {metrics}')
        logger.info(f'Finished in {datetime.now() - start} s')

    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise e
    logger.info('END train.py')
