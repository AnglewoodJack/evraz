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
                           TARGET)
from prod.metrics import metrics_stat, evraz_metric
from _testing.traintest import traintest
from _testing.dummy_data_gen import generate_dummy_df

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        Модель прдсказания содержания углерода и температуры чугуна 
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
        generate_dummy_df(1000, 45, 1, 1)
        logger.info('\n\nSTART train.py')
        args = vars(parse_args())
        logger.info('Load train dataframe')
        train_df = pd.read_csv(args['d'])
        logger.info(f'Input shape: {train_df.shape}')
        # Деление на признаковые и целевые датафреймы.
        X_train = train_df[NUM_FEATURES + CATEGORICAL_STE_FEATURES + CATEGORICAL_OHE_FEATURES]
        y_train = train_df[TARGET]
        logger.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
        model = PredictionModel(numerical_features=NUM_FEATURES,
                                ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                                ste_categorical_features=CATEGORICAL_STE_FEATURES,
                                model_params=MODEL_PARAMS)
        logger.info('Fit model')
        model.fit(X_train, y_train)
        # Сохранение модели.
        logger.info('Save model')
        model.save(args['mp'])
        # Предсказание.
        predictions = model.predict(X_train)
        # TODO: использовать нужную метрику.
        metrics = metrics_stat(y_train.values, predictions)
        logger.info(f'Metrics stat for training data with offers prices: {metrics}')
        logger.info(f'Running catboost for comparing with evraz metrics...')
        answers, user_csv = traintest(train_df)
        logger.info(f'Evraz metric using catboost: {evraz_metric(answers, user_csv)}')
        logger.info(f'Finished in {datetime.now() - start} s')

    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise e
    logger.info('END train.py')