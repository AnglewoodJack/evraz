import argparse
import pandas as pd
import logging.config
from traceback import format_exc
from prod.model import PredictionModel
from prod.settings import LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES, CATEGORICAL_STE_FEATURES, TARGET

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        Модель предсказания содержания углерода и температуры чугуна 
        во время процесса продувки металла для хакатона EVRAZ.
        Скрипт для предсказания модели.
        Пример запуска:
            python3 predict.py --test_data /path/to/test/data --model_path /path/to/model --output /path/to/output/file.csv.gzip
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--test_data", "-d", type=str, dest="d", required=True,
                        help="Путь до отложенной выборки")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True,
                        help="Путь до сериализованной ML модели")
    parser.add_argument("--output", "-o", type=str, dest="o", required=True,
                        help="Путь до выходного файла")
    return parser.parse_args()

if __name__ == "__main__":

    try:
        logger.info('START predict.py')
        args = vars(parse_args())
        logger.info('Load test dataframe')
        test_df = pd.read_csv(args['d'])
        logger.info(f'Input shape: {test_df.shape}')
        # Загрузка модели.
        logger.info('Load model')
        model = PredictionModel.load(args['mp'])
        # Предсказание.
        logger.info('Predict')
        target = model.predict(test_df[NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES])
        # Сохранение результатов.
        logger.info('Save results')
        target_df = pd.DataFrame(data=target, columns=TARGET)
        pd.concat([test_df['NPLV'], target_df], axis=1).to_csv(args['o'], index=False)

    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise (e)
    logger.info('END predict.py')
