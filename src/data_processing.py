import joblib
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)


class DataProcessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.processed_data_path = "artifacts/processed"
        os.makedirs(self.processed_data_path, exist_ok=True)

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            logger.info('Read the data successfully')

        except Exception as e:
            logger.error(f'Error reading the data, {e}')
            raise CustomException('Failed to read data', e)

    def handle_outliers(self, column):
        try:
            logger.info('Starting Handling Outliers')
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)

            IQR = Q3-Q1

            lower_value = Q1 - 1.5*IQR
            upper_value = Q3 + 1.5*IQR

            sepal_median = np.median(self.data[column])

            for i in self.data[column]:
                if i > upper_value or i < lower_value:
                    self.data[column] = self.data[column].replace(i,
                                                                  sepal_median)

            logger.info('Handled outliers successfully')

        except Exception as e:
            logger.error(f'Error handling outliers {e}')
            raise CustomException('Failed to handle outliers', e)

    def split_data(self):
        try:
            X = self.data[['SepalLengthCm', 'SepalWidthCm',
                           'PetalLengthCm', 'PetalWidthCm']]
            y = self.data['Species']

            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=0.2,
                                                                random_state=32)

            logger.info('Data splitted successfully')

            joblib.dump(X_train, os.path.join(self.processed_data_path,
                                              'X_train.pkl'))
            joblib.dump(X_test, os.path.join(self.processed_data_path,
                                             'X_test.pkl'))
            joblib.dump(y_train, os.path.join(self.processed_data_path,
                                              'y_train.pkl'))
            joblib.dump(y_test, os.path.join(self.processed_data_path,
                                             'y_test.pkl'))

            logger.info('File saved successfully for data processing step.')

        except Exception as e:
            logger.error(f'Error while splitting data, {e}')
            raise CustomException('Failed to split data', e)

    def run(self):
        self.load_data()
        self.handle_outliers('SepalWidthCm')
        self.split_data()


if __name__ == '__main__':
    data_processor = DataProcessing('artifacts/raw/data.csv')
    data_processor.run()
