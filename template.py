import logging
import os

from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    'artifacts/raw/data',
    'artifacts/processed/data',
    'kubeflow_pipeline/__init__.py',
    'notebook/code.ipynb',
    'src/data_processing.py',
    'src/__init__.py',
    'src/logger.py',
    'src/model_training.py',
    'src/custom_exception.py',
    'pipeline/__init__.py',
    'templates/style.css',
    'static/data',
    'setup.py',
    'requirements.txt'

]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory: {filedir} for the file {filename}')

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f'Creating empty file: {filepath}')

    else:
        logging.info(f'{filename} already exists')
