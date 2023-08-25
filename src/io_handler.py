from ast import literal_eval
import pickle
from positional_indexing import get_tokens_info_dict
import pandas as pd

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DIRECTORY_PATH = '../datasets/'
PREPROCESSED_PATH = DIRECTORY_PATH + 'preprocessed/'


def get_preprocessed_df(file_name):
    preprocessed_file_path = PREPROCESSED_PATH + file_name + '.pkl'

    try:
        df = pd.read_pickle(preprocessed_file_path)
    except FileNotFoundError:
        raise FileNotFoundError

    return df


def get_df(file_name):
    file_path = DIRECTORY_PATH + file_name + '.xlsx'
    preprocessed_file_path = PREPROCESSED_PATH + file_name + '.pkl'

    is_preprocessed = False
    try:
        df = get_preprocessed_df(file_name)
        logger.info(f'Used preprocessed version found in {preprocessed_file_path}')
        is_preprocessed = True
        return df, is_preprocessed

    except FileNotFoundError:
        logger.warning(
            f'Preprocessed version of {file_name} could not be found in {preprocessed_file_path}\n'
            f'Using the raw version'
        )
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
            return df, is_preprocessed
        except FileNotFoundError:
            logger.error(
                f'File {file_name} could not be found in {file_path}\n'
                f'Are you sure you placed the file in the correct path?'
            )
            exit(0)


def get_tokens_info(file_name, df):
    file_path = PREPROCESSED_PATH + file_name

    try:
        f = open(file_path, 'rb')
        tokens_info = pickle.load(f)
        logger.info(f"Positional Indexing loaded from {file_path}")
    except FileNotFoundError:
        logger.info(f"Positional Indexing cache wasn't found in {file_path}")
        tokens_info = get_tokens_info_dict(df=df)
        logger.info("Positional Indexing computed.")
        with open(file_path, 'wb') as f:
            pickle.dump(tokens_info, f)
            logger.info(f"Positional Indexing save to path {file_path}.")

    return tokens_info
