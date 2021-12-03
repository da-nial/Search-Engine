import logging
from ast import literal_eval

import pandas as pd


def get_preprocessed_df(file_name):
    preprocessed_file_path = f'./dataset/preprocessed/{file_name}'

    try:
        df = pd.read_excel(preprocessed_file_path, engine="openpyxl")
        df['tokens'] = df['tokens'].apply(literal_eval)
    except FileNotFoundError:
        raise FileNotFoundError

    return df


def get_df(file_name):
    file_path = f'./dataset/{file_name}'
    preprocessed_file_path = f'./dataset/preprocessed/{file_name}'

    is_preprocessed = False
    try:
        df = get_preprocessed_df(file_name)
        logging.info(f'Used preprocessed version found in {preprocessed_file_path}')
        is_preprocessed = True
        return df, is_preprocessed

    except FileNotFoundError:
        logging.warning(
            f'Preprocessed version of {file_name} could not be found in {preprocessed_file_path}\n'
            f'Using the raw version'
        )
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
            return df, is_preprocessed
        except FileNotFoundError:
            logging.error(
                f'File {file_name} could not be found in {file_path}\n'
                f'Are you sure you placed the file in the correct path?'
            )
            exit(0)


def set_preprocessed_df(df, file_path):
    df.to_excel(file_path, index=False)
