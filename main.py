import logging

import pandas as pd

from positional_indexing import get_token_to_positional_indexes_dict
from preprocessing import preprocess_df
from query_processing import process_query


def get_preprocessed_df(file_name):
    preprocessed_file_path = f'./dataset/preprocessed/{file_name}'

    try:
        preprocessed_df = pd.read_excel(preprocessed_file_path, engine="openpyxl")
        logging.info(f'Used preprocessed version found in {preprocessed_file_path}')
    except FileNotFoundError:
        logging.warning(
            f'Preprocessed version of {file_name} could not be found in {preprocessed_file_path}\n'
            f'Using the raw version'
        )
        df = get_df(file_name)
        preprocessed_df = preprocess_df(df)
        df.to_excel(preprocessed_file_path, index=False)

    return preprocessed_df


def get_df(file_name):
    file_path = f'./dataset/{file_name}'
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
        return df
    except FileNotFoundError:
        logging.error(
            f'File {file_name} could not be found in {file_path}\n'
            f'Are you sure you placed the file in the correct path?'
        )
        exit(0)


def main():
    file_name = "IR1_7k_news.xlsx"
    df = get_preprocessed_df(file_name)
    token_to_pos_idx = get_token_to_positional_indexes_dict(df)
    print(df.head())

    while True:
        query = input('Enter your query: ')
        process_query(query, token_to_pos_idx, df)


if __name__ == '__main__':
    main()
