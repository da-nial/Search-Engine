from io_handler import get_df
from preprocessing import Preprocessor
from query_processing import QueryEngine


def main():
    file_name = "IR1_7k_news.xlsx"
    df, is_preprocessed = get_df(file_name)
    preprocessor = Preprocessor(df)

    if not is_preprocessed:
        preprocessor.preprocess_df()

        preprocessed_file_path = f'./dataset/preprocessed/{file_name}'
        df.to_excel(preprocessed_file_path, index=False)

    print(df.head())

    while True:
        query = input('Enter your query: ')
        engine = QueryEngine(df, preprocessor, query)
        engine.process_query()


if __name__ == '__main__':
    main()
