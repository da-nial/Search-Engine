from io_handler import get_df, get_tokens_info, PREPROCESSED_PATH
from preprocessing import Preprocessor
from logging import getLogger

from indexed_query_processing import IndexedQueryEngine
from vectorized_query_processing import VectorizedQueryEngine
from word_embedding_query_processing import WordEmbeddingQueryEngine
from k_means_query_processing import KMeansEngineWordEmbedding

logger = getLogger('main')


def main():
    # file_name_without_extension = "IR1_7k_news"
    file_name_without_extension = "IR1_7k_news"
    file_name = file_name_without_extension + ".xlsx"

    df, is_preprocessed = get_df(file_name)
    preprocessor = Preprocessor(df)

    if not is_preprocessed:
        preprocessor.preprocess_df()

        # Cache
        file_name = file_name_without_extension + '.pkl'
        preprocessed_file_path = PREPROCESSED_PATH + file_name
        df.to_pickle(preprocessed_file_path)

    tokens_info = get_tokens_info(file_name=file_name_without_extension + '_pos_idx' + ".pkl", df=df)

    print(df.head())

    # engine_cls = IndexedQueryEngine
    # engine_cls = VectorizedQueryEngine
    # engine_cls = WordEmbeddingQueryEngine
    engine_cls = KMeansEngineWordEmbedding

    engine_cls.initialize(df=df, preprocessor=preprocessor, tokens_info=tokens_info)

    while True:
        query = input('Enter your query: ')
        engine = engine_cls(query)
        engine.process_query()


if __name__ == '__main__':
    main()
