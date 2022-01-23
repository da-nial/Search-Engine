from io_handler import get_df, get_tokens_info, PREPROCESSED_PATH
from preprocessing import Preprocessor
from logging import getLogger

from query_processing_indexed import IndexedQueryEngine
from query_processing_vectorized import VectorizedQueryEngine
from query_processing_word_embedding import WordEmbeddingQueryEngine
from query_processing_k_means import KMeansEngineWordEmbedding

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
