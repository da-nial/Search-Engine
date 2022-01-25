from io_handler import get_df, get_tokens_info, PREPROCESSED_PATH
from preprocessing import Preprocessor
from logging import getLogger

from query_processing_indexed import IndexedQueryEngine
from query_processing_vectorized import VectorizedQueryEngine
from query_processing_word_embedding import WordEmbeddingQueryEngine
from query_processing_k_means import KMeansQueryEngine
from query_processing_knn import KNNQueryEngine, KNNClassifier

logger = getLogger('main')

import pandas as pd


def main():
    file_name = 'IR3_50k_news'

    df, is_preprocessed = get_df(file_name)
    preprocessor = Preprocessor(df)

    if not is_preprocessed:
        preprocessor.preprocess_df()

        # Cache
        preprocessed_file_path = PREPROCESSED_PATH + file_name + '.pkl'
        df.to_pickle(preprocessed_file_path)

    tokens_info = get_tokens_info(file_name=file_name + '_pos_idx' + '.pkl', df=df)

    print(df.head())

    # engine_cls = IndexedQueryEngine
    # engine_cls = VectorizedQueryEngine
    # engine_cls = WordEmbeddingQueryEngine
    # engine_cls = KMeansQueryEngine
    engine_cls = KNNQueryEngine

    engine_cls.initialize(df=df, preprocessor=preprocessor, tokens_info=tokens_info)

    while True:
        query = input('Enter your query: ')
        engine = engine_cls(query)
        engine.process_query()


def classify_docs():
    file_name = "IR3_50k_news"

    labeled_df, is_preprocessed = get_df(file_name)
    preprocessor = Preprocessor(labeled_df)

    if not is_preprocessed:
        preprocessor.preprocess_df()

        # Cache
        file_name = file_name + '.pkl'
        preprocessed_file_path = PREPROCESSED_PATH + file_name
        labeled_df.to_pickle(preprocessed_file_path)

    print(labeled_df.head())
    tokens_info = get_tokens_info(file_name=file_name + '_pos_idx' + ".pkl", df=labeled_df)

    engine_cls = KNNClassifier
    engine_cls.initialize(df=labeled_df, preprocessor=preprocessor, tokens_info=tokens_info)

    # ----------------------------------------------------

    # unlabeled_df, is_preprocessed = pd.read_pickle('./dataset/IR1_100_news.pkl'), False
    # file_name = "IR1_100_news"

    file_name = "IR1_7k_news"
    unlabeled_df, is_preprocessed = get_df(file_name)
    preprocessor = Preprocessor(unlabeled_df)

    if not is_preprocessed:
        preprocessor.preprocess_df()

        # Cache
        file_name = file_name + '.pkl'
        preprocessed_file_path = PREPROCESSED_PATH + file_name
        unlabeled_df.to_pickle(preprocessed_file_path)

    unlabeled_df_tokens_info = get_tokens_info(file_name=file_name + '_pos_idx' + ".pkl",
                                               df=unlabeled_df)

    engine = engine_cls(
        unlabeled_df=unlabeled_df,
        unlabeled_df_tokens_info=unlabeled_df_tokens_info
    )
    labeled_unlabeled_df = engine.classify()

    merged = labeled_df + labeled_unlabeled_df
    file_name = 'IR3_57k_news' + '.pkl'
    preprocessed_file_path = PREPROCESSED_PATH + file_name
    merged.to_pickle(preprocessed_file_path)


if __name__ == '__main__':
    main()
    # classify_docs()
