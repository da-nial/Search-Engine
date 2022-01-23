from typing import Dict

import pandas as pd
from hazm import word_tokenize

from models import TokenInfo
from preprocessing import Preprocessor


class BaseQueryEngine:
    initialized = False
    df: pd.DataFrame = None
    num_docs: int = None
    tokens_info: Dict[str, TokenInfo] = None
    preprocessor: Preprocessor = None

    @classmethod
    def initialize(cls, df, preprocessor, tokens_info):
        cls.initialized = True
        cls.df = df
        cls.num_docs = len(df)
        cls.preprocessor = preprocessor
        cls.tokens_info = tokens_info
        print("BaseQueryEngine Initializing finished")

    def __init__(self, query):
        self.query_tokens = self._preprocess_query(query)

    def _preprocess_query(self, query):
        normalized_query = self.preprocessor.normalize_text(query)
        tokens = word_tokenize(normalized_query)
        tokens = self.preprocessor.remove_stopwords(tokens)
        tokens = self.preprocessor.stem_tokens(tokens)
        return tokens

    def process_query(self):
        pass
