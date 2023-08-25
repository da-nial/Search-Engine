from math import log10
from typing import List, Dict, Tuple

from utils import l2_norm

from models import TokenInfo

from query_processing_base import BaseQueryEngine


class VectorizedQueryEngine(BaseQueryEngine):
    doc_lengths: List[float] = None
    token_champions_list: Dict[str, List[int]] = None

    @classmethod
    def initialize(cls, df, preprocessor, tokens_info):
        super(VectorizedQueryEngine, cls).initialize(df, preprocessor, tokens_info)

        cls.initialize_doc_lengths()
        cls.initialize_token_champions_list()
        print("VectorizedQueryEngine Initializing finished")

    @classmethod
    def initialize_doc_lengths(cls):
        cls.doc_lengths = []
        for doc_id in range(cls.num_docs):
            doc_weights = []
            for token in cls.df.iloc[doc_id].tokens:
                token_weight = cls.tokens_info.get(token, TokenInfo()).get_weight(doc_id, num_docs=cls.num_docs)
                doc_weights.append(token_weight)

            doc_length = l2_norm(doc_weights)
            cls.doc_lengths.append(doc_length)

    @classmethod
    def initialize_token_champions_list(cls):
        r = 20
        cls.token_champions_list = {}
        for token in cls.tokens_info.keys():
            token_info = cls.tokens_info.get(token)

            doc_weights = []
            for doc_id in token_info.docs_containing:
                weight = token_info.get_weight(doc_id, cls.num_docs)
                doc_weights.append((doc_id, weight))

            doc_weights = sorted(doc_weights, key=lambda item: item[1], reverse=True)
            doc_weights = doc_weights[:r]
            token_champions_list = [doc_id for doc_id, weight in doc_weights]

            cls.token_champions_list[token] = token_champions_list

    def process_query(self):
        results = self.get_similarities(use_champions_list=True)
        self._print_similarities_result(results)

    def _get_query_token_weight(self, token) -> float:
        f = self.query_tokens.count(token)
        tf = log10(1 + f)

        num_docs_containing = self.tokens_info.get(token, TokenInfo()).num_docs_containing
        idf = log10(self.num_docs / num_docs_containing)

        weight = tf * idf
        return weight

    def get_similarities(self, use_champions_list=False):
        k = 5

        scores = [0 for doc_id in range(self.num_docs)]

        for token in self.query_tokens:
            token_info = self.tokens_info.get(token, TokenInfo())

            token_weight_in_query = self._get_query_token_weight(token)

            if use_champions_list:
                target_docs = self.token_champions_list.get(token, [])
            else:
                target_docs = token_info.docs_containing

            for doc_id in target_docs:
                token_weight_in_doc = token_info.get_weight(doc_id, self.num_docs)
                scores[doc_id] += token_weight_in_query * token_weight_in_doc

        similarities = []
        for doc_id in range(self.num_docs):
            scores[doc_id] /= self.doc_lengths[doc_id]
            similarities.append((doc_id, scores[doc_id]))

        similarities = sorted(similarities, key=lambda item: item[1], reverse=True)
        return similarities[:k]

    def _print_similarities_result(self, results: List[Tuple[int, float]]):
        print('Docs found (ranked):\n')
        for doc, similarity in results:
            title = self.df.iloc[doc]['title'].strip('\n')
            print(f"Doc#{doc} Title: {title[:80]}...  Similarity: {similarity}")
        print('\n')
