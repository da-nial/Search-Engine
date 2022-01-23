import multiprocessing
from gensim.models import Word2Vec
from typing import List, Tuple
from positional_indexing import TokenInfo
from math import log10

from utils import similarity

from base_query_processing import BaseQueryEngine


class Doc:
    def __init__(self, id, vec, cat=None):
        self.id = id
        self.vec = vec
        self.cat = cat


class WordEmbeddingQueryEngine(BaseQueryEngine):
    w2v_model: Word2Vec = None
    docs: List[Doc]

    @classmethod
    def initialize(cls, df, preprocessor, tokens_info):
        super(WordEmbeddingQueryEngine, cls).initialize(df, preprocessor, tokens_info)
        cls.initialize_w2v_model()
        cls.initialize_doc_vectors()
        print("WordEmbeddingQueryEngine Initializing finished")

    @classmethod
    def initialize_w2v_model(cls):
        cores = multiprocessing.cpu_count()
        print(f'Num CPU cores: {cores}')

        # training_data = cls.df['tokens'].tolist()
        w2v_model = Word2Vec(
            min_count=1,
            window=5,
            vector_size=300,
            alpha=0.03,
            workers=cores - 1
        )
        # w2v_model.build_vocab(training_data)
        # print(f'Built vocab from training data')
        #
        # w2v_model.train(training_data, total_examples=w2v_model.corpus_count, epochs=20)
        # w2v_model.save('./dataset/w2v/w2v_300d.model')

        w2v_model = Word2Vec.load('./dataset/w2v/w2v_150k_hazm_300_v2.model')
        # w2v_model.wv.load('./dataset/w2v/w2v_150k_hazm_300_v2.model.wv.vectors.npy')

        cls.w2v_model = w2v_model

    @classmethod
    def initialize_doc_vectors(cls):
        cls.docs = []
        for doc_id in range(cls.num_docs):
            doc_vector = cls.compute_doc_vector(doc_id)
            cls.docs.append(
                Doc(doc_id, doc_vector)
            )

    @classmethod
    def compute_doc_vector(cls, doc_id):
        weights = []
        token_vectors = []

        for token in cls.df.iloc[doc_id].tokens:
            if token not in cls.tokens_info:
                continue
            token_info = cls.tokens_info.get(token)
            weight = token_info.get_weight(doc_id, cls.num_docs)

            try:
                token_vector = weight * cls.w2v_model.wv.get_vector(token)
            except KeyError:
                continue

            weights.append(weight)
            token_vectors.append(token_vector)

        return sum(token_vectors) / sum(weights)

    def get_query_vector(self):
        weights = []
        token_vectors = []

        for token in self.query_tokens:
            if token not in self.tokens_info:
                continue

            token_info = self.tokens_info.get(token)
            weight = self._get_query_token_weight(token)
            token_vector = weight * self.w2v_model.wv.get_vector(token)

            weights.append(weight)
            token_vectors.append(token_vector)

        return sum(token_vectors) / sum(weights)

    def _get_query_token_weight(self, token) -> float:
        f = self.query_tokens.count(token)
        tf = log10(1 + f)

        num_docs_containing = self.tokens_info.get(token, TokenInfo()).num_docs_containing
        idf = log10(self.num_docs / num_docs_containing)

        weight = tf * idf
        return weight

    def process_query(self):
        results = self.get_similarities()
        self._print_similarities_result(results)

    def get_similarities(self):
        k = 5

        similarities = []
        query_vector = self.get_query_vector()

        for doc_id in range(self.num_docs):
            doc_vec = self.docs[doc_id].vec
            doc_similarity = similarity(doc_vec, query_vector)
            similarities.append((doc_id, doc_similarity))

        similarities = sorted(similarities, key=lambda item: item[1], reverse=True)
        return similarities[:k]

    def _print_similarities_result(self, results: List[Tuple[int, float]]):
        print('Docs found (ranked):\n')
        for doc, similarity in results:
            title = self.df.iloc[doc]['title'].strip('\n')
            print(f"Doc#{doc} Title: {title[:80]}...  Similarity: {similarity}")
        print('\n')
