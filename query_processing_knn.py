from heapq import heapify, heappush, heappushpop, nlargest

from query_processing_word_embedding import WordEmbeddingQueryEngine, Doc
from query_processing_k_means import KMeansQueryEngine
from utils import similarity
from typing import Dict, List
from models import TokenInfo

from math import log10


class MaxHeap:
    def __init__(self, top_n):
        self.h = []
        self.length = top_n
        heapify(self.h)

    def add(self, element):
        if len(self.h) < self.length:
            heappush(self.h, element)
        else:
            heappushpop(self.h, element)

    def get_top(self):
        return nlargest(self.length, self.h)


class KNN(WordEmbeddingQueryEngine):
    k = 15
    cat_to_doc: Dict[str, List[int]] = {}  # Maintain an inverted index from category to list of documents of that cat

    unlabeled_df = None
    unlabeled_df_tokens_info: Dict[str, TokenInfo] = None
    num_unlabeled_docs = None
    unlabeled_docs = List[Doc]

    @classmethod
    def initialize_doc_vectors(cls):
        cls.docs = []
        for doc_id in range(cls.num_docs):
            doc_vector = cls.compute_doc_vector(doc_id)
            doc_cat = cls.df.iloc[doc_id]['topic']
            cls.docs.append(
                Doc(doc_id, doc_vector, doc_cat)
            )
        print('KNN initialization finished!')

    @classmethod
    def initialize_cat_to_doc_index(cls):
        cls.docs = []
        for doc in cls.docs:
            cat = doc.cat
            cls.cat_to_doc[cat] = cls.cat_to_doc.get(cat, []).append(doc.id)


class KNNClassifier(KNN):
    k = 15
    cat_to_doc: Dict[str, List[int]] = {}  # Maintain an inverted index from category to list of documents of that cat

    def __init__(self, unlabeled_df, unlabeled_df_tokens_info):
        super().__init__(query='')
        self.unlabeled_df = unlabeled_df
        self.unlabeled_df_tokens_info = unlabeled_df_tokens_info
        self.num_unlabeled_docs = len(unlabeled_df)
        self.unlabeled_docs: List[Doc] = []

    def classify(self):
        self.unlabeled_df.insert(2, 'topic', '')

        self.compute_unlabeled_docs_vectors()
        for doc_id, doc in enumerate(self.unlabeled_docs):
            doc.cat = self.classify_doc_vec(doc.vec)
            self.unlabeled_df.at[doc.id, 'topic'] = doc.cat

            if doc_id % 500 == 0:
                print(f'#Docs Labeled: {doc_id} ({doc_id * 100 / self.num_unlabeled_docs}%)')

        return self.unlabeled_df

    def compute_unlabeled_docs_vectors(self):
        self.unlabeled_docs = []
        for doc_id in range(self.num_unlabeled_docs):
            doc_vector = self.compute_unlabeled_doc_vector(doc_id)
            self.unlabeled_docs.append(
                Doc(doc_id, doc_vector)
            )

    def compute_unlabeled_doc_vector(self, unlabeled_doc_id):
        weights = []
        token_vectors = []

        for token in self.unlabeled_df.iloc[unlabeled_doc_id].tokens:
            if token not in self.unlabeled_df_tokens_info:
                continue
            weight = self.get_unlabeled_doc_token_weight(token, unlabeled_doc_id)
            try:
                token_vector = weight * self.w2v_model.wv.get_vector(token)
            except KeyError:
                continue

            weights.append(weight)
            token_vectors.append(token_vector)

        return sum(token_vectors) / sum(weights)

    def get_unlabeled_doc_token_weight(self, token, unlabeled_doc_id):
        tf = self.unlabeled_df_tokens_info[token].tf(unlabeled_doc_id)

        num_docs_containing = 1
        if token in self.tokens_info:
            num_docs_containing = self.tokens_info[token].num_docs_containing

        idf = log10(self.num_docs / num_docs_containing)
        return tf * idf

    def classify_doc_vec(self, new_doc_vec):
        nearest_docs_ids = self.get_k_nearest_docs_ids(new_doc_vec)

        neighbors_cat_counts = {}

        for doc_id in nearest_docs_ids:
            doc_cat = self.docs[doc_id].cat
            neighbors_cat_counts[doc_cat] = neighbors_cat_counts.get(doc_cat, 0) + 1

        cat_with_max_count = max(neighbors_cat_counts, key=neighbors_cat_counts.get)
        return cat_with_max_count

    def get_k_nearest_docs_ids(self, new_doc_vec) -> List[int]:
        k_nearest_neighbors = MaxHeap(self.k)  # Each neighbor (heap item) is a Tuple (similarity, doc_id)

        for doc in self.docs:
            _similarity = similarity(doc.vec, new_doc_vec)

            k_nearest_neighbors.add(
                (_similarity, doc.id)
            )

        k_nearest_neighbors.get_top()
        k_nearest_docs_ids = [doc_id for _similarity, doc_id in k_nearest_neighbors.get_top()]
        return k_nearest_docs_ids


class KNNQueryEngine(KNN):
    def __init__(self, query, cat=None):
        super().__init__(query)
        self.cat = cat

    def get_similarities(self):
        k = 5

        query_vector = self.get_query_vector()

        target_docs = self.docs
        if self.cat is not None:
            docs_with_same_cat = self.cat_to_doc.get(self.cat, [])
            target_docs = docs_with_same_cat

        similarities = []
        for doc in target_docs:
            doc_similarity = similarity(doc.vec, query_vector)
            similarities.append(
                (doc.id, doc_similarity)
            )

        similarities = sorted(similarities, key=lambda item: item[1], reverse=True)
        return similarities[:k]
