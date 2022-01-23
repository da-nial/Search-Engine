from heapq import heapify, heappush, heappushpop, nlargest

from query_processing_word_embedding import Doc
from query_processing_k_means import KMeansEngineWordEmbedding
from utils import similarity
from typing import List


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


class KNN(KMeansEngineWordEmbedding):
    k = 15
    cat_to_doc = []  # Maintain an inverted index from

    @classmethod
    def initialize_doc_vectors(cls):
        print('KNN initialization called!')
        cls.docs = []
        for doc_id in range(cls.num_docs):
            doc_vector = cls.compute_doc_vector(doc_id)
            doc_cat = cls.df['topic']
            cls.docs.append(
                Doc(doc_id, doc_vector, doc_cat)
            )

    def classify(self, new_doc_id):
        nearest_docs_ids = self.get_k_nearest_docs_ids(new_doc_id)

        neighbors_cat_counts = {}

        for doc_id in nearest_docs_ids:
            doc_cat = self.docs[doc_id].cat
            neighbors_cat_counts[doc_cat] = neighbors_cat_counts.get(doc_cat, 0) + 1

        cat_with_max_count = max(neighbors_cat_counts, key=neighbors_cat_counts.get)
        return cat_with_max_count

    def get_k_nearest_docs_ids(self, new_doc_id: int) -> List[int]:
        new_doc_vec = self.docs[new_doc_id].vec

        k_nearest_neighbors = MaxHeap(self.k)  # Each neighbor (heap item) is a Tuple (similarity, doc_id)

        for doc in self.docs:
            _similarity = similarity(doc.vec, new_doc_vec)

            k_nearest_neighbors.add(
                (_similarity, doc.id)
            )

        k_nearest_neighbors.get_top()
        k_nearest_docs_ids = [doc_id for _similarity, doc_id in k_nearest_neighbors.get_top()]
        return k_nearest_docs_ids

    def __init__(self, query, cat=None):
        super().__init__(query)
        self.cat = cat

    def process_query(self):
        pass
