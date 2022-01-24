from query_processing_word_embedding import WordEmbeddingQueryEngine, Doc
from utils import l2_norm, mean, similarity
from random import sample, randint
from typing import List, Tuple, Union

import pickle


class Cluster:
    def __init__(self, id, centroid, docs: List[Doc] = None):
        if docs is None:
            docs = []

        self.id = id
        self.centroid = centroid
        self.docs = docs

    def add_doc(self, doc: Doc):
        self.docs.append(doc)

    def empty(self):
        self.docs = []

    def compute_centroid(self):
        doc_vectors = [doc.vec for doc in self.docs]

        if len(doc_vectors) == 0:
            print("Cluster without any documents!")
            return None

        new_centroid = mean(doc_vectors)
        return new_centroid

    @property
    def rss(self):
        return sum(
            [l2_norm(doc.vec - self.centroid) for doc in self.docs]
        )


class KMeansQueryEngine(WordEmbeddingQueryEngine):
    num_clusters = 15  # K (num seeds)
    max_iterations = 50
    min_cluster_changed_ratio = 0.01
    epochs = 5  # Number of tries for finding the optimal clustering (based on rss)

    clusters = None
    b = 3  # We search `b` nearest clusters for query

    @classmethod
    def initialize(cls, df, preprocessor, tokens_info):
        super(KMeansQueryEngine, cls).initialize(df, preprocessor, tokens_info)
        cls.initialize_clusters()

    @classmethod
    def initialize_clusters(cls):
        try:
            f = open('./dataset/preprocessed/clustering.pkl', 'rb')
            cls.clusters = pickle.load(f)
        except FileNotFoundError:
            cls.compute_clusters()
            cls.save_clusters()

    @classmethod
    def compute_clusters(cls):
        optimal_clusters = None
        min_rss = None

        for i in range(cls.epochs):
            print(f"Epoch #{i}")
            clusters = cls.k_means()
            rss = cls.rss(clusters)

            if (min_rss is None) or (rss < min_rss):
                optimal_clusters = clusters
                min_rss = rss

        cls.clusters = optimal_clusters

    @classmethod
    def k_means(cls):
        doc_id_to_cluster_id: List[Union[int, None]] = [None for doc_id in range(cls.num_docs)]

        seeds = cls.create_seeds()
        clusters = cls.create_clusters(seeds)

        stop_criteria = False
        iteration = 0
        while not stop_criteria:
            iteration += 1
            num_docs_with_changed_cluster = 0

            for cluster_id, cluster in enumerate(clusters):
                cluster.empty()

            for doc_id, doc_cluster_id in enumerate(doc_id_to_cluster_id):
                doc_cluster_changed = cls.reassign_doc(doc_id, doc_id_to_cluster_id, clusters)
                if doc_cluster_changed:
                    num_docs_with_changed_cluster += 1

            for cluster_id, cluster in enumerate(clusters):
                new_centroid = cluster.compute_centroid()

                if new_centroid is None:  # Why would a cluster be empty after the assignments?
                    random_doc_id = randint(0, cls.num_docs)
                    new_centroid = cls.docs[random_doc_id].vec

                cluster.centroid = new_centroid

            clusters_changed_ratio = num_docs_with_changed_cluster / cls.num_docs
            stop_criteria = (
                    iteration > cls.max_iterations or
                    clusters_changed_ratio < cls.min_cluster_changed_ratio
            )

            if iteration % 10 == 0:
                print(f"\tIteration #{iteration}")

        return clusters

    @classmethod
    def create_seeds(cls):
        doc_ids = list(range(cls.num_docs))
        doc_ids_selected = sample(doc_ids, cls.num_clusters)

        seeds = []
        for doc_id in doc_ids_selected:
            seeds.append(
                cls.docs[doc_id].vec
            )
        return seeds

    @classmethod
    def create_clusters(cls, seeds=None) -> List[Cluster]:
        clusters: List[Cluster] = []

        for cluster_id in range(cls.num_clusters):
            clusters.append(
                Cluster(id=cluster_id, centroid=seeds[cluster_id])
            )

        return clusters

    @classmethod
    def reassign_doc(cls, doc_id: int, doc_id_to_cluster_id: List[Union[None, int]], clusters: List[Cluster]):
        doc_vec = cls.docs[doc_id].vec

        # Get new cluster id
        doc_cluster_id = doc_id_to_cluster_id[doc_id]
        doc_new_cluster_id = cls.get_vec_n_nearest_cluster_ids(doc_vec, clusters, n=1)[0]

        # Update doc_to_cluster index
        doc_id_to_cluster_id[doc_id] = doc_new_cluster_id

        # Update new cluster docs (cluster_to_doc index)
        doc_new_cluster = clusters[doc_new_cluster_id]
        doc_new_cluster.add_doc(
            Doc(doc_id, doc_vec)
        )

        doc_cluster_changed = (doc_cluster_id != doc_new_cluster_id)
        return doc_cluster_changed

    @classmethod
    def get_vec_n_nearest_cluster_ids(cls, vec: List, clusters: List[Cluster], n=1):
        cluster_id_similarities: [Tuple[int, float]] = []  # A List of (cluster_id, similarity)

        for cluster_id in range(cls.num_clusters):
            centroid = clusters[cluster_id].centroid
            _similarity = similarity(vec, centroid)

            cluster_id_similarities.append(
                (cluster_id, _similarity)
            )
        cluster_id_similarities.sort(key=lambda item: item[1], reverse=True)
        n_nearest_cluster_ids = [cluster_id for cluster_id, _similarity in cluster_id_similarities[:n]]

        return n_nearest_cluster_ids

    @staticmethod
    def rss(clusters: List[Cluster]):
        rss = 0
        for cluster in clusters:
            rss += cluster.rss
        return rss

    @classmethod
    def save_clusters(cls):
        with open('./dataset/preprocessed/clustering.pkl', 'wb') as f:
            pickle.dump(cls.clusters, f)

    def get_similarities(self):
        k = 5

        query_vector = self.get_query_vector()

        n_nearest_cluster_ids = self.get_vec_n_nearest_cluster_ids(
            vec=query_vector,
            clusters=self.clusters,
            n=self.b
        )
        n_nearest_cluster_docs = []
        for cluster_id in n_nearest_cluster_ids:
            cluster = self.clusters[cluster_id]
            n_nearest_cluster_docs.extend(
                cluster.docs
            )

        similarities = []
        for doc in n_nearest_cluster_docs:
            doc_similarity = similarity(doc.vec, query_vector)
            similarities.append(
                (doc.id, doc_similarity)
            )

        similarities = sorted(similarities, key=lambda item: item[1], reverse=True)
        return similarities[:k]
