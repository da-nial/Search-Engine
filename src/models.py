from math import log10


class TokenInfo:
    def __init__(
            self,
            total_frequency: int = 0,
    ):
        self.total_frequency = total_frequency
        self.doc_to_indexes = {}

    @property
    def num_docs_containing(self):
        return len(self.doc_to_indexes.keys())

    @property
    def docs_containing(self):
        return self.doc_to_indexes.keys()

    def update_indexes(self, doc_id, idx):
        self.total_frequency += 1

        if doc_id not in self.doc_to_indexes:
            self.doc_to_indexes.update({doc_id: []})

        self.doc_to_indexes[doc_id].append(idx)

    def get_indexes(self, doc_id):
        return self.doc_to_indexes.get(doc_id, [])

    def get_frequency(self, doc_id: int):
        return len(self.doc_to_indexes.get(doc_id, []))

    def tf(self, doc_id):
        f = self.get_frequency(doc_id)
        tf = log10(1 + f)
        return tf

    def get_weight(self, doc_id: int, num_docs: int):
        tf = self.tf(doc_id)
        idf = log10(num_docs / self.num_docs_containing)

        return tf * idf


class TokenDocInfo:
    def __init__(self):
        self.indexes = []
        self.frequency = 0
        self.weight = None


class Substring:
    def __init__(self, first_index):
        self.indexes = [first_index]

    @property
    def score(self):
        return len(self.indexes)

    @property
    def last_token_index(self):
        return self.indexes[-1]

    def add_next_token_index(self):
        previous_last_token_index = self.last_token_index
        self.indexes.append(previous_last_token_index + 1)
