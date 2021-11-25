class PositionalIndex:
    def __init__(
            self,
            total_frequency: int = 0,
    ):
        self.total_frequency = total_frequency
        self.doc_to_indexes = {}

    @property
    def num_docs_containing(self):
        return len(self.indices.keys())

    @property
    def docs_containing(self):
        return self.doc_to_indexes.keys()

    def update_indexes(self, doc_id, idx):
        self.total_frequency += 1

        if doc_id not in self.doc_to_indexes:
            self.doc_to_indexes.update({doc_id: []})

        self.doc_to_indexes[doc_id].append(idx)
