from typing import List, Dict

import pandas as pd
from hazm import word_tokenize

from models import TokenInfo, Substring
from positional_indexing import get_tokens_info_dict
from preprocessing import Preprocessor


class QueryEngine:
    df: pd.DataFrame = None
    tokens_info: Dict[str, TokenInfo] = None
    preprocessor: Preprocessor = None

    def __init__(self, df, preprocessor, query):
        if self.df is None:
            self.df = df
            self.preprocessor = preprocessor
            self.tokens_info = get_tokens_info_dict(df)

        self.query_tokens = self._preprocess_query(query)

    def _preprocess_query(self, query):
        normalized_query = self.preprocessor.normalize_text(query)
        tokens = word_tokenize(normalized_query)
        tokens = self.preprocessor.remove_stopwords(tokens)
        tokens = self.preprocessor.stem_tokens(tokens)
        return tokens

    def process_query(self):
        if len(self.query_tokens) == 1:
            docs = self._process_single_token_query()
            self._print_single_token_query_results(docs)
        else:
            docs_to_substrings = self._process_multiple_token_query()
            self._print_multiple_token_query_results(docs_to_substrings)

    def _process_single_token_query(self):
        query_token = self.query_tokens[0]
        docs_containing_token = []
        if query_token in self.tokens_info:
            docs_containing_token = list(self.tokens_info.get(query_token).docs_containing)
        return docs_containing_token

    def _print_single_token_query_results(self, docs_with_token: List[int]):
        print('Docs found:\n')

        for doc in docs_with_token:
            title = self.df.iloc[doc]['title'].strip('\n')
            print(f"Doc#{doc} Title: {title[:80]}")
        print('\n')

    def _process_multiple_token_query(self) -> Dict[int, List[Substring]]:
        docs_to_substrings = {}
        all_docs_containing_query_tokens = set()
        for current_token in self.query_tokens:
            current_token_info = self.tokens_info.get(current_token)
            if current_token_info is None:
                continue

            docs_with_current_token = current_token_info.docs_containing
            all_docs_containing_query_tokens.update(set(docs_with_current_token))

        for doc_id in all_docs_containing_query_tokens:
            substrings = self._get_doc_substrings_for_query(doc_id)
            docs_to_substrings.update({
                doc_id: substrings
            })

        return docs_to_substrings

    def _print_multiple_token_query_results(self, docs_to_substrings: Dict[int, List[Substring]]):
        docs = sorted(docs_to_substrings, key=lambda doc: docs_to_substrings[doc][0].score, reverse=True)
        print('Docs found (ranked):\n')
        for doc in docs:
            title = self.df.iloc[doc]['title'].strip('\n')
            print(f"Doc#{doc} Title: {title[:80]}...  Score: {docs_to_substrings[doc][0].score}")
        print('\n')

    def _get_doc_substrings_for_query(self, doc_id: int) -> List[Substring]:
        query_len = len(self.query_tokens)

        prev_max_substrings = []
        prev_max_score = 0

        for starting_idx in range(query_len):
            max_possible_score = query_len - starting_idx
            if max_possible_score < prev_max_score:
                break

            subquery = self.query_tokens[starting_idx:]
            substrings = self._get_longest_substrings_for_subquery(subquery, doc_id)
            if not substrings:
                continue
            substrings_score = substrings[0].score

            if substrings_score > prev_max_score:
                prev_max_substrings = substrings
                prev_max_score = substrings_score

            if substrings_score == max_possible_score:
                break

        return prev_max_substrings

    def _get_longest_substrings_for_subquery(self, subquery_tokens: List[str], doc_id: int):
        first_token = subquery_tokens[0]
        first_token_info = self.tokens_info.get(first_token)
        if first_token_info is None:
            return []

        first_token_indexes = first_token_info.get_indexes(doc_id)
        found_substrings = [Substring(idx) for idx in first_token_indexes]

        for i in range(1, len(subquery_tokens)):
            next_token = subquery_tokens[i]
            next_token_info = self.tokens_info.get(next_token)
            if next_token_info is None:
                break

            next_token_indexes = next_token_info.get_indexes(doc_id)

            found_substrings_with_next_token = self._merge_substrings_with_next_token(
                found_substrings,
                next_token_indexes
            )

            if not found_substrings_with_next_token:
                break

            found_substrings = found_substrings_with_next_token

        return found_substrings

    @staticmethod
    def _merge_substrings_with_next_token(substrings: List[Substring], next_token_indexes: List[int]):
        substrings_with_next_token = []
        for substring in substrings:
            last_index = substring.last_token_index
            if (last_index + 1) in next_token_indexes:
                substring.add_next_token_index()
                substrings_with_next_token.append(substring)

        return substrings_with_next_token
