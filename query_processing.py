from typing import List, Dict

import pandas as pd
from hazm import word_tokenize, Normalizer, Stemmer

from models import PositionalIndex


def preprocess_query(query):
    query = Normalizer().normalize(query)
    query = Stemmer().stem(query)
    # TODO should we remove stopwords?
    tokens = word_tokenize(query)
    return tokens


def process_query(query: str, token_to_pos_idx: Dict[str, PositionalIndex], df: pd.DataFrame):
    query_tokens = preprocess_query(query)

    if len(query_tokens) == 1:
        result = process_single_token_query(query_tokens[0], token_to_pos_idx)
        print_single_token_query_results(result, df)
    else:
        result = process_multiple_token_query(query_tokens, token_to_pos_idx)
        print_multiple_token_query_results(result, df)


def process_single_token_query(query_token: str, token_to_pos_idx: Dict[str, PositionalIndex]):
    docs_with_token = []
    if query_token in token_to_pos_idx:
        docs_with_token = list(token_to_pos_idx.get(query_token).docs_containing)
    return docs_with_token


def print_single_token_query_results(docs_with_token: List[int], df: pd.DataFrame):
    print('Docs found:\n')

    for doc in docs_with_token:
        title = df.iloc[doc]['title'].strip('\n')
        print(f"Doc#{doc} Title: {title[:80]}")
    print('\n')


def process_multiple_token_query(query_tokens: List[str], token_to_pos_idx: Dict[str, PositionalIndex]):
    docs_to_score = {}
    for current_token in query_tokens:
        docs_with_current_token = token_to_pos_idx.get(current_token).doc_to_indexes.keys()

        update_result_docs(docs_with_current_token, docs_to_score)

    return docs_to_score


def update_result_docs(docs_with_current_token: List[int], docs_to_score: Dict[int, int]):
    for doc in docs_with_current_token:
        docs_to_score[doc] = docs_to_score.get(doc, 0) + 1


def print_multiple_token_query_results(docs_to_score: Dict[int, int], df: pd.DataFrame):
    docs = sorted(docs_to_score, key=lambda doc: docs_to_score[doc], reverse=True)
    print('Docs found (ranked):\n')
    for doc in docs:
        title = df.iloc[doc]['title'].strip('\n')
        print(f"Doc#{doc} Title: {title[:80]}...  NumberOfCommonTokens: {docs_to_score[doc]}")
    print('\n')


# ================================= Not Used =================================

def does_all_query_tokens_exist(tokens: List[str], positional_indexes: Dict[str, PositionalIndex]):
    return all([token in positional_indexes for token in tokens])


def get_query_tokens_processing_order(tokens: List[str], positional_indexes: Dict[str, PositionalIndex]):
    tokens_to_num_docs = {}
    for token in tokens:
        num_docs_containing_token = 0
        if token in positional_indexes:
            num_docs_containing_token = positional_indexes[token].num_docs_containing

        tokens_to_num_docs.update(
            {token: num_docs_containing_token}
        )

    tokens_order = sorted(tokens_to_num_docs, key=lambda token: tokens_to_num_docs[token])
    return tokens_order


def docs_list_union(docs_1: List[str], docs_2: List[str]):
    return list(set(docs_1).union(set(docs_2)))


def display_single_token_query_results(docs_with_token: List[int], df: pd.DataFrame):
    print('Docs found:\n')
    display_df = df.loc[df['doc_id'].isin(docs_with_token)][df.columns.intersection(['doc_id', 'title'])]
    print(display_df.to_string(index=False))
    print('\n')


def display_multiple_token_query_results(docs_to_score: Dict[int, int], df: pd.DataFrame):
    docs = sorted(docs_to_score, key=lambda doc: docs_to_score[doc], reverse=True)
    print('Docs found (ranked):\n')
    display_df = df.iloc[docs][df.columns.intersection(['title'])].copy()
    display_df['number_of_common_tokens'] = df['doc'].apply(lambda doc: docs_to_score[doc])
    print(display_df.to_string(index=False))
    print('\n')
