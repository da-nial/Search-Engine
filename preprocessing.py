import logging
from collections import Counter
from typing import List

import pandas as pd
from hazm import Normalizer, Stemmer, word_tokenize


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    tokenize(df)
    normalize(df)
    remove_stopwords(df)
    stem(df)

    return df


def tokenize(df: pd.DataFrame):
    df['tokens'] = df['content'].apply(lambda text: word_tokenize(text))
    return df


def normalize(df: pd.DataFrame):
    normalizer = Normalizer()
    df['content'] = df['content'].apply(lambda text: normalizer.normalize(text))
    return df


def count_words(df: pd.DataFrame):
    counts = Counter()

    for _, row in df.iterrows():
        tokens = row['tokens']
        counts.update(token for token in tokens)

    return counts


def remove_stopwords(df: pd.DataFrame):
    num_stopwords = 30
    counts = count_words(df)

    common_words_counts = counts.most_common(num_stopwords)
    stopwords = [word for word, count in common_words_counts]
    logging.info(f'Most common words with their counts: {common_words_counts}')
    df['tokens'] = df['tokens'].apply(lambda tokens: remove_stopword(tokens, stopwords))


def remove_stopword(tokens: List[str], stopwords: List[str]):
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens


def stem(df: pd.DataFrame):
    stemmer = Stemmer()
    df['tokens'] = df['tokens'].apply(lambda tokens: stem_tokens(tokens, stemmer))


def stem_tokens(tokens: List[str], stemmer: Stemmer):
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens
