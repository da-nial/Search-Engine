import logging
from collections import Counter
from typing import List

import pandas as pd
from hazm import Normalizer, Stemmer, word_tokenize


class Preprocessor:
    df: pd.DataFrame = None
    stopwords: List[str] = None

    def __init__(self, df):
        if self.df is None:
            self.df = df
        self.set_stopwords()
        self.stemmer = Stemmer()
        self.normalizer = Normalizer()

    def set_stopwords(self):
        if self.stopwords is not None:
            return self.stopwords

        num_stopwords = 30
        counts = self.count_words()

        common_words_counts = counts.most_common(num_stopwords)
        logging.info(f'Most common words with their counts: {common_words_counts}')

        stopwords = [word for word, count in common_words_counts]
        self.stopwords = stopwords

    def count_words(self):
        counts = Counter()

        for _, row in self.df.iterrows():
            words = row['content'].split()
            counts.update(token for token in words)

        return counts

    def preprocess_df(self):
        self.normalize()
        self.tokenize()
        self.remove_stopwords_from_df()
        self.stem()

    def tokenize(self):
        self.df['tokens'] = self.df['content'].apply(lambda text: word_tokenize(text))
        return self.df

    def normalize(self):
        self.df['content'] = self.df['content'].apply(lambda text: self.normalize_text(text))
        return self.df

    def normalize_text(self, text):
        normalized_text = self.normalizer.normalize(text)
        return normalized_text

    def remove_stopwords_from_df(self):
        self.df['tokens'] = self.df['tokens'].apply(lambda tokens: self.remove_stopwords(tokens))

    def stem(self):
        self.df['tokens'] = self.df['tokens'].apply(lambda tokens: self.stem_tokens(tokens))

    def stem_tokens(self, tokens: List[str]):
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return stemmed_tokens

    def remove_stopwords(self, tokens: List[str]):
        filtered_tokens = [token for token in tokens if token not in self.stopwords]
        return filtered_tokens
