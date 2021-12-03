from typing import Dict

import pandas as pd

from models import TokenInfo


def get_tokens_info_dict(df: pd.DataFrame) -> Dict[str, TokenInfo]:
    tokens_info = {}
    for doc_id, row in df.iterrows():
        for idx, token in enumerate(row['tokens']):
            if token not in tokens_info:
                tokens_info.update({token: TokenInfo()})

            token_info = tokens_info[token]
            token_info.update_indexes(doc_id, idx)

    return tokens_info
