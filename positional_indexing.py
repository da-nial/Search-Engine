from typing import Dict

import pandas as pd

from models import PositionalIndex


def get_token_to_positional_indexes_dict(df: pd.DataFrame) -> Dict[str, PositionalIndex]:
    token_to_pos_idx = {}
    for doc_id, row in df.iterrows():
        for idx, token in enumerate(row['tokens']):
            if token not in token_to_pos_idx:
                token_to_pos_idx.update({token: PositionalIndex()})

            pos_idx = token_to_pos_idx[token]
            pos_idx.update_indexes(doc_id, idx)

    return token_to_pos_idx
