import os
import pandas as pd
import numpy as np
import ast


def load_and_process_embeddings() -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "../../data/embeddings.csv")

    # Verifique se o arquivo existe
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado.")

    # Carregue e processe o arquivo CSV
    df = pd.read_csv(file_path)
    df["embedding"] = (
        df["embedding"]
        .apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else None)
        .apply(np.array)
    )
    return df
