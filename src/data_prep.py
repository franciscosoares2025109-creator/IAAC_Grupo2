import numpy as np
import pandas as pd

TARGET = "Class"
ID_COLS = ["Unnamed: 0", "Flow ID", "Source IP", "Destination IP", "Timestamp"]
MIXED_TYPE_COLS = ["Packet Length Std", "CWE Flag Count"]


def load_raw(path):
    """Lê o CSV original. low_memory=False evita tipos misturados por blocos."""
    return pd.read_csv(path, low_memory=False)


def clean_dataset(df):
    """Limpa o dataset de keylogger. Devolve um novo DataFrame, sem alterar o original."""
    df = df.copy()

    # Nomes de colunas com espaços no início/fim causam KeyError difíceis de perceber
    df.columns = df.columns.str.strip()

    # Colunas com lixo (ex.: "SCAREWARE") passam a NaN para serem removidas depois
    for col in MIXED_TYPE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Infinitos (divisões por zero) partem vários modelos; este dataset não tem,
    # mas fica protegido caso o ficheiro seja atualizado
    df = df.replace([np.inf, -np.inf], np.nan)

    # Linhas quase vazias ou desalinhadas
    df = df.dropna()

    # Identificadores descrevem a máquina ou o momento, não o comportamento
    df = df.drop(columns=ID_COLS)

    # Duplicados inflacionam a accuracy se caírem no treino e no teste
    df = df.drop_duplicates().reset_index(drop=True)
    return df
