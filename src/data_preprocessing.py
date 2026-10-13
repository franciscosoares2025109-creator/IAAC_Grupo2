import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_clean_data(filepath: str):
    """
    Carrega o dataset, limpa espaços nos nomes das colunas,
    elimina identificadores (data leakage) e colunas com variância zero.
    """
    df = pd.read_csv(filepath)
    
    # 1. Limpar espaços nos nomes das colunas
    df.columns = df.columns.str.strip()
    
    # 2. Identificadores/Metadados a remover para evitar Data Leakage
    cols_to_drop = [
        'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 
        'Destination IP', 'Destination Port', 'Timestamp'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 3. Remover colunas constantes (variância zero)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_var_cols = [col for col in numeric_cols if df[col].std() == 0]
    if zero_var_cols:
        df = df.drop(columns=zero_var_cols)
        
    return df

def prepare_pipeline_data(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Prepara e divide os dados para o treino e teste do modelo.
    """
    df = load_and_clean_data(filepath)
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Codificar a classe alvo (Benign -> 0, Trojan -> 1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Divisão Treino / Teste com estratificação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Normalização/Escalamento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler, label_encoder