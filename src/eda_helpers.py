import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_data_summary(df: pd.DataFrame):
    """Retorna um resumo estatístico detalhado: média, desvio padrão, skewness e kurtosis."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
    stats['skewness'] = df[numeric_cols].skew()
    stats['kurtosis'] = df[numeric_cols].kurtosis()
    return stats

def plot_class_distribution(df: pd.DataFrame, target_col: str = 'Class'):
    """Gera o gráfico da distribuição das classes (Trojan vs Benign)."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target_col, palette='viridis')
    plt.title('Distribuição da Classe Alvo (Trojan vs Benign)')
    plt.xlabel('Classe')
    plt.ylabel('Contagem de Fluxos')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_top_correlations(df: pd.DataFrame, target_col: str = 'Class', top_n: int = 10):
    """Calcula e exibe as features numericamente mais correlacionadas com a variável alvo."""
    temp_df = df.copy()
    temp_df['target_encoded'] = (temp_df[target_col] == 'Trojan').astype(int)
    
    numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
    corrs = temp_df[numeric_cols].corrwith(temp_df['target_encoded']).abs().sort_values(ascending=False)
    corrs = corrs.drop('target_encoded').head(top_n)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=corrs.values, y=corrs.index, palette='magma')
    plt.title(f'Top {top_n} Features Mais Correlacionadas com Trojan')
    plt.xlabel('Correlação Absoluta')
    plt.tight_layout()
    plt.show()