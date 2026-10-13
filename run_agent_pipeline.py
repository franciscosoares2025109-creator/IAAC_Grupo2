import glob
import pandas as pd
from src.data_preprocessing import load_and_clean_data
from src.agent_system import TrojanDetectorAgentSystem

def main():
    print("=== A Iniciar Sistema Multiagente de Deteção de Trojans ===")

    # 1. Localizar o melhor modelo treinado
    model_files = glob.glob("models/trojan_best_model_*.pkl")
    if not model_files:
        print("Erro: Nenhum modelo encontrado na pasta 'models/'. Executa primeiro 'python train_trojan.py'.")
        return
    
    best_model_path = model_files[0]
    scaler_path = "models/trojan_scaler.pkl"
    encoder_path = "models/trojan_label_encoder.pkl"

    print(f"Modelo Carregado: {best_model_path}")

    # 2. Carregar o detetor multiagente
    detector_system = TrojanDetectorAgentSystem(best_model_path, scaler_path, encoder_path)

    # 3. Simular a captura de um fluxo de dados de teste (amostra)
    df_clean = load_and_clean_data("Trojan_Detection.csv")
    sample = df_clean.drop(columns=['Class']).iloc[[0]] # Primeira linha como exemplo

    # 4. Executar a Previsão de ML
    predicted_class, trojan_prob = detector_system.predict_sample(sample)
    
    print("\n--- Resultado do Classificador de ML ---")
    print(f"Classe Prevista : {predicted_class}")
    print(f"Probabilidade   : {trojan_prob * 100:.2f}%")

    # Extrair algumas métricas chave para o agente generativo
    flow_summary = {
        "Flow Duration": sample['Flow Duration'].values[0] if 'Flow Duration' in sample else 'N/A',
        "Total Fwd Packets": sample['Total Fwd Packets'].values[0] if 'Total Fwd Packets' in sample else 'N/A',
        "Protocol": sample['Protocol'].values[0] if 'Protocol' in sample else 'N/A'
    }

    # 5. Gerar Relatório com o Sistema Multiagente
    print("\n--- A Executar Sistema Multiagente (CrewAI) ---")
    report = detector_system.generate_incident_report(predicted_class, trojan_prob, flow_summary)
    
    print("\n================ RELATÓRIO FINAL DO AGENTE ================")
    print(report)

if __name__ == "__main__":
    main()