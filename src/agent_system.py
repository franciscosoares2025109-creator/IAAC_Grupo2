import joblib
import pandas as pd
import numpy as np
from crewai import Agent, Task, Crew, Process

class TrojanDetectorAgentSystem:
    def __init__(self, model_path: str, scaler_path: str, encoder_path: str):
        """Carrega o modelo de classificação e os transformadores guardados."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)

    def predict_sample(self, sample_df: pd.DataFrame):
        """
        Executa a inferência do modelo de ML num fluxo de rede.
        Devolve a classe prevista e a probabilidade de ser Trojan.
        """
        # Garantir escalamento correto
        scaled_data = self.scaler.transform(sample_df)
        
        prediction_numeric = self.model.predict(scaled_data)[0]
        probabilities = self.model.predict_proba(scaled_data)[0]
        
        predicted_class = self.encoder.inverse_transform([prediction_numeric])[0]
        trojan_prob = probabilities[1] if len(probabilities) > 1 else 1.0
        
        return predicted_class, trojan_prob

    def generate_incident_report(self, predicted_class: str, probability: float, flow_summary: dict):
        """
        Orquestra o sistema multiagente (CrewAI) para gerar um relatório analítico.
        """
        # Agente 1: Analista de Triagem de Segurança
        analyst = Agent(
            role='Analista Sénior de SOC (Cibersegurança)',
            goal='Analisar os alertas gerados pelo modelo de ML e identificar padrões de tráfego de Trojans.',
            backstory='És um perito em análise de tráfego de rede e malware com anos de experiência em SOC.',
            verbose=True,
            allow_delegation=False
        )

        # Agente 2: Especialista em Resposta a Incidentes
        reporter = Agent(
            role='Especialista em Resposta a Incidentes de Cibersegurança',
            goal='Elaborar relatórios técnicos claros e planos de mitigação imediatos.',
            backstory='És responsável por comunicar incidentes de segurança a equipas técnicas e gestores.',
            verbose=True,
            allow_delegation=False
        )

        # Tarefa 1: Triagem e Diagnóstico
        task_analysis = Task(
            description=f"""
            Analisa o seguinte resultado retornado pelo modelo de Machine Learning:
            - Classificação Prevista: {predicted_class}
            - Probabilidade/Confiança de Trojan: {probability * 100:.2f}%
            - Métricas do Fluxo de Rede: {flow_summary}

            Identifica a gravidade da ameaça e quais as métricas de rede que sustentam este diagnóstico.
            """,
            expected_output="Análise técnica sumária do incidente e nível de risco (Baixo, Médio, Crítico).",
            agent=analyst
        )

        # Tarefa 2: Elaboração do Relatório
        task_report = Task(
            description="""
            Com base na análise do Analista de SOC, elabora um Relatório de Incidente em Português com:
            1. Resumo Executivo
            2. Detalhes Técnicos da Ameaça Detetada
            3. Ações Imediatas de Mitigação (ex: bloquear IPs, isolar hosts, rever regras de Firewall)
            """,
            expected_output="Relatório estruturado de Incidente de Cibersegurança em texto Markdown.",
            agent=reporter
        )

        # Criar a Crew (Equipa Multiagente)
        crew = Crew(
            agents=[analyst, reporter],
            tasks=[task_analysis, task_report],
            process=Process.sequential
        )

        return crew.kickoff()