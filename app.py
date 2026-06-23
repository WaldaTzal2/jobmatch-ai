import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da Página
st.set_page_config(page_title="JobMatch AI", page_icon="🤖", layout="wide")

# ==============================================================================
# 🛠️ MOTOR DE INTELIGÊNCIA ARTIFICIAL (AUTO-TREINAMENTO)
# ==============================================================================


@st.cache_resource
def inicializar_ia():
    diretorio = os.path.dirname(os.path.abspath(__file__))
    caminho_csv = os.path.join(diretorio, 'base_vagas.csv')

    if not os.path.exists(caminho_csv):
        return None, None, None, None, None, False, "Arquivo base_vagas.csv não encontrado no GitHub!"

    # Carrega a base de dados
    df = pd.read_csv(caminho_csv)

    # --- TREINAMENTO DE EMERGÊNCIA (Evita erros de versão .pkl) ---
    # 1. Vetorização (NLP)
    vec_vaga = TfidfVectorizer(stop_words='english', max_features=1000)
    X_vagas = vec_vaga.fit_transform(df['descricao_vaga'].fillna(''))

    # 2. Modelo de Regressão (Salário)
    # Treinamos na hora para garantir compatibilidade 100%
    reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    reg.fit(X_vagas, df['salario_real'].fillna(df['salario_real'].mean()))

    # 3. Modelo de Classificação (Fit/No Fit)
    # Criamos um rotulador rápido baseado na descrição
    vec_cls = TfidfVectorizer(stop_words='english', max_features=1000)
    # Simulamos o treinamento de classificação para o ambiente de demo
    X_cls = vec_cls.fit_transform(df['descricao_vaga'])
    # Criamos labels sintéticos para a demonstração funcional
    y_cls = [1 if i % 2 == 0 else 0 for i in range(len(df))]
    cls = SVC(probability=True, kernel='linear', random_state=42)
    cls.fit(X_cls, y_cls)

    return cls, reg, vec_cls, vec_vaga, df, True, ""


# Inicializa o sistema
modelo_cls, modelo_reg, vectorizer, vectorizer_vaga, df_vagas, sucesso, erro_msg = inicializar_ia()

# ==============================================================================
# 🎨 INTERFACE VISUAL (FRONTEND)
# ==============================================================================
st.title("🤖 JobMatch AI")
st.subheader("Sistema Inteligente de Recomendação: Currículos & Vagas")
st.markdown("---")

if not sucesso:
    st.error(f"❌ Erro Crítico: {erro_msg}")
else:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📝 Perfil do Candidato")
        input_texto = st.text_area(
            "Cole seu currículo aqui:", height=300, placeholder="Ex: Especialista em Python e SQL...")
        botao = st.button("🚀 Analisar Compatibilidade")

    with col2:
        st.header("🎯 Top-5 Vagas Recomendadas")
        if botao and input_texto.strip() != "":
            # Processamento em Tempo Real
            v_input = vectorizer_vaga.transform([input_texto])

            # Ranking por Similaridade
            X_vagas = vectorizer_vaga.transform(
                df_vagas['descricao_vaga'].fillna(''))
            sims = cosine_similarity(v_input, X_vagas).flatten()
            df_vagas['score_match'] = sims * 100

            # Top 5
            top5 = df_vagas.sort_values(
                by='score_match', ascending=False).head(5)

            for _, r in top5.iterrows():
                # Previsão de Salário
                v_vaga_individual = vectorizer_vaga.transform(
                    [str(r['descricao_vaga'])])
                sal_pred = modelo_reg.predict(v_vaga_individual)[0]

                with st.expander(f"💼 {r['titulo_vaga']} (Match: {r['score_match']:.1f}%)"):
                    st.write(f"**Estima de Salário:** R$ {sal_pred:,.2f}")
                    st.write(f"**Descrição:** {r['descricao_vaga']}")
                    st.info(
                        f"💡 **Dica:** O modelo sugere focar nas skills: {r['skills_exigidas']}")
        else:
            st.info(
                "Insira seu currículo e clique no botão para ver as recomendações da IA.")

st.markdown("---")
st.caption("Projeto Final E2E Machine Learning - JobMatch AI 2026")
