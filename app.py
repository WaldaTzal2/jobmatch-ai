import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="JobMatch AI", page_icon="🧠", layout="centered")

st.title("🧠 JobMatch AI")
st.subheader("Análise Inteligência de Compatibilidade de Currículos & Vagas")

# Verificar se os arquivos gerados pelo treinar.py realmente existem na pasta
arquivos_necessarios = [
    'modelo_classificacao.pkl',
    'modelo_regressao.pkl',
    'vectorizer.pkl',
    'vectorizer_vaga.pkl',
    'base_vagas.csv'
]

erros_arquivos = [
    arq for arq in arquivos_necessarios if not os.path.exists(arq)]

if erros_arquivos:
    st.error(
        f"⚠️ Atenção! Os seguintes arquivos não foram encontrados: {erros_arquivos}")
    st.info(
        "Por favor, certifique-se de que o script 'treinar.py' rodou na mesma pasta.")
else:
    try:
        # Carregar os arquivos com segurança
        modelo_cls = joblib.load('modelo_classificacao.pkl')
        modelo_reg = joblib.load('modelo_regressao.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        vectorizer_vaga = joblib.load('vectorizer_vaga.pkl')
        df_vagas = pd.read_csv('base_vagas.csv')

        st.success(
            "✅ Todos os modelos de IA e bases de dados foram carregados com sucesso!")

        # --- ÁREA DO SEU FORMULÁRIO / INTERFACE ---
        st.markdown("---")
        texto_curriculo = st.text_area(
            "Cole o texto do Currículo aqui:", height=200)

        if st.button("Analisar Compatibilidade"):
            if texto_curriculo:
                st.write(
                    "⏳ Processando análise... (Aqui entra a lógica de predição)")
                # Adicione aqui as linhas de transformação de texto e predict que você usava antes
            else:
                st.warning(
                    "Por favor, digite ou cole um currículo para analisar.")

    except Exception as e:
        st.error("🚨 Ocorreu um erro ao carregar os arquivos de inteligência.")
        st.code(str(e))
