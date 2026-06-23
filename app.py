# ==============================================================================
# 🛡️ REMENDO UNIVERSAL DE COMPATIBILIDADE (DEVE SER A PRIMEIRA COISA DO ARQUIVO)
# ==============================================================================
import sys
import sklearn.ensemble

# Esse bloco cria um atalho para o modelo encontrar os cálculos de perda (_loss)
# que mudaram de lugar nas versões mais recentes do scikit-learn.
try:
    import sklearn.ensemble._gb_losses as _gb_losses
    sys.modules['sklearn.ensemble._loss'] = _gb_losses
except ImportError:
    try:
        import sklearn._loss as _loss
        sys.modules['sklearn.ensemble._loss'] = _loss
    except:
        pass

# ==============================================================================
# 📦 IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da página
st.set_page_config(page_title="JobMatch AI", page_icon="🤖", layout="wide")

# ==============================================================================
# 🧠 CARREGAMENTO DOS MODELOS
# ==============================================================================


@st.cache_resource
def carregar_componentes():
    diretorio = os.path.dirname(os.path.abspath(__file__))
    try:
        modelo_cls = joblib.load(os.path.join(
            diretorio, 'modelo_classificacao.pkl'))
        modelo_reg = joblib.load(os.path.join(
            diretorio, 'modelo_regressao.pkl'))
        vec = joblib.load(os.path.join(diretorio, 'vectorizer.pkl'))
        vec_vaga = joblib.load(os.path.join(diretorio, 'vectorizer_vaga.pkl'))
        df_vagas = pd.read_csv(os.path.join(diretorio, 'base_vagas.csv'))
        return modelo_cls, modelo_reg, vec, vec_vaga, df_vagas, True, ""
    except Exception as e:
        return None, None, None, None, None, False, str(e)


m_cls, m_reg, v, v_vaga, df, sucesso, erro = carregar_componentes()

# ==============================================================================
# 🎨 INTERFACE (FRONTEND)
# ==============================================================================
st.title("🤖 JobMatch AI")
st.subheader("Sistema Inteligente de Recomendação de Vagas e Candidatos")
st.markdown("---")

if not sucesso:
    st.error("⚠️ Erro ao carregar os arquivos de Inteligência Artificial.")
    st.info(f"**Detalhe técnico para suporte:** {erro}")
else:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("📝 Seu Perfil")
        txt = st.text_area("Cole seu currículo aqui:", height=300)
        btn = st.button("🚀 Analisar Vagas")

    with col2:
        st.header("🎯 Top-5 Vagas Recomendadas")
        if btn and txt.strip() != "":
            res = []
            for _, linha in df.iterrows():
                # Classificação
                t_comb = txt + " [SEP] " + str(linha['descricao_vaga'])
                v_comb = v.transform([t_comb])
                prob = m_cls.predict_proba(v_comb)[0][1]

                # Regressão e Similaridade
                v_vaga_sim = v_vaga.transform([str(linha['descricao_vaga'])])
                sal = m_reg.predict(v_vaga_sim)[0]

                res.append({
                    'titulo': linha['titulo_vaga'],
                    'score': prob * 100,
                    'salario': sal,
                    'desc': linha['descricao_vaga']
                })

            df_res = pd.DataFrame(res).sort_values(
                by='score', ascending=False).head(5)
            for _, r in df_res.iterrows():
                with st.expander(f"💼 {r['titulo']} — Match: {r['score']:.1f}%"):
                    st.write(f"**Salário Estimado:** R$ {r['salario']:,.2f}")
                    st.write(f"**Descrição:** {r['desc']}")
        elif btn:
            st.warning("Insira o texto do currículo.")
