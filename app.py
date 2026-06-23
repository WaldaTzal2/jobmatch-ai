import sys
import os

# ==============================================================================
# 🛡️ PROTOCOLO DE COMPATIBILIDADE DEFINITIVA (PATCH UNIVERSAL)
# ==============================================================================
# Este bloco DEVE vir antes de qualquer outro import. Ele resolve o erro '_loss'.
try:
    import sklearn.ensemble._gb_losses as _gb_losses
    sys.modules['sklearn.ensemble._loss'] = _gb_losses
except:
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
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da página
st.set_page_config(page_title="JobMatch AI", page_icon="🤖", layout="wide")

# ==============================================================================
# 🧠 CARREGAMENTO BLINDADO
# ==============================================================================


@st.cache_resource
def carregar_tudo():
    # Caminho absoluto para evitar erro de "Arquivo não encontrado"
    pasta = os.path.dirname(os.path.abspath(__file__))
    try:
        m_cls = joblib.load(os.path.join(pasta, 'modelo_classificacao.pkl'))
        m_reg = joblib.load(os.path.join(pasta, 'modelo_regressao.pkl'))
        v = joblib.load(os.path.join(pasta, 'vectorizer.pkl'))
        v_vaga = joblib.load(os.path.join(pasta, 'vectorizer_vaga.pkl'))
        df = pd.read_csv(os.path.join(pasta, 'base_vagas.csv'))
        return m_cls, m_reg, v, v_vaga, df, True, ""
    except Exception as e:
        return None, None, None, None, None, False, str(e)


m_cls, m_reg, v, v_vaga, df, ok, erro = carregar_tudo()

# ==============================================================================
# 🎨 INTERFACE VISUAL
# ==============================================================================
st.title("🤖 JobMatch AI")
st.subheader("Análise Inteligente de Compatibilidade: Currículos & Vagas")
st.markdown("---")

if not ok:
    st.error(f"⚠️ Erro Técnico de Versão: {erro}")
    st.info("💡 Dica: Se o erro persistir, o modelo precisa ser re-salvo no Colab hoje.")
else:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("📝 Seu Currículo")
        texto = st.text_area("Cole seu resumo profissional:", height=300)
        if st.button("🚀 Analisar Agora"):
            if texto.strip() == "":
                st.warning("Por favor, cole seu currículo.")
            else:
                st.session_state['analisar'] = True

    with col2:
        st.header("🎯 Resultados Recomendados")
        if st.session_state.get('analisar'):
            res = []
            for _, linha in df.iterrows():
                # Lógica de Classificação e Score
                t_comb = texto + " [SEP] " + str(linha['descricao_vaga'])
                v_comb = v.transform([t_comb])
                score = m_cls.predict_proba(v_comb)[0][1]

                # Lógica de Salário
                v_vaga_sim = v_vaga.transform([str(linha['descricao_vaga'])])
                sal = m_reg.predict(v_vaga_sim)[0]

                res.append({'titulo': linha['titulo_vaga'], 'score': score *
                           100, 'salario': sal, 'desc': linha['descricao_vaga']})

            df_final = pd.DataFrame(res).sort_values(
                by='score', ascending=False).head(5)
            for _, r in df_final.iterrows():
                with st.expander(f"💼 {r['titulo']} (Match: {r['score']:.1f}%)"):
                    st.write(f"**Salário Estimado:** R$ {r['salario']:,.2f}")
                    st.write(f"**Descrição:** {r['desc']}")
