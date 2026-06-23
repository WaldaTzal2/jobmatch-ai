import sys
import os

# ==============================================================================
# 🛠️ PASSO 0: PATCH DE COMPATIBILIDADE (CORREÇÃO DO ERRO _loss)
# ==============================================================================
# Este bloco resolve o erro "No module named '_loss'" que acontece pela
# diferença de versões entre o Google Colab e o Streamlit Cloud.
try:
    import sklearn.ensemble._gb_losses as _gb_losses
    sys.modules['sklearn.ensemble._loss'] = _gb_losses
except:
    pass

# ==============================================================================
# 📦 PASSO 1: IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Configuração visual da página
st.set_page_config(
    page_title="JobMatch AI - Recomendação Inteligente",
    page_icon="🤖",
    layout="wide"
)

# ==============================================================================
# 🧠 PASSO 2: CARREGAMENTO DOS MODELOS DE IA
# ==============================================================================


@st.cache_resource
def carregar_arquivos_ia():
    """Carrega os modelos treinados e a base de vagas do GitHub"""
    # Descobre onde o script está rodando para não errar o caminho dos arquivos
    diretorio = os.path.dirname(os.path.abspath(__file__))

    try:
        # Carrega os arquivos .pkl (Cérebros da IA)
        modelo_cls = joblib.load(os.path.join(
            diretorio, 'modelo_classificacao.pkl'))
        modelo_reg = joblib.load(os.path.join(
            diretorio, 'modelo_regressao.pkl'))
        vec = joblib.load(os.path.join(diretorio, 'vectorizer.pkl'))
        vec_vaga = joblib.load(os.path.join(diretorio, 'vectorizer_vaga.pkl'))

        # Carrega a base de dados das vagas
        df_vagas = pd.read_csv(os.path.join(diretorio, 'base_vagas.csv'))

        return modelo_cls, modelo_reg, vec, vec_vaga, df_vagas, True, ""
    except Exception as e:
        return None, None, None, None, None, False, str(e)


# Executa o carregamento
modelo_classificacao, modelo_regressao, vectorizer, vectorizer_vaga, df_vagas, sucesso, erro_msg = carregar_arquivos_ia()

# ==============================================================================
# 🎨 PASSO 3: INTERFACE VISUAL (FRONTEND)
# ==============================================================================
st.title("🤖 JobMatch AI")
st.subheader("Sistema Inteligente de Recomendação de Vagas e Candidatos")
st.markdown("---")

if not sucesso:
    st.error("⚠️ Erro ao carregar os arquivos de Inteligência Artificial.")
    st.info(f"Detalhe técnico para suporte: {erro_msg}")
else:
    # Criando duas colunas na tela
    col_esq, col_dir = st.columns([1, 2])

    with col_esq:
        st.header("📝 Seu Perfil")
        curriculo_texto = st.text_area(
            "Cole aqui o texto do seu currículo ou resumo profissional:",
            height=350,
            placeholder="Ex: Sou analista de dados com experiência em Python, SQL e Power BI..."
        )
        botao = st.button("🚀 Analisar Compatibilidade")

    with col_dir:
        st.header("🎯 Vagas Recomendadas (Top-5)")

        if botao and curriculo_texto.strip() != "":
            resultados = []

            # Loop para analisar cada vaga da nossa base de dados
            for _, vaga in df_vagas.iterrows():
                # 1. Preparar texto para Classificação (Fit/No Fit)
                texto_entrada = curriculo_texto + \
                    " [SEP] " + str(vaga['descricao_vaga'])
                vetor_entrada = vectorizer.transform([texto_entrada])

                # Predição de Classificação e Score
                fit_binario = modelo_classificacao.predict(vetor_entrada)[0]
                score_match = modelo_classificacao.predict_proba(vetor_entrada)[
                    0][1]

                # 2. Similaridade de Cosseno (Para o Ranking)
                vetor_curr = vectorizer_vaga.transform([curriculo_texto])
                vetor_vaga = vectorizer_vaga.transform(
                    [str(vaga['descricao_vaga'])])
                sim_cosseno = cosine_similarity(vetor_curr, vetor_vaga)[0][0]

                # 3. Estimativa Salarial (Regressão)
                salario_estimado = modelo_regressao.predict(vetor_vaga)[0]

                # 4. Análise de Skills (Gaps)
                skills_vaga = [s.strip().lower()
                               for s in str(vaga['skills_exigidas']).split(',')]
                faltantes = [
                    s for s in skills_vaga if s not in curriculo_texto.lower()]
                alinhadas = [
                    s for s in skills_vaga if s in curriculo_texto.lower()]

                resultados.append({
                    'titulo': vaga['titulo_vaga'],
                    'descricao': vaga['descricao_vaga'],
                    'status': "🟢 FIT" if fit_binario == 1 else "🔴 NO FIT",
                    'score': score_match * 100,
                    'salario': salario_estimado,
                    'skills_faltantes': faltantes,
                    'skills_ok': alinhadas
                })

            # Ordenar e mostrar as Top-5 melhores vagas
            df_res = pd.DataFrame(resultados).sort_values(
                by='score', ascending=False).head(5)

            for _, r in df_res.iterrows():
                with st.expander(f"💼 {r['titulo']} — Score: {r['score']:.1f}%"):
                    st.write(f"**Resultado da Classificação:** {r['status']}")
                    st.write(
                        f"**Estimativa Salarial:** R$ {r['salario']:,.2f}")
                    st.write(f"**Descrição:** {r['descricao']}")

                    st.markdown("---")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.success(
                            f"✅ **Skills Identificadas:** {', '.join(r['skills_ok'])}")
                    with c2:
                        st.error(
                            f"❌ **Skills Faltantes:** {', '.join(r['skills_faltantes'])}")

                    if r['skills_faltantes']:
                        st.warning(
                            f"💡 **Dica de Estudo:** Para esta vaga, sugerimos desenvolver conhecimentos em: *{', '.join(r['skills_faltantes'])}*.")

        elif botao:
            st.warning(
                "⚠️ Por favor, cole o texto do seu currículo antes de analisar.")
        else:
            st.info(
                "Aguardando seu currículo para iniciar o processamento de Machine Learning...")

# Rodapé do projeto
st.markdown("---")
st.caption("JobMatch AI - Projeto Final E2E Machine Learning | 2026")
