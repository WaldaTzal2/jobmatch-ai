from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sys
import sklearn

# --- PATCH DE COMPATIBILIDADE (ADICIONE ISSO AQUI) ---
import sklearn.ensemble._gb_losses as _gb_losses
sys.modules['sklearn.ensemble._loss'] = _gb_losses
# -----------------------------------------------------

# Configuração da página do Streamlit
st.set_page_config(
    page_title="JobMatch AI - Recrutamento Inteligente",
    page_icon="🤖",
    layout="wide"
)

# ==============================================================================
# FUNÇÃO CORRIGIDA PARA CARREGAR OS ARQUIVOS DA IA
# ==============================================================================


@st.cache_resource
def carregar_componentes():
    # Descobre a pasta onde o script app.py está a rodar na nuvem
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))

    # Mapeia o caminho exato para cada arquivo na raiz do repositório
    modelo_cls = joblib.load(os.path.join(
        diretorio_atual, 'modelo_classificacao.pkl'))
    modelo_reg = joblib.load(os.path.join(
        diretorio_atual, 'modelo_regressao.pkl'))
    vec = joblib.load(os.path.join(diretorio_atual, 'vectorizer.pkl'))
    vec_vaga = joblib.load(os.path.join(
        diretorio_atual, 'vectorizer_vaga.pkl'))
    df_vagas = pd.read_csv(os.path.join(diretorio_atual, 'base_vagas.csv'))

    return modelo_cls, modelo_reg, vec, vec_vaga, df_vagas


try:
    modelo_classificacao, modelo_regressao, vectorizer, vectorizer_vaga, df_vagas = carregar_componentes()
    arquivos_carregados = True
except Exception as e:
    arquivos_carregados = False
    erro_detalhado = str(e)

# ==============================================================================
# INTERFACE DO USUÁRIO (FRONTEND)
# ==============================================================================
st.title("🤖 JobMatch AI")
st.subheader("Análise Inteligente de Compatibilidade de Currículos & Vagas")
st.markdown("---")

if not arquivos_carregados:
    st.error("⚠️ ERRO: Arquivos de IA não encontrados na mesma pasta do script!")
    st.info(f"**Detalhe do erro técnico:** {erro_detalhado}")
else:
    # Divisão da tela em duas colunas
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📝 Perfil do Candidato")
        perfil_usuario = st.text_area(
            "Cole seu Currículo ou Resumo Profissional aqui:",
            height=300,
            placeholder="Exemplo: Sou especialista em dados, programo em Python e crio modelos de Machine Learning..."
        )

        botao_analisar = st.button("🚀 Procurar Vagas Ideais")

    with col2:
        st.header("🎯 Vagas Recomendadas e Análise")

        if botao_analisar and perfil_usuario.strip() != "":
            resultados = []

            for idx, linha in df_vagas.iterrows():
                # Preparação do texto combinado para o classificador
                texto_combinado = perfil_usuario + \
                    " [SEP] " + str(linha['descricao_vaga'])
                vetor_comb = vectorizer.transform([texto_combinado])

                classificacao = modelo_classificacao.predict(vetor_comb)[0]
                probabilidade = modelo_classificacao.predict_proba(vetor_comb)[
                    0][1]

                # Cálculo de Similaridade por Cosseno
                vetor_usuario_sim = vectorizer_vaga.transform([perfil_usuario])
                vetor_vaga_sim = vectorizer_vaga.transform(
                    [str(linha['descricao_vaga'])])
                sim_cosseno = cosine_similarity(
                    vetor_usuario_sim, vetor_vaga_sim)[0][0]

                # Estimativa Salarial usando o Regressor
                vetor_vaga_reg = vectorizer_vaga.transform(
                    [str(linha['descricao_vaga'])])
                salario_previsto = modelo_regressao.predict(vetor_vaga_reg)[0]

                # Análise de Habilidades Faltantes
                skills_vaga = [s.strip().lower()
                               for s in str(linha['skills_exigidas']).split(',')]
                skills_faltantes = [
                    skill for skill in skills_vaga if skill not in perfil_usuario.lower()]
                skills_compativeis = [
                    skill for skill in skills_vaga if skill in perfil_usuario.lower()]

                resultados.append({
                    'titulo': linha['titulo_vaga'],
                    'descricao': linha['descricao_vaga'],
                    'fit_original': classificacao,
                    'score': probabilidade,
                    'similaridade': sim_cosseno,
                    'salario': salario_previsto,
                    'skills_faltantes': skills_faltantes,
                    'skills_compativeis': skills_compativeis
                })

            df_resultados = pd.DataFrame(resultados)
            df_top5 = df_resultados.sort_values(
                by='score', ascending=False).head(5)

            for idx, vaga in df_top5.iterrows():
                if vaga['fit_original'] == 1:
                    status_tag = "🟢 **FIT CONFIRMADO**"
                else:
                    status_tag = "🔴 **NO FIT**"

                with st.expander(f"💼 {vaga['titulo']} — (Score: {vaga['score']*100:.1f}%)"):
                    st.write(f"**Status da Classificação:** {status_tag}")
                    st.write(
                        f"**Estimativa da Faixa Salarial:** R$ {vaga['salario']:,.2f}")
                    st.write(f"**Descrição da Posição:** {vaga['descricao']}")

                    st.markdown("---")
                    st.subheader("📊 Diagnóstico de Skills")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.success(
                            f"✔️ **Skills Alinhadas:** {', '.join(vaga['skills_compativeis']) if vaga['skills_compativeis'] else 'Nenhuma identificada'}")
                    with c2:
                        st.error(
                            f"❌ **Skills Faltantes:** {', '.join(vaga['skills_faltantes']) if vaga['skills_faltantes'] else 'Nenhuma! Perfil Completo!'}")

                    if vaga['skills_faltantes']:
                        st.info(
                            f"💡 **Sugestão de Desenvolvimento:** Para aumentar suas chances, foque em estudar e adicionar projetos práticos voltados para: *{', '.join(vaga['skills_faltantes'])}*.")
        else:
            st.info("💡 Insira o seu perfil ou currículo na barra lateral esquerda e clique em 'Procurar Vagas Ideais' para executar o motor de Machine Learning.")
