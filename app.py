import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da página do Streamlit
st.set_page_config(
    page_title="JobMatch AI - Recrutamento Inteligente",
    page_icon="🤖",
    layout="wide"
)

# ==============================================================================
# FUNÇÃO PARA CARREGAR OS ARQUIVOS DA IA
# ==============================================================================


@st.cache_resource
def carregar_componentes():
    # Carrega os cérebros salvos no Colab
    modelo_cls = joblib.load('modelo_classificacao.pkl')
    modelo_reg = joblib.load('modelo_regressao.pkl')
    vec = joblib.load('vectorizer.pkl')
    vec_vaga = joblib.load('vectorizer_vaga.pkl')
    df_vagas = pd.read_csv('base_vagas.csv')
    return modelo_cls, modelo_reg, vec, vec_vaga, df_vagas


try:
    modelo_classificacao, modelo_regressao, vectorizer, vectorizer_vaga, df_vagas = carregar_componentes()
    arquivos_carregados = True
except Exception as e:
    arquivos_carregados = False

# ==============================================================================
# INTERFACE DO USUÁRIO (FRONTEND)
# ==============================================================================
st.title("🤖 JobMatch AI")
st.subheader("Análise Inteligente de Compatibilidade de Currículos & Vagas")
st.markdown("---")

if not arquivos_carregados:
    st.error("⚠️ ERRO: Arquivos de IA não encontrados na mesma pasta do script! Certifique-se de colocar os arquivos '.pkl' e '.csv' gerados no Colab aqui.")
else:
    # Divisão da tela em duas colunas [cite: 19, 20]
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📝 Perfil do Candidato")
        # Campo para o usuário colar o texto do currículo [cite: 19]
        perfil_usuario = st.text_area(
            "Cole seu Currículo ou Resumo Profissional aqui:",
            height=300,
            placeholder="Exemplo: Sou especialista em dados, programo em Python e crio modelos de Machine Learning..."
        )

        botao_analisar = st.button("🚀 Procurar Vagas Ideais")

    with col2:
        st.header("🎯 Vagas Recomendadas e Análise")

        if botao_analisar and perfil_usuario.strip() != "":
            # Lista para guardar os resultados calculados para cada vaga da base
            resultados = []

            # Varrendo a base de vagas para calcular afinidade uma por uma [cite: 33]
            for idx, linha in df_vagas.iterrows():
                # 1. Preparação do texto combinado para o classificador
                texto_combinado = perfil_usuario + \
                    " [SEP] " + linha['descricao_vaga']
                vetor_comb = vectorizer.transform([texto_combinado])

                # Predição de Fit (Sim/Não) e probabilidade do Score [cite: 32, 34, 35]
                classificacao = modelo_classificacao.predict(vetor_comb)[0]
                probabilidade = modelo_classificacao.predict_proba(
                    vetor_comb)[0][1]  # Pega a chance de ser "Fit"

                # 2. Cálculo de Similaridade por Cosseno puro (Para refinar o Ranking das Top-5) [cite: 41, 49, 50]
                vetor_usuario_sim = vectorizer_vaga.transform([perfil_usuario])
                vetor_vaga_sim = vectorizer_vaga.transform(
                    [linha['descricao_vaga']])
                sim_cosseno = cosine_similarity(
                    vetor_usuario_sim, vetor_vaga_sim)[0][0]

                # 3. Estimativa Salarial usando o Regressor treinado [cite: 42, 51]
                vetor_vaga_reg = vectorizer_vaga.transform(
                    [linha['descricao_vaga']])
                salario_previsto = modelo_regressao.predict(vetor_vaga_reg)[0]

                # 4. Análise de Habilidades Faltantes (Opcional - bônus do projeto) [cite: 36, 38]
                skills_vaga = [s.strip().lower()
                               for s in linha['skills_exigidas'].split(',')]
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

            # Transformando em Dataframe e ordenando pelo Score de maior aderência (Ranking Top-5) [cite: 15, 40, 41]
            df_resultados = pd.DataFrame(resultados)
            df_top5 = df_resultados.sort_values(
                by='score', ascending=False).head(5)

            # Exibindo os resultados na tela de forma visual e amigável
            for idx, vaga in df_top5.iterrows():
                # Define a cor do selo de acordo com o resultado [cite: 35]
                if vaga['fit_original'] == 1:
                    status_tag = "🟢 **FIT CONFIRMADO**"
                else:
                    status_tag = "🔴 **NO FIT**"

                # Cria um quadro expansível para cada vaga
                with st.expander(f"💼 {vaga['titulo']} — (Score: {vaga['score']*100:.1f}%)"):
                    st.write(f"**Status da Classificação:** {status_tag}")
                    st.write(
                        f"**Estimativa da Faixa Salarial:** R$ {vaga['salario']:,.2f}")[cite: 42]
                    st.write(f"**Descrição da Posição:** {vaga['descricao']}")

                    # Seção de Análise de Competências (Opcional do roteiro) [cite: 36]
                    st.markdown("---")
                    st.subheader("📊 Diagnóstico de Skills")[cite: 15]

                    c1, c2 = st.columns(2)
                    with c1:
                        st.success(
                            f"✔️ **Skills Alinhadas:** {', '.join(vaga['skills_compativeis']) if vaga['skills_compativeis'] else 'Nenhuma identificada'}")[cite: 37]
                    with c2:
                        st.error(
                            f"❌ **Skills Faltantes:** {', '.join(vaga['skills_faltantes']) if vaga['skills_faltantes'] else 'Nenhuma! Perfil Completo!'}")[cite: 38]

                    # Sugestão de desenvolvimento automático [cite: 39]
                    if vaga['skills_faltantes']:
                        st.info(
                            f"💡 **Sugestão de Desenvolvimento:** Para aumentar suas chances, foque em estudar e adicionar projetos práticos voltados para: *{', '.join(vaga['skills_faltantes'])}*.")[cite: 39]
        else:
            st.info("💡 Insira o seu perfil ou currículo na barra lateral esquerda e clique em 'Procurar Vagas Ideais' para executar o motor de Machine Learning.")
