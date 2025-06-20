# IGATP Dashboard - Streamlit Application

import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from streamlit_keplergl import keplergl_static
from keplergl import KeplerGl


# CONFIG
st.set_page_config(layout="wide")
st.title("IGATP - Índice de Atratividade Turística Percecionada na AMP")

# DATA LOADING
@st.cache_data

def load_data():
    df_index = pd.read_csv("C:/Users/Fernanda Costa/OneDrive - Universidade de Aveiro/Desktop/seminar_project/6_unsupervised_learning/composite_index_with_clusters.csv")
    df_topics = pd.read_csv("C:/Users/Fernanda Costa/OneDrive - Universidade de Aveiro/Desktop/seminar_project/6_unsupervised_learning/ratings_polarity_lda_topics.csv")
    df_freg = pd.read_csv("C:/Users/Fernanda Costa/OneDrive - Universidade de Aveiro/Desktop/seminar_project/8_spatial_analysis/mean_freg_all_by_parish.csv")
    shape_mun = gpd.read_file("C:/Users/Fernanda Costa/OneDrive - Universidade de Aveiro/Desktop/seminar_project/1_data_collection/spatial_data_AMP/shape_CAOP_Conc_AMP.shp").to_crs("EPSG:4326")
    shape_freg = gpd.read_file("C:/Users/Fernanda Costa/OneDrive - Universidade de Aveiro/Desktop/seminar_project/1_data_collection/spatial_data_AMP/shape_CAOP_Freg_AMP.shp").to_crs("EPSG:4326")

    df = pd.merge(df_index, df_topics[["Nome_Local", "Categoria", "dominant_topic"]],
                  on=["Nome_Local", "Categoria"], how="left")
    df = df.dropna(subset=["Latitude_Nova", "Longitude_Nova"])

    gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude_Nova"], df["Latitude_Nova"]),
                                   crs="EPSG:4326")
    return gdf_points, shape_mun, shape_freg, df_freg

# Load
points, mun_shape, freg_shape, df_freg = load_data()

# SECTION: ABOUT
with st.expander("ℹ️ Sobre o Projeto"):
    st.markdown("""
    Este dashboard interativo apresenta o **IGATP - Índice Global de Atratividade Turística Percecionada**, desenvolvido a partir de dados públicos do Google Maps (ratings e comentários). O objetivo é avaliar a atratividade percebida de locais turísticos na **Área Metropolitana do Porto (AMP)** com base em três dimensões principais:

    - **Rating Bayesiano**: qualidade percebida ajustada ao número de avaliações;
    - **Popularidade**: número de reviews associadas ao local;
    - **Sentimento**: polaridade média dos comentários (após tradução e análise de sentimentos).

    Estes sub-índices são combinados com pesos ajustáveis para gerar um valor final de IGATP, entre 0 e 1.

    O dashboard permite:
    - Visualizar os locais georreferenciados e o valor de IGATP;
    - Analisar a interpolação espacial do índice por município e por freguesia;
    - Consultar rankings dos melhores locais, municípios e freguesias;
    - Explorar a evolução temporal do sentimento nos comentários;
    - Filtrar os resultados por **grupo temático** e **perfil turístico** (cluster K-Medoids).

    ---

    ### 🧭 Perfis Turísticos (Clusters K-Medoids)

    Os locais turísticos foram agrupados em 6 perfis distintos, com base numa análise PCA (componentes principais) e clustering:

    - **Mainstream Core** (Cluster 2): locais com popularidade e qualidade médias — perfil dominante.
    - **Flagship Venues** (Cluster 3): locais de destaque com alta popularidade e excelentes avaliações.
    - **Hidden Popular** (Cluster 1): locais com reviews menos positivas, mas muito populares (potencialmente sobrevalorizados).
    - **Underperformers** (Cluster 4): locais com pouca visibilidade e avaliações negativas.
    - **Boutique / Niche** (Cluster 0): locais especializados ou de nicho — avaliações altas, mas com pouca exposição.
    - **Extreme Outlier** (Cluster 5): local com características excecionalmente distintas — potencial caso anómalo.

    ---

    ### 🧠 Tópicos Dominantes (LDA Topic Modeling)

    Os comentários dos utilizadores foram analisados por modelação de tópicos (LDA), resultando em quatro temas principais:

    - **Topic 0 – Lazer ao Ar Livre e Natureza**  
      (“beach”, “walk”, “sand”, “view”, “restaurant”, “quiet”)  
      → experiências costeiras e naturais, valorizando a tranquilidade e paisagem.

    - **Topic 1 – Alojamento e Conforto**  
      (“room”, “clean”, “bed”, “staff”, “breakfast”)  
      → serviços de alojamento, conforto das instalações e apoio ao cliente.

    - **Topic 2 – Visitas Culturais e Património**  
      (“museum”, “history”, “visit”, “portuguese”)  
      → locais históricos, museológicos e com valor cultural.

    - **Topic 3 – Experiência Gastronómica**  
      (“food”, “service”, “restaurant”, “wine”)  
      → avaliação da qualidade da comida, serviço e experiência gastronómica.

    ---
    
    🔍 Para mais detalhes sobre a metodologia utilizada, consulte a documentação do projeto ou explore os mapas interativos disponíveis nas várias abas do dashboard.
    """)


# SIDEBAR - FILTERS
with st.sidebar:
    st.markdown("### 🎛️ **Filtros de Visualização**")

    # Filtro: Grupo Temático
    st.markdown("**Grupo Temático**")
    grupos = st.multiselect(
        "Seleciona os grupos a incluir:",
        options=points["Grupo_Tematico"].dropna().unique().tolist(),
        default=points["Grupo_Tematico"].dropna().unique().tolist(),
        help="Seleciona os tipos de serviços turísticos que queres incluir no índice."
    )

    st.markdown("---")

    # Filtro: Cluster K-Medoids com nomes legíveis
    st.markdown("**Cluster (K-Medoids)**")

    cluster_labels = {
        0: "Boutique / Niche",
        1: "Hidden Popular",
        2: "Mainstream Core",
        3: "Flagship Venues",
        4: "Underperformers",
        5: "Extreme Outlier"
    }

    available_clusters = sorted(points["cluster_k7_pam"].dropna().unique())
    cluster_options = {cluster_labels[c]: c for c in available_clusters if c in cluster_labels}

    selected_cluster_labels = st.multiselect(
        "Seleciona os perfis de local:",
        options=list(cluster_options.keys()),
        default=list(cluster_options.keys()),
        help="Clusters obtidos via PCA e K-Medoids. Representam perfis turísticos distintos."
    )

    selected_clusters = [cluster_options[label] for label in selected_cluster_labels]

    st.markdown("---")

    # Pesos dos subíndices
    st.markdown("### ⚖️ **Pesos dos Sub-índices**")

    w1 = st.slider("Rating Bayesiano", 0.0, 1.0, 1/3, help="Avaliação ajustada à popularidade (Bayesiano).")
    w2 = st.slider("Popularidade", 0.0, 1.0, 1/3, help="Número de reviews do local.")
    w3 = st.slider("Sentimento", 0.0, 1.0, 1/3, help="Polaridade média dos comentários.")

    total = w1 + w2 + w3 or 1
    w1, w2, w3 = w1 / total, w2 / total, w3 / total




# CALCULATE IGATP & FILTER DATA
filtered = points[
    (points["Grupo_Tematico"].isin(grupos)) &
    (points["cluster_k7_pam"].isin(selected_clusters))
].copy()

filtered["IGATP"] = w1 * filtered["Rating_Bayes_norm"] + w2 * filtered["Popularity_norm"] + w3 * filtered["Sentiment_norm"]
filtered_nonull = filtered[filtered["IGATP"].notna()]

# Filtrar apenas pontos dentro dos limites da AMP
mun_shape = mun_shape.to_crs("EPSG:4326")
filtered_nonull = gpd.sjoin(
    filtered_nonull.to_crs("EPSG:4326"),
    mun_shape[["geometry"]],
    how="inner",
    predicate="within"
).drop(columns="index_right", errors="ignore")


# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📍 Mapa Pontual", "🗺️ Mapa por Município", "🏘️ Mapa por Freguesia", "📊 Rankings", "📈 Evolução Temporal"])

# TAB 1 - KEPLER MAP (Pontos)
with tab1:
    st.subheader("Locais com IGATP")

    # Instruções para configurar o pop-up corretamente no Kepler.gl
    st.markdown("""
    ℹ️ **Para ver detalhes completos de cada local ao clicar num ponto no mapa:**

    1. Clique no botão com **a seta cinzenta →** no lado esquerdo do mapa (imagem abaixo).
    2. Aceda ao separador **"Interactions"** no topo do painel lateral.
    3. Em **"Tooltip"**, selecione os seguintes campos para exibir no pop-up:

       - `Nome_Local` → Nome do Local  
       - `Cidade`  
       - `Categoria`  
       - `IGATP` → Índice IGATP (com base nos pesos definidos)  
       - `Locais_Semelhantes_Perto` → Nº de Locais Semelhantes Próximos  
       - `dominant_topic` → Tópico dominante associado ao local

    ⚠️ Estes campos não aparecem por defeito — é necessário ativá-los manualmente.
    """)

    # Mapa com Kepler.gl
    mapa1 = KeplerGl(
        height=600,
        data={
            "IGATP Points": filtered_nonull,
            "AMP Municípios": mun_shape
        },
        config={
            "version": "v1",
            "config": {
                "mapState": {
                    "latitude": 41.15,
                    "longitude": -8.6,
                    "zoom": 8,
                    "bearing": 0,
                    "pitch": 0
                },
                "mapStyle": {
                    "styleType": "muted_night"
                },
                "visState": {
                    "layers": []  # camada por defeito, tooltip configurado manualmente
                }
            }
        }
    )

    keplergl_static(mapa1)
    st.caption("Pontos representam locais turísticos. Cores e atributos podem ser ajustados diretamente na interface do Kepler.gl.")

# TAB 2 - Interpolação por Município
with tab2:
    st.subheader("IGATP médio por Município")

    # Instrução para o utilizador
    st.markdown("""
    ℹ️ **Para visualizar corretamente o gradiente de cores no mapa, siga estes passos no painel do Kepler (ícone da camada):**
    1. Clique no botão com **a seta cinzenta →** no lado esquerdo do mapa (imagem abaixo).
    2. Clique em **"Municípios IGATP"** na lista de camadas.
    3. Em **"Fill Color"**, escolha a variável `IGATPScaled`.
    4. Escolha uma escala de cor (sugestão: `quantile` ou `sequential`).
    5. Ajuste o intervalo se necessário (0 a 1).
    """)

    # Spatial Join entre pontos e municípios para calcular IGATP médio
    filtered_mun = gpd.sjoin(
        filtered_nonull.to_crs("EPSG:4326"),
        mun_shape[["Municipio_", "geometry"]],
        how="left",
        predicate="within"
    )

    filtered_mun["Municipio_"] = filtered_mun["Municipio_"].str.lower().str.strip()
    mun_shape["Municipio_"] = mun_shape["Municipio_"].str.lower().str.strip()

    # Cálculo da média do IGATP por município
    mean_mun = filtered_mun.groupby("Municipio_")["IGATP"].mean().reset_index()

    # Merge com shapefile
    mun_map = mun_shape.merge(mean_mun, on="Municipio_", how="left")

    # Normalizar IGATP para gradiente de cores
    scaler = MinMaxScaler()
    mun_map["IGATPScaled"] = scaler.fit_transform(mun_map[["IGATP"]].fillna(0))

    # Mapa com Kepler.gl
    mapa2 = KeplerGl(
        height=600,
        data={"Municípios IGATP": mun_map},
        config={
            "version": "v1",
            "config": {
                "mapState": {
                    "latitude": 41.15,
                    "longitude": -8.6,
                    "zoom": 8,
                    "bearing": 0,
                    "pitch": 0
                },
                "mapStyle": {
                    "styleType": "muted_night"
                }
                # A camada será configurada manualmente pelo utilizador
            }
        }
    )
    keplergl_static(mapa2)

    st.caption("Interpolação por município com base no valor médio de IGATP. Configure o preenchimento da camada no Kepler para ver o gradiente.")


# TAB 3 - Mapa por Freguesia
with tab3:
    st.subheader("IGATP médio por Freguesia")

    # Instruções para o utilizador
    st.markdown("""
    ℹ️ **Para visualizar corretamente o gradiente de cores no mapa:**
    1. Clique no botão com **a seta cinzenta →** no lado esquerdo do mapa (imagem abaixo).
    2. Clique na camada **"Freguesias IGATP"** no painel do Kepler (ícone do cubo).
    3. Em **"Fill Color"**, selecione `IGATPScaled`.
    4. Escolha uma escala de cor (`quantile`, `sequential`, ou `continuous`).
    5. Ajuste o intervalo de valores, se necessário (de 0 a 1).
    """)

    # Garantir que o código da freguesia está em formato string
    df_freg["Parish_Code"] = df_freg["Parish_Code"].astype(str)

    # Juntar shapefile com dados agregados
    freg_map = freg_shape.merge(df_freg, left_on="DICOFRE_le", right_on="Parish_Code", how="left")

    # Calcular coluna normalizada
    scaler = MinMaxScaler()
    if "IGATP_Mean" in freg_map.columns:
        freg_map["IGATPScaled"] = scaler.fit_transform(freg_map[["IGATP_Mean"]].fillna(0))
    else:
        freg_map["IGATPScaled"] = 0  # fallback defensivo

    # Garantir CRS
    freg_map = freg_map.to_crs("EPSG:4326")

    # Criar o mapa com Kepler.gl
    mapa3 = KeplerGl(
        height=600,
        data={"Freguesias IGATP": freg_map},
        config={
            "version": "v1",
            "config": {
                "mapState": {
                    "latitude": 41.15,
                    "longitude": -8.6,
                    "zoom": 8,
                    "bearing": 0,
                    "pitch": 0
                },
                "mapStyle": {
                    "styleType": "muted_night"
                }
            }
        }
    )
    keplergl_static(mapa3)

    st.caption("Mapa por freguesia baseado no valor médio de IGATP. Configure o preenchimento manualmente no Kepler.")



# TAB 4 - Rankings
with tab4:
    st.subheader("🏆 Top 5 por Sub-índice IGATP")

    # Top 5 locais por subíndice
    top_igatp = filtered_nonull.groupby("Nome_Local")["IGATP"].mean().sort_values(ascending=False).head(5)
    top_rating = filtered_nonull.groupby("Nome_Local")["Rating_Bayes_norm"].mean().sort_values(ascending=False).head(5)
    top_pop = filtered_nonull.groupby("Nome_Local")["Popularity_norm"].mean().sort_values(ascending=False).head(5)
    top_sent = filtered_nonull.groupby("Nome_Local")["Sentiment_norm"].mean().sort_values(ascending=False).head(5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🌐 Top 5 IGATP**")
        st.dataframe(top_igatp)

        st.markdown("**⭐ Top 5 Rating Bayesiano**")
        st.dataframe(top_rating)

    with col2:
        st.markdown("**📣 Top 5 Popularidade**")
        st.dataframe(top_pop)

        st.markdown("**💬 Top 5 Sentimento**")
        st.dataframe(top_sent)

    # NOVA SECÇÃO – Rankings territoriais
    st.markdown("---")
    st.subheader("🗺️ Rankings Territoriais de IGATP")

    # Top 3 municípios
    top_mun = mun_map[["Municipio_", "IGATP"]].dropna().sort_values("IGATP", ascending=False)
    st.markdown("**🏙️ Municípios com Maior IGATP Médio**")
    st.dataframe(top_mun.head(3).reset_index(drop=True))

    st.markdown("**🏙️ Municípios com Menor IGATP Médio**")
    st.dataframe(top_mun.tail(3).reset_index(drop=True))

    # Top 3 freguesias
    top_freg = freg_map[["Parish", "IGATP_Mean"]].dropna().sort_values("IGATP_Mean", ascending=False)
    st.markdown("**🏘️ Freguesias com Maior IGATP Médio**")
    st.dataframe(top_freg.head(3).reset_index(drop=True))

    st.markdown("**🏘️ Freguesias com Menor IGATP Médio**")
    st.dataframe(top_freg.tail(3).reset_index(drop=True))



# TAB 5 - Evolução Temporal
with tab5:
    st.subheader("Polaridade média ao longo do tempo")
    df_topics = pd.read_csv("C:/Users/Fernanda Costa/OneDrive - Universidade de Aveiro/Desktop/seminar_project/6_unsupervised_learning/ratings_polarity_lda_topics.csv")
    df_topics["Data_Convertida"] = pd.to_datetime(df_topics["Data_Convertida"], errors='coerce')
    df_month = df_topics.groupby(pd.Grouper(key="Data_Convertida", freq="M")).agg({"Polaridade": "mean"}).dropna().reset_index()
    chart = alt.Chart(df_month).mark_line().encode(
        x=alt.X("Data_Convertida:T", title="Data"),
        y=alt.Y("Polaridade:Q", title="Polaridade Média")
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)
    st.caption("Linha temporal da polaridade média dos comentários (sentimento). Tendência geralmente positiva e estável.")

# FOOTER
st.markdown("---")
st.caption("Projeto desenvolvido por Beatriz Santos e Joana Guerreiro | Seminário 2025 | Mestrado em Ciência de Dados para Ciências Sociais | Universidade de Aveiro")