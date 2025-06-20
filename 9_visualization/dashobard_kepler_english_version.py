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
with st.expander("ℹ️ About the Project"):
    st.markdown("""
    This interactive dashboard presents the **IGATP – Global Index of Perceived Touristic Attractiveness**, developed from public Google Maps data (ratings and comments). The goal is to evaluate the perceived attractiveness of tourist locations in the **Porto Metropolitan Area (AMP)** based on three key dimensions:

    - **Bayesian Rating**: quality adjusted by number of reviews;
    - **Popularity**: number of reviews per place;
    - **Sentiment**: average polarity of translated user comments.

    These sub-indices are combined using customizable weights to generate a final IGATP score ranging from 0 to 1.

    The dashboard allows you to:
    - Visualize georeferenced places and their IGATP values;
    - Analyze spatial interpolation of the index by municipality and parish;
    - Explore top-ranked places, municipalities, and parishes;
    - Track sentiment evolution over time;
    - Filter results by **thematic group** and **tourist profile** (K-Medoids cluster).

    ---

    ### 🧱 Tourist Profiles (K-Medoids Clusters)

    Tourist spots were grouped into 6 distinct profiles based on PCA and clustering:

    - **Mainstream Core** (Cluster 2): average quality and popularity — dominant profile.
    - **Flagship Venues** (Cluster 3): high-quality, highly visible top locations.
    - **Hidden Popular** (Cluster 1): widely visited places with lower sentiment — possibly overrated.
    - **Underperformers** (Cluster 4): low visibility and low ratings.
    - **Boutique / Niche** (Cluster 0): niche or specialized places — high ratings, low exposure.
    - **Extreme Outlier** (Cluster 5): extremely distinct outlier case.

    ---

    ### 🧠 Dominant Topics (LDA Topic Modeling)

    User reviews were analyzed through topic modeling (LDA), resulting in four key themes:

    - **Topic 0 – Outdoor & Nature Leisure**  
      (“beach”, “walk”, “sand”, “view”, “restaurant”, “quiet”)  
      → natural and scenic experiences, calm and open-air enjoyment.

    - **Topic 1 – Accommodation & Comfort**  
      (“room”, “clean”, “bed”, “staff”, “breakfast”)  
      → lodging services, cleanliness, and hospitality.

    - **Topic 2 – Cultural & Heritage Visits**  
      (“museum”, “history”, “visit”, “portuguese”)  
      → museums, historic and educational value.

    - **Topic 3 – Gastronomic Experience**  
      (“food”, “service”, “restaurant”, “wine”)  
      → food quality, dining satisfaction, and service evaluation.

    ---

    🔍 For more details on methodology, see the project documentation or explore the maps in each dashboard tab.
    """)


# SIDEBAR - FILTERS
with st.sidebar:
    st.markdown("### 🎛️ **Visualization Filters**")

    # Filter: Thematic Group
    st.markdown("**Thematic Group**")
    grupos = st.multiselect(
        "Select the groups to include:",
        options=points["Grupo_Tematico"].dropna().unique().tolist(),
        default=points["Grupo_Tematico"].dropna().unique().tolist(),
        help="Select the types of tourism services to be included in the index."
    )

    st.markdown("---")

    # Filter: K-Medoids Cluster with readable names
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
        "Select location profiles:",
        options=list(cluster_options.keys()),
        default=list(cluster_options.keys()),
        help="Clusters obtained through PCA and K-Medoids. Represent distinct tourism profiles."
    )

    selected_clusters = [cluster_options[label] for label in selected_cluster_labels]

    st.markdown("---")

    # Sub-index Weights
    st.markdown("### ⚖️ **Sub-index Weights**")

    w1 = st.slider("Bayesian Rating", 0.0, 1.0, 1/3, help="Rating adjusted for popularity (Bayesian).")
    w2 = st.slider("Popularity", 0.0, 1.0, 1/3, help="Number of reviews for the location.")
    w3 = st.slider("Sentiment", 0.0, 1.0, 1/3, help="Average sentiment polarity of user comments.")

    total = w1 + w2 + w3 or 1
    w1, w2, w3 = w1 / total, w2 / total, w3 / total





# CALCULATE IGATP & FILTER DATA
filtered = points[
    (points["Grupo_Tematico"].isin(grupos)) &
    (points["cluster_k7_pam"].isin(selected_clusters))
].copy()

filtered["IGATP"] = w1 * filtered["Rating_Bayes_norm"] + w2 * filtered["Popularity_norm"] + w3 * filtered["Sentiment_norm"]
filtered_nonull = filtered[filtered["IGATP"].notna()]

# Filter only points within the AMP boundaries
mun_shape = mun_shape.to_crs("EPSG:4326")
filtered_nonull = gpd.sjoin(
    filtered_nonull.to_crs("EPSG:4326"),
    mun_shape[["geometry"]],
    how="inner",
    predicate="within"
).drop(columns="index_right", errors="ignore")


# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📍 Point Map", "🗺️ Municipality Map", "🏘️ Parish Map", "📊 Rankings", "📈 Temporal Evolution"])

# TAB 1 - KEPLER MAP (Points)
with tab1:
    st.subheader("Locations with IGATP")

    # Instructions to correctly configure the tooltip in Kepler.gl
    st.markdown("""
    ℹ️ **To view complete details for each location by clicking a point on the map:**

    1. Click the **grey arrow →** button on the left side of the map (as shown below).
    2. Go to the **"Interactions"** tab at the top of the side panel.
    3. In **"Tooltip"**, select the following fields to display in the pop-up:

       - `Nome_Local` → Name of the Location  
       - `Cidade`  
       - `Categoria`  
       - `IGATP` → IGATP Score (based on selected weights)  
       - `Locais_Semelhantes_Perto` → Number of Similar Nearby Locations  
       - `dominant_topic` → Dominant topic associated with the location

    ⚠️ These fields are not shown by default — they must be manually activated.
    """)

    # Map with Kepler.gl
    mapa1 = KeplerGl(
        height=600,
        data={
            "IGATP Points": filtered_nonull,
            "AMP Municipalities": mun_shape
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
                    "layers": []  # default layer, tooltip configured manually
                }
            }
        }
    )

    keplergl_static(mapa1)
    st.caption("Points represent tourist locations. Colors and attributes can be customized directly in the Kepler.gl interface.")


# TAB 2 - Interpolation by Municipality
with tab2:
    st.subheader("Average IGATP by Municipality")

    # Instruction for the user
    st.markdown("""
    ℹ️ **To properly visualize the color gradient on the map, follow these steps in the Kepler panel (layer icon):**
    1. Click the button with the **grey arrow →** on the left side of the map (see image below).
    2. Click on **"IGATP Municipalities"** in the layer list.
    3. In **"Fill Color"**, select the variable `IGATPScaled`.
    4. Choose a color scale (suggestion: `quantile` or `sequential`).
    5. Adjust the range if needed (0 to 1).
    """)

    # Spatial Join between points and municipalities to calculate average IGATP
    filtered_mun = gpd.sjoin(
        filtered_nonull.to_crs("EPSG:4326"),
        mun_shape[["Municipio_", "geometry"]],
        how="left",
        predicate="within"
    )

    filtered_mun["Municipio_"] = filtered_mun["Municipio_"].str.lower().str.strip()
    mun_shape["Municipio_"] = mun_shape["Municipio_"].str.lower().str.strip()

    # Calculate average IGATP per municipality
    mean_mun = filtered_mun.groupby("Municipio_")["IGATP"].mean().reset_index()

    # Merge with shapefile
    mun_map = mun_shape.merge(mean_mun, on="Municipio_", how="left")

    # Normalize IGATP for color gradient
    scaler = MinMaxScaler()
    mun_map["IGATPScaled"] = scaler.fit_transform(mun_map[["IGATP"]].fillna(0))

    # Map with Kepler.gl
    mapa2 = KeplerGl(
        height=600,
        data={"IGATP Municipalities": mun_map},
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
                # Layer will be configured manually by the user
            }
        }
    )
    keplergl_static(mapa2)

    st.caption("Municipality-level interpolation based on average IGATP. Configure the layer fill in Kepler to view the gradient.")


# TAB 3 - Map by Parish
with tab3:
    st.subheader("Average IGATP by Parish")

    # Instructions for the user
    st.markdown("""
    ℹ️ **To properly visualize the color gradient on the map:**
    1. Click the button with the **grey arrow →** on the left side of the map (see image below).
    2. Click on the **"IGATP Parishes"** layer in the Kepler panel (cube icon).
    3. In **"Fill Color"**, select `IGATPScaled`.
    4. Choose a color scale (`quantile`, `sequential`, or `continuous`).
    5. Adjust the value range if necessary (from 0 to 1).
    """)

    # Ensure parish code is string
    df_freg["Parish_Code"] = df_freg["Parish_Code"].astype(str)

    # Merge shapefile with aggregated data
    freg_map = freg_shape.merge(df_freg, left_on="DICOFRE_le", right_on="Parish_Code", how="left")

    # Calculate normalized column
    scaler = MinMaxScaler()
    if "IGATP_Mean" in freg_map.columns:
        freg_map["IGATPScaled"] = scaler.fit_transform(freg_map[["IGATP_Mean"]].fillna(0))
    else:
        freg_map["IGATPScaled"] = 0  # defensive fallback

    # Ensure CRS
    freg_map = freg_map.to_crs("EPSG:4326")

    # Create map with Kepler.gl
    mapa3 = KeplerGl(
        height=600,
        data={"IGATP Parishes": freg_map},
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

    st.caption("Parish-level map based on average IGATP. Manually configure the layer fill in Kepler to view the color gradient.")




# TAB 4 - Rankings
with tab4:
    st.subheader("🏆 Top 5 by IGATP Sub-index")

    # Top 5 places by subindex
    top_igatp = filtered_nonull.groupby("Nome_Local")["IGATP"].mean().sort_values(ascending=False).head(5)
    top_rating = filtered_nonull.groupby("Nome_Local")["Rating_Bayes_norm"].mean().sort_values(ascending=False).head(5)
    top_pop = filtered_nonull.groupby("Nome_Local")["Popularity_norm"].mean().sort_values(ascending=False).head(5)
    top_sent = filtered_nonull.groupby("Nome_Local")["Sentiment_norm"].mean().sort_values(ascending=False).head(5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🌐 Top 5 IGATP**")
        st.dataframe(top_igatp)

        st.markdown("**⭐ Top 5 Bayesian Rating**")
        st.dataframe(top_rating)

    with col2:
        st.markdown("**📣 Top 5 Popularity**")
        st.dataframe(top_pop)

        st.markdown("**💬 Top 5 Sentiment**")
        st.dataframe(top_sent)

    # NEW SECTION – Territorial Rankings
    st.markdown("---")
    st.subheader("🗺️ IGATP Territorial Rankings")

    # Top 3 municipalities
    top_mun = mun_map[["Municipio_", "IGATP"]].dropna().sort_values("IGATP", ascending=False)
    st.markdown("**🏙️ Municipalities with Highest Average IGATP**")
    st.dataframe(top_mun.head(3).reset_index(drop=True))

    st.markdown("**🏙️ Municipalities with Lowest Average IGATP**")
    st.dataframe(top_mun.tail(3).reset_index(drop=True))

    # Top 3 parishes
    top_freg = freg_map[["Parish", "IGATP_Mean"]].dropna().sort_values("IGATP_Mean", ascending=False)
    st.markdown("**🏘️ Parishes with Highest Average IGATP**")
    st.dataframe(top_freg.head(3).reset_index(drop=True))

    st.markdown("**🏘️ Parishes with Lowest Average IGATP**")
    st.dataframe(top_freg.tail(3).reset_index(drop=True))


# TAB 5 - Temporal Evolution
with tab5:
    st.subheader("Average Sentiment Polarity Over Time")
    df_topics = pd.read_csv("C:/Users/Fernanda Costa/OneDrive - Universidade de Aveiro/Desktop/seminar_project/6_unsupervised_learning/ratings_polarity_lda_topics.csv")
    df_topics["Data_Convertida"] = pd.to_datetime(df_topics["Data_Convertida"], errors='coerce')
    df_month = df_topics.groupby(pd.Grouper(key="Data_Convertida", freq="M")).agg({"Polaridade": "mean"}).dropna().reset_index()
    chart = alt.Chart(df_month).mark_line().encode(
        x=alt.X("Data_Convertida:T", title="Date"),
        y=alt.Y("Polaridade:Q", title="Average Polarity")
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)
    st.caption("Time series of average comment polarity (sentiment). Generally stable and positive trend.")


# FOOTER
st.markdown("---")
st.caption("Project developed by Beatriz Santos and Joana Guerreiro | Seminar 2025 | Master's in Data Science for Social Sciences | University of Aveiro")
