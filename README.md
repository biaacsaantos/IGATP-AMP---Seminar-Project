# IGATP – Perceived Tourist Attractiveness Index (AMP)

This repository contains the full project developed within the "Seminário" course (Master's in Data Science for Social Sciences – University of Aveiro). The objective is to build a composite index of perceived tourist attractiveness for the Porto Metropolitan Area, using public data from Google Places API.

## Structure

- `1_data_collection/`: Google Places API queries and geocoding
- `2_pre_processing_NLP/`: text cleaning and language normalization
- `3_exploratory_analysis/`: descriptive and spatial exploratory analysis
- `4_bayesian_rating_adjustment/`: adjusted scoring for low-review places
- `5_composite_index/`: final IGATP index calculation
- `6_unsupervised_learning/`: clustering (K-Medoids, PCA, BERTopic)
- `7_validation/`: statistical validation and interpretation
- `8_spatial_analysis/`: thematic maps and spatial aggregation
- `9_visualization/`: Streamlit dashboard with Kepler.gl
- `10_powerpoint/`: presentation slides
- `11_final_report/`: final academic report

## Note on large files

Due to GitHub’s file size limitations, the CAOP2023 GPKG file used for geographic processing is **not included** in this repository.  
Please download it directly from [DGT - Direção-Geral do Território](https://www.dgterritorio.gov.pt/).

## Technologies

- Python, Pandas, GeoPandas, NumPy
- scikit-learn, Streamlit, Kepler.gl
- NLTK, VADER, Gensim, BERTopic

## Authors

Beatriz Santos & Joana Guerreiro  
Mestrado em Ciência de Dados para Ciências Sociais  
Universidade de Aveiro — 2024/2025
