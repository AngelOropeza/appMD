import streamlit as st
import pandas as pd

from mdalgorithms import acd
from mdalgorithms import eda
from mdalgorithms import distancias
from mdalgorithms import cjerarquico
from mdalgorithms import cparticional
from mdalgorithms import mdapriori
from mdalgorithms import a_decision
from urllib.error import URLError


algorithms = [
    "EDA🔎", "ACD y PCA✅", "Distancias🗺️",
    "Clustering Jerárquico🏷️", "Clustering Particional🏷️",
    "Apriori📋", "Árboles de decisión🌳 - Pronóstico",
    "Árboles de decisión🌳 - Clasificación"
              ]

try:
    st.sidebar.title("Minería de Datos")
    st.sidebar.image("img\logo1.jpg", width=300)
    st.sidebar.text("Sistema creado para aplicar y \nproveer distintos algoritmos de \nminería de datos")
    page_selected = st.sidebar.radio("ALGORITMOS:", algorithms)
    file = st.sidebar.file_uploader("Selecciona archivo 📁", ['csv', 'txt'])
    st.sidebar.code("""Desarrollado por:\nOropeza Castañeda Angel Eduardo""")

    if file is not None:
        df = pd.read_csv(file)
        if page_selected == "EDA🔎":
            eda.eda(df)
        elif page_selected == "ACD y PCA✅":
            acd.acd(df)
        elif page_selected == "Distancias🗺️":
            distancias.distancias(df)
        elif page_selected == "Clustering Jerárquico🏷️":
            cjerarquico.clustering_jerarquico(df)
        elif page_selected == "Clustering Particional🏷️":
            cparticional.clustering_particional(df)
        elif page_selected == "Apriori📋":
            mdapriori.algoritmoapriori(df)
        elif page_selected == "Árboles de decisión🌳 - Pronóstico":
            a_decision.pronostico(df)
        elif page_selected == "Árboles de decisión🌳 - Clasificación":
            a_decision.clasificacion(df)
    else:
        st.error(
            """
            Dataframe no seleccionado :(
        """
        )

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )