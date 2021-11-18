import streamlit as st
import pandas as pd
from mdalgorithms import acd
from mdalgorithms import eda
from mdalgorithms import distancias
from mdalgorithms import cjerarquico
from mdalgorithms import cparticional
from mdalgorithms import mdapriori
from urllib.error import URLError

algorithms = ["EDA", "ACD", "Distancias",
              "Clustering Jerárquico", "Clustering Particional", "Apriori"]

try:
    st.sidebar.image("img\logo1.jpg", width=300)
    st.sidebar.title("MD APPv2")
    st.sidebar.text("App created to apply and provide DM algorithms")
    page_selected = st.sidebar.radio("ALGORITHMS:", algorithms)
    file = st.sidebar.file_uploader("Selecciona archivo", ['csv', 'txt'])
    if file is not None:
        df = pd.read_csv(file)
        if page_selected == "EDA":
            eda.eda(df)
        elif page_selected == "ACD":
            acd.acd(df)
        elif page_selected == "Distancias":
            distancias.distancias(df)
        elif page_selected == "Clustering Jerárquico":
            cjerarquico.clustering_jerarquico(df)
        elif page_selected == "Clustering Particional":
            cparticional.clustering_particional(df)
        elif page_selected == "Apriori":
            mdapriori.algoritmoapriori(df)
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
