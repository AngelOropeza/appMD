import streamlit as st
import pandas as pd
import mdalgorithms.acd
import mdalgorithms.cjerarquico
import mdalgorithms.cparticional
import mdalgorithms.distancias
import mdalgorithms.eda
import mdalgorithms.mdapriori
from urllib.error import URLError

algorithms = ["EDA", "ACD", "Distancias", "Clustering Jerárquico", "Clustering Particional", "Apriori"]

try:
    st.sidebar.image("img\logo1.jpg",width=300)
    st.sidebar.title("MD APPv2")
    st.sidebar.text("App created to apply and provide DM algorithms")
    page_selected = st.sidebar.radio("ALGORITHMS:", algorithms)
    file = st.sidebar.file_uploader("Selecciona archivo" ,['csv','txt'])
    if file is not None:
        df = pd.read_csv(file)
        if page_selected == "EDA":
            mdalgorithms.eda(df)
        elif page_selected == "ACD":
            mdalgorithms.acd(df)
        elif page_selected == "Distancias":
            mdalgorithms.distancias(df)
        elif page_selected == "Clustering Jerárquico":
            mdalgorithms.clustering_jerarquico(df)
        elif page_selected == "Clustering Particional":
            mdalgorithms.clustering_particional(df)
        elif page_selected == "Apriori":
            mdalgorithms.algoritmoapriori(df)
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