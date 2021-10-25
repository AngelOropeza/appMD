import streamlit as st
import pandas as pd
import altair as alt
import io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance

from urllib.error import URLError

def get_UN_data():
    file = st.file_uploader("Selecciona archivo" ,['csv','txt'])
    if file is not None:
        df = pd.read_csv(file)
        return df

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def distancias():
    distancias = ['Euclidiana', 'Chebyshev', 'Manhattan', 'Minkowski']
    try:
        st.title("Métricas de distancia")
        st.write("## Selección de datos")
        df = get_UN_data()
        if str(df) != 'None':
            only_h = st.radio("Solo cabecera:", ["Sí", "No"])
            if only_h == "Sí": 
                st.write("### Dataframe head", df.head().sort_index())
            else:
                st.write("### Dataframe", df.sort_index())
            with st.container():
                st.write("## Estandarización de los datos")
                col_info, col_sdata = st.columns(2)
                with col_info:
                    st.caption('''\n\n\nEn los algoritmos basados en distancias es fundamental escalar o normalizar los datos para que cada una de las variables contribuyan por igual en el análisis''')
                    estandarizar = StandardScaler()
                    MEstandarizada = estandarizar.fit_transform(df)
                with col_sdata:
                    st.write(MEstandarizada)
            with st.container():
                st.write("## Matrices de distancia")
                distancia = st.selectbox("Distancia a calcular:", distancias)
                if distancia == 'Euclidiana':
                    st.write(distancia)
                    DstEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean')
                    MEuclidiana = pd.DataFrame(DstEuclidiana)  
                    st.write(MEuclidiana.round(3))
                elif distancia == 'Chebyshev':
                    st.write(distancia)
                    DstChebyshev = cdist(MEstandarizada, MEstandarizada, metric='chebyshev')
                    MChebyshev = pd.DataFrame(DstChebyshev)
                    st.write(MChebyshev.round(3))
                elif distancia == 'Manhattan':
                    st.write(distancia)
                    DstManhattan = cdist(MEstandarizada, MEstandarizada, metric='cityblock')
                    MManhattan = pd.DataFrame(DstManhattan)
                    st.write(MManhattan.round(3))
                elif distancia == 'Minkowski':
                    st.write(distancia)
                    DstMinkowski = cdist(MEstandarizada, MEstandarizada, metric='minkowski', p=1.5)
                    MMinkowski = pd.DataFrame(DstMinkowski)
                    st.write(MMinkowski.round(3))
                    


        else:
            st.error(
            """
            Dataframe no seleccionado
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


distancias()