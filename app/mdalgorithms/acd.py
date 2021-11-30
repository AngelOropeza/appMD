import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from urllib.error import URLError


def get_UN_data():
    file = st.file_uploader("Selecciona archivo", ['csv', 'txt'])
    if file is not None:
        df = pd.read_csv(file)
        return df


def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()),
                    unsafe_allow_html=True)


def acd(df):
    try:
        st.title("Selección de características ACD & PCA")
        only_h = st.radio("Solo cabecera:", ["Sí", "No"])
        if only_h == "Sí":
            st.write("### Dataframe head", df.head().sort_index())
        else:
            st.write("### Dataframe", df.sort_index())
        with st.container():
            xAxis, yAxis = '', ''
            st.write("##  Evaluación visual")
            st.text("Gráficos de dispersión de datos en pares de variables: ")
            col_x, col_y = st.columns(2)
            with col_x:
                xAxis = st.selectbox("Selecciona eje x: ", df.columns)
            with col_y:
                yAxis = st.selectbox("Selecciona eje y: ", df.columns)
            if st.button("Graficar par"):
                if xAxis != '' and yAxis != '':
                    plt.plot(df[xAxis], df[yAxis], 'o')
                    plt.xlabel(xAxis)
                    plt.ylabel(yAxis)
                    col_x.pyplot(plt)

        with st.container():
            xAxis, yAxis = '', ''
            st.write("##  Identificación de relaciones entre variables")
            st.text(
                "Una matriz de correlaciones es útil para analizar la relación entre las variables numéricas.")
            varTop = st.selectbox("Selecciona variable a revisar: ", df.select_dtypes(
                include=['int64', 'float64']).columns)
            col_mat, col_top = st.columns(2)
            with col_mat:
                st.caption("Matriz de correlaciones: ")
                st.write(df.corr())
                plt.clf()
                plt.figure(figsize=(14, 7))
                MatrizInf = np.triu(df.corr())
                sns.heatmap(df.corr(), cmap='RdBu_r',
                            annot=True, mask=MatrizInf)
                st.pyplot(plt)
            with col_top:
                st.caption("Top por variable: ")
                st.write(df.select_dtypes(include=['int64', 'float64']).corr()[
                         varTop].sort_values(ascending=False)[:10])
                plt.clf()
                plt.figure(figsize=(14, 7))
                MatrizSup = np.tril(df.corr())
                sns.heatmap(df.corr(), cmap='RdBu_r',
                            annot=True, mask=MatrizSup)
                st.pyplot(plt)

            with st.container():
                st.write("##  Elección de variables")
                dropVars = st.multiselect("Selecciona las variables a eliminar post análisis: ", df.select_dtypes(
                    include=['int64', 'float64']).columns)
                final_df = df.drop(columns=dropVars)
                st.write(final_df.head())
                col_sep, col_name = st.columns(2)
                sep = col_sep.text_input("Separador: ", value=',', max_chars=2)
                df_name = col_name.text_input("Nombre: ", value='output')
                csv = final_df.to_csv().encode('utf-8')
                st.download_button("Descargar dataset modificado",
                                   data=csv, file_name="{}.csv".format(df_name))

    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )
