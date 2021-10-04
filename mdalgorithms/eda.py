import streamlit as st
import pandas as pd
import altair as alt
import io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from urllib.error import URLError

def get_UN_data():
    file = st.file_uploader("Selecciona archivo" ,['csv','txt'])
    if file is not None:
        df = pd.read_csv(file)
        return df

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def eda():
    try:
        st.title("ANÁLISIS EXPLORATORIO DE DATOS")
        st.write("## Selección de datos")
        df = get_UN_data()
        if str(df) != 'None':
            only_h = st.radio("Solo cabecera:", ["Sí", "No"])
            if only_h == "Sí": 
                st.write("### Dataframe head", df.head().sort_index())
            else:
                st.write("### Dataframe", df.sort_index())
            with st.container():
                st.write("## Descripción de los datos")
                col_shape, col_types = st.columns(2)
                with col_shape:
                    st.write("Filas / Columnas:\n\n", df.shape)
                with col_types:
                    tipos = [str(tipo) for tipo in df.dtypes]
                    st.write("Tipos de datos (variables): ")
                    st.write(pd.DataFrame({
                        'Variable':df.columns,
                        'Tipo':tipos
                    }))
            with st.container():
                st.write("## Identificación de datos faltantes")
                col_nullcount, col_infodf = st.columns(2)
                with col_nullcount:
                    nulls_list = [str(null_c) for null_c in df.isnull().sum()]
                    st.caption("La suma de elementos nulos por columnas es:")
                    st.write(pd.DataFrame({
                        'Variable':df.columns,
                        '# nulls':nulls_list
                    }))
                with col_infodf:
                    st.caption("Información relevante del dataframe:")
                    df_info = io.StringIO()
                    df.info(buf=df_info)
                    local_css("styles\column.css")
                    st.text(df_info.getvalue())
            with st.container():
                st.write("## Detección de valores atípicos")
                st.write("### Distribución de variables numéricas")
                st.caption("Histográmas por variable:")
                df.hist(figsize=(14,14), xrot=45)
                st.pyplot(plt)
                st.write("### Resumen estadístico de variables numéricas")
                st.caption("Resumen estadístico por variables:")
                st.write(df.describe())
                st.write("### Posibles valores atípicos")
                atipicos = st.multiselect("¿Qué variables te gustaría revisar?: ", df.columns)
                for col in atipicos:
                    plt.clf()
                    plt.figure(figsize=(6,3))
                    sns.boxplot(col, data=df)
                    st.pyplot(plt)
                st.write("### Distribución de variables categóricas")
                col_catdescribe, col_catplot = st.columns(2)
                with col_catdescribe:
                    st.caption("Resumen estadístico de variables categóricas:")
                    df_cat = df.select_dtypes(include='object')
                    st.write(df_cat)
                with col_catplot:
                    st.caption("Histográmas variables categóricas:")
                    for col in df_cat:
                        plt.clf()
                        if df_cat[col].nunique()<10:
                            plt.figure(figsize=(6,3))
                            sns.countplot(y=col, data=df_cat)
                            st.pyplot(plt)
            
            with st.container():
                st.write("## Relación entre pares de variables")
                col_matcorr, col_heatmap = st.columns(2)
                with col_matcorr:
                    st.caption("Matriz de correlación: ")
                    st.write(df.corr())
                with col_heatmap:
                    plt.clf()
                    plt.figure(figsize=(14,12))
                    sns.heatmap(df.corr(), cmap='RdBu_r', annot=True)
                    st.pyplot(plt)

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

