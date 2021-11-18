import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from urllib.error import URLError
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering


def select_columns(df, selected_cols):
    if len(selected_cols) == 0:
        return np.array(df)
    return np.array(df[selected_cols])


def clustering_jerarquico(df):
    try:
        st.title("Clusterización jerárquica")
        if str(df) != 'None':
            with st.container():
                final_np = np.array(df)
                st.subheader("Selección de características")
                selVars = st.multiselect("Selecciona las variables con las que trabajarás: ", df.select_dtypes(
                    include=['int64', 'float64']).columns)
                final_np = select_columns(df, selVars)
                st.write(pd.DataFrame(final_np).head())
            with st.container():
                st.write("## Aplicación del algoritmo")
                st.write(
                    "Después de estandarizar, nuestro árbol jerárquico se ve de la siguiente manera: ")
                estandarizar = StandardScaler()
                MEstandarizada = estandarizar.fit_transform(final_np)
                if st.button("Generar árbol"):
                    plt.clf()
                    plt.figure(figsize=(10, 7))
                    Arbol = shc.dendrogram(shc.linkage(
                        MEstandarizada, method='complete', metric='euclidean'))
                    st.pyplot(plt)
            with st.container():
                num_clusteres = st.number_input(
                    "Selecciona el número de clusters: ", min_value=1, max_value=10, step=1)
                if st.button("¡Clusterizar!"):
                    MJerarquico = AgglomerativeClustering(
                        n_clusters=num_clusteres, linkage='complete', affinity='euclidean')
                    MJerarquico.fit_predict(MEstandarizada)
                    df_final = df[selVars]
                    df_final['clusterH'] = MJerarquico.labels_
                    st.write(df_final.head())
                    col_count, col_centroides = st.columns([1, 4])
                    with col_count:
                        st.write("Cluster / \#")
                        st.write(df_final.groupby(['clusterH'])[
                                 'clusterH'].count())
                    with col_centroides:
                        st.write("Centroides")
                        st.write(df_final.groupby('clusterH').mean())
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
