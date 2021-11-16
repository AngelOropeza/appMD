import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from urllib.error import URLError
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from sklearn.cluster import KMeans
from kneed import KneeLocator
from mdalgorithms import select_columns

def clustering_particional(df):
    try:
        st.title("Clusterización Particional")
        if str(df) != 'None':
            with st.container():
                final_np = np.array(df)
                st.subheader("Selección de características")
                selVars = st.multiselect("Selecciona las variables con las que trabajarás: ", df.select_dtypes(include=['int64','float64']).columns)
                final_np = select_columns(df,selVars)           
                st.write(pd.DataFrame(final_np).head())
            with st.container():
                st.write("## Aplicación del algoritmo")
                st.write("Después de estandarizar, nuestro árbol jerárquico se ve de la siguiente manera: ")
                estandarizar = StandardScaler()
                MEstandarizada = estandarizar.fit_transform(final_np)
                SSE = []
                for i in range(2, 12):
                    km = KMeans(n_clusters=i, random_state=0)
                    km.fit(MEstandarizada)
                    SSE.append(km.inertia_)
                if st.button("Generar gráfico del método del codo"):
                    plt.clf()
                    plt.figure(figsize=(10, 7))
                    plt.plot(range(2, 12), SSE, marker='o')
                    plt.xlabel('Cantidad de clusters *k*')
                    plt.ylabel('SSE')
                    plt.title('Elbow Method')
                    st.pyplot(plt)
            with st.container():
                col_nclust, col_sugerencia = st.columns(2)
                k1 = KneeLocator(range(2,12), SSE, curve="convex", direction="decreasing")
                with col_nclust:
                    num_clusteres = st.number_input("Selecciona el número de clusters: ", min_value=1, max_value=10, step=1, value=k1.elbow)
                with col_sugerencia:
                    st.caption(f'Empleando kneed locator; se sugieren -> {k1.elbow} <- clusters')
                if st.button("¡Clusterizar!"):
                    MParticional = KMeans(n_clusters=num_clusteres, random_state=0).fit(MEstandarizada)
                    MParticional.predict(MEstandarizada)
                    df_final = df[selVars]
                    df_final['clusterP'] = MParticional.labels_
                    st.write(df_final.head())
                    col_count, col_centroides = st.columns([1,4])
                    with col_count:
                        st.write("Cluster / \#")
                        st.write(df_final.groupby(['clusterP'])['clusterP'].count())
                    with col_centroides:
                        st.write("Centroides")
                        st.write(df_final.groupby('clusterP').mean())                
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