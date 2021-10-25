import streamlit as st
import pandas as pd
import altair as alt
import io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from urllib.error import URLError
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator


algorithms = ["EDA", "ACD", "Distancias", "Clustering Jerárquico", "Clustering Particional"]

def select_columns(df, selected_cols):
    if len(selected_cols) == 0:
        return np.array(df)
    return np.array(df[selected_cols])


def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def eda(df):
    try:
        st.title("ANÁLISIS EXPLORATORIO DE DATOS")
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

    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

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
            st.text("Gráficos de dispersión de datos: ")
            col_x, col_y = st.columns(2)
            with col_x:
                xAxis = st.selectbox("Selecciona eje x: ", df.columns)
            with col_y:
                yAxis = st.selectbox("Selecciona eje y: ", df.columns)
            if xAxis != '' and yAxis != '':
                plt.plot(df[xAxis], df[yAxis], 'o')
                plt.xlabel(xAxis)
                plt.ylabel(yAxis)
                col_x.pyplot(plt)
                plt.clf()
                sns.scatterplot(x=xAxis, y =yAxis, data=df, hue='Type') # TODO handler 4 every df
                plt.xlabel(xAxis)
                plt.ylabel(yAxis)
                col_y.pyplot(plt)

        with st.container():
            xAxis, yAxis = '', ''
            st.write("##  Identificación de relaciones entre variables")
            st.text("Una matriz de correlaciones es útil para analizar la relación entre las variables numéricas.")
            varTop = st.selectbox("Selecciona variable a revisar: ", df.select_dtypes(include=['int64','float64']).columns)
            col_mat, col_top = st.columns(2)
            with col_mat:
                st.caption("Matriz de correlaciones: ")
                st.write(df.corr())
                plt.clf()
                plt.figure(figsize=(14,7))
                MatrizInf = np.triu(df.corr())
                sns.heatmap(df.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
                st.pyplot(plt)
            with col_top:
                st.caption("Top por variable: ")
                st.write(df.select_dtypes(include=['int64','float64']).corr()[varTop].sort_values(ascending=False)[:10])
                plt.clf()
                plt.figure(figsize=(14,7))
                MatrizSup = np.tril(df.corr())
                sns.heatmap(df.corr(), cmap='RdBu_r', annot=True, mask=MatrizSup)
                st.pyplot(plt)

            with st.container():
                st.write("##  Elección de variables")
                dropVars = st.multiselect("Selecciona las variables a eliminar post análisis: ", df.select_dtypes(include=['int64','float64']).columns)
                final_df = df.drop(columns=dropVars)
                st.write(final_df.head())
                col_sep, col_name = st.columns(2)
                sep = col_sep.text_input("Separador: ", value=',', max_chars=2)
                df_name = col_name.text_input("Nombre: ", value='output')
                csv = final_df.to_csv().encode('utf-8')
                st.download_button("Descargar dataset modificado", data=csv, file_name="{}.csv".format(df_name))


    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

def distancias(df):
    try:
        distancias = ['Euclidiana', 'Chebyshev', 'Manhattan', 'Minkowski']
        st.title("Métricas de distancia")
        st.write("## Selección de datos")
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

def clustering_jerarquico(df):
    try:
        st.title("Clusterización jerárquica")
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
                if st.button("Generar árbol"):
                    plt.clf()
                    plt.figure(figsize=(10, 7))
                    Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
                    st.pyplot(plt)
            with st.container():
                num_clusteres = st.number_input("Selecciona el número de clusters: ", min_value=1, max_value=10, step=1)
                if st.button("¡Clusterizar!"):
                    MJerarquico = AgglomerativeClustering(n_clusters=num_clusteres, linkage='complete', affinity='euclidean')
                    MJerarquico.fit_predict(MEstandarizada)
                    df_final = df[selVars]
                    df_final['clusterH'] = MJerarquico.labels_
                    st.write(df_final.head())
                    col_count, col_centroides = st.columns([1,4])
                    with col_count:
                        st.write("Cluster / \#")
                        st.write(df_final.groupby(['clusterH'])['clusterH'].count())
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

# M A I N 
try:
    st.sidebar.image("img\logo1.jpg",width=300)
    st.sidebar.title("MD APPv2")
    st.sidebar.text("App created to apply and provide DM algorithms")
    page_selected = st.sidebar.radio("ALGORITHMS:", algorithms)
    file = st.sidebar.file_uploader("Selecciona archivo" ,['csv','txt'])
    if file is not None:
        df = pd.read_csv(file)
        if page_selected == "EDA":
            eda(df)
        elif page_selected == "ACD":
            acd(df)
        elif page_selected == "Distancias":
            distancias(df)
        elif page_selected == "Clustering Jerárquico":
            clustering_jerarquico(df)
        elif page_selected == "Clustering Particional":
            clustering_particional(df)
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