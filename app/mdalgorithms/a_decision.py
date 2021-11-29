from typing import final
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.error import URLError
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
from sklearn import model_selection


def pronostico(df):
    try:
        st.title("Árboles de decisión - PRONÓSTICO")
        only_h = st.radio("Solo cabecera:", ["Sí", "No"])
        if only_h == "Sí":
            st.write("### Dataframe head", df.head().sort_index())
        else:
            st.write("### Dataframe", df.sort_index())
        with st.container():
            st.write("##  Elección de variables")
            selectedVars = st.multiselect("Selecciona las variables a seleccionar", df.select_dtypes(
                include=['int64', 'float64']).columns)
            if len(selectedVars) > 0:
                final_df = df.reindex(selectedVars, axis=1)
                st.write(final_df.head())
            else:
                final_df = df

        with st.container():
            st.write("#### Selección de variables predictoras → X ←")
            predict_vars = st.multiselect("Selecciona las variables a seleccionar", final_df.select_dtypes(
                include=['int64', 'float64']).columns, key="X")
            if len(predict_vars) > 0:
                X = np.array(df[predict_vars])
            pronostic_var = st.selectbox("Selecciona las variables a seleccionar", final_df.select_dtypes(
                include=['int64', 'float64']).columns, key="Y")
            if pronostic_var != "":
                Y = np.array(df[[pronostic_var]])
            if len(predict_vars) > 0 and pronostic_var != "":
                X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,
                                                                                    test_size=0.2,
                                                                                    random_state=1234,
                                                                                    shuffle=True)
                pronosticoAD = DecisionTreeRegressor()
                pronosticoAD.fit(X_train, Y_train)
                Y_pronostico = pronosticoAD.predict(X_test)
                if st.button("Graficar pronóstico"):
                    plt.plot(Y_test, color='green', marker='o', label='Y_test')
                    plt.plot(Y_pronostico, color='red',
                             marker='o', label='Y_pronostico')
                    plt.xlabel('Elemento')
                    plt.ylabel(pronostic_var)
                    plt.title('MODEL FIT')
                    plt.grid(True)
                    plt.legend()
                    st.pyplot(plt)
                with st.container():
                    col_par, col_y = st.columns(2)
                    with col_par:
                        st.caption(f"Criterio: {pronosticoAD.criterion}")
                        st.caption(
                            f"Importancia variables: {pronosticoAD.feature_importances_}")
                        st.caption(f"MAE: {mean_absolute_error(Y_test, Y_pronostico)}")
                        st.caption(f"MSE: { mean_squared_error(Y_test, Y_pronostico)}")
                        st.caption(
                            f"RMSE: {mean_squared_error(Y_test, Y_pronostico, squared=False)}")
                        st.caption(f"Score: {r2_score(Y_test, Y_pronostico)}")
                    with col_y:
                        importancia = pd.DataFrame({'Variable': list(final_df),
                                                    'Importancia': pronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
                        st.dataframe(importancia)
                with st.container():
                    st.caption("Imprimir el árbol puede tardar algunos minutos...")
                    if st.button("Imprimir Árbol"):
                        plt.clf()
                        reporte = export_text(pronosticoAD, feature_names=predict_vars)
                        st.code(reporte)
                with st.container():
                    st.write("## Nuevos pronósticos 📈")
                    values = []
                    with st.form("nuevo_pronostico"):
                        for var in predict_vars:
                            res = st.number_input(var, format="%.4f")
                            values.append(res)
                        aux_d = {}
                        for i in range(len(predict_vars)):
                            aux_d[predict_vars[i]] = [values[i]]
                        df_pronostic = pd.DataFrame(aux_d)
                        submitted = st.form_submit_button(
                            f"Pronosticar {pronostic_var}")
                        if submitted:
                            p_value = pronosticoAD.predict(df_pronostic)[0]
                            score = str(r2_score(Y_test, Y_pronostico))+" model score"
                            st.metric("PRONOSTICO", p_value, score)

    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

def clasificacion(df):
    try:
        st.title("Árboles de decisión - CLASIFICACIÓN")
        only_h = st.radio("Solo cabecera:", ["Sí", "No"])
        if only_h == "Sí":
            st.write("### Dataframe head", df.head().sort_index())
        else:
            st.write("### Dataframe", df.sort_index())
        with st.container():
            st.write("##  Elección de variables")
            selectedVars = st.multiselect("Selecciona las variables a seleccionar", df.select_dtypes(
                include=['int64', 'float64']).columns)
            if len(selectedVars) > 0:
                final_df = df.reindex(selectedVars, axis=1)
                st.write(final_df.head())
            else:
                final_df = df

        with st.container():
            st.write("#### Selección de variables predictoras → X ← y variable clase → Y ←")
            predict_vars = st.multiselect("Selecciona las variables predictoras", final_df.select_dtypes(
                include=['int64', 'float64']).columns, key="X")
            if len(predict_vars) > 0:
                X = np.array(df[predict_vars])
            pronostic_var = st.selectbox("Selecciona la variable clase", df.columns, key="Y")
            if pronostic_var != "":
                Y = np.array(df[[pronostic_var]])
            if len(predict_vars) > 0 and pronostic_var != "":
                X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,
                                                                                    test_size=0.2,
                                                                                    random_state=0,
                                                                                    shuffle=True)
                clasificacionAD = DecisionTreeClassifier()
                clasificacionAD.fit(X_train, Y_train)
                Y_clasificacion = clasificacionAD.predict(X_validation)
                with st.container():
                    st.write("### Validación del modelo")
                    Y_clasificacion = clasificacionAD.predict(X_validation)
                    matriz_clasificacion = pd.crosstab(Y_validation.ravel(), 
                                        Y_clasificacion, 
                                        rownames=['Real'], 
                                        colnames=['Clasificación'])
                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.dataframe(matriz_clasificacion)
                    with col_r:
                        st.caption(f"Criterio: {clasificacionAD.criterion}")
                        st.caption(f"Importancia variables: {clasificacionAD.feature_importances_}")
                        st.caption(f"Exactitud: {clasificacionAD.score(X_validation, Y_validation)}")
                        st.text(classification_report(Y_validation, Y_clasificacion))
                    importancia = pd.DataFrame({'Variable': list(final_df),
                                                    'Importancia': clasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.dataframe(importancia)
                with st.container():
                    st.caption("Imprimir el árbol puede tardar algunos minutos...")
                    if st.button("Imprimir Árbol"):
                        plt.clf()
                        reporte = export_text(clasificacionAD, feature_names=predict_vars)
                        st.code(reporte)
                with st.container():
                    st.write("## Nuevas clasificaciones 📈")
                    values = []
                    with st.form("nueva_clasificacion"):
                        for var in predict_vars:
                            res = st.number_input(var, format="%.4f")
                            values.append(res)
                        aux_d = {}
                        for i in range(len(predict_vars)):
                            aux_d[predict_vars[i]] = [values[i]]
                        df_pronostic = pd.DataFrame(aux_d)
                        submitted = st.form_submit_button(
                            f"Clasificar {pronostic_var}")
                        if submitted:
                            c_value = clasificacionAD.predict(df_pronostic)[0]
                            score = clasificacionAD.score(X_validation, Y_validation)
                            st.write("CLASIFICACIÓN:")
                            st.title(f"→ {c_value} ←")
                            st.caption(f"Score: {score}")

    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )