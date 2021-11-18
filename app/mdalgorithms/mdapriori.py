import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from urllib.error import URLError
from apyori import apriori


def algoritmoapriori(df):
    try:
        st.title("Apriori")
        if str(df) != 'None':
            with st.container():
                Transacciones = df.values.reshape(-1).tolist()
                ListaM = pd.DataFrame(Transacciones)
                ListaM['Frecuencia'] = 0
                ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(
                    by=['Frecuencia'], ascending=True)  # Conteo
                ListaM['Porcentaje'] = (
                    ListaM['Frecuencia'] / ListaM['Frecuencia'].sum())  # Porcentaje
                ListaM = ListaM.rename(columns={0: 'Item'})
                TransLista = df.stack().groupby(level=0).apply(list).tolist()
            with st.container():
                if st.button("Generar gráfico de frecuencia"):
                    plt.clf()
                    plt.figure(figsize=(16, 20), dpi=300)
                    plt.ylabel('Item')
                    plt.xlabel('Frecuencia')
                    plt.barh(ListaM['Item'],
                             width=ListaM['Frecuencia'], color='blue')
                    st.pyplot(plt)
            with st.container():
                col_sup, col_conf, col_lif = st.columns(3)
                with st.form("my_form"):
                    with col_sup:
                        min_support = st.number_input("Soporte")
                    with col_conf:
                        min_confidence = st.number_input("Confianza")
                    with col_lif:
                        min_lift = st.number_input("Elevación", step=1)
                    submitted = st.form_submit_button("Calcular reglas")
                    if submitted:
                        Reglas = apriori(TransLista, min_support=min_support,
                                         min_confidence=min_confidence, min_lift=min_lift)
                        Resultados = list(Reglas)
                        if len(Resultados) == 0:
                            st.warning(
                                "No hay reglas con los parámetros introducidos")
                        else:
                            st.info(f'Se encontraron {len(Resultados)} reglas')
                            st.json(Resultados)

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
