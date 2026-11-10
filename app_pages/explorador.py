"""Página 3 · Explorador vectorial: búsqueda semántica directa (sin LLM) con filtros por metadatos."""
import pandas as pd
import streamlit as st

import servicios

st.header("Explorador vectorial", anchor=False)
st.caption("Consulta directa a Qdrant Cloud: muestra qué fragmentos recuperaría el RAG y su similitud coseno.")

r = servicios.recuperador()

with st.form("busqueda"):
    consulta = st.text_input("Consulta en lenguaje natural",
                             value="¿Cuál es el esquema de tratamiento farmacológico inicial de la hipertensión?")
    c1, c2, c3, c4 = st.columns(4)
    tipo = c1.selectbox("Tipo de contenido", [None, "documento", "registro_clinico"],
                        format_func=lambda v: "Todos" if v is None else v)
    audiencia = c2.selectbox("Audiencia", [None, "profesionales", "pacientes"],
                             format_func=lambda v: "Todas" if v is None else v)
    fuente = c3.selectbox("Fuente", [None] + sorted(servicios.estado_indice()["por_fuente"]),
                          format_func=lambda v: "Todas" if v is None else v)
    k = c4.slider("Top-k", 1, 20, 8)
    buscar = st.form_submit_button("Buscar", icon=":material/search:", type="primary")

if buscar and consulta:
    frags, ms = r.buscar(consulta, k, r.filtro(tipo, audiencia, fuente))
    if not frags:
        st.info("Sin resultados para los filtros elegidos.")
    else:
        df = pd.DataFrame([{"rank": i, "similitud": round(f.score, 4), "origen": f.origen,
                            "tipo": f.payload.get("tipo_contenido"), "fragmento": f.texto}
                           for i, f in enumerate(frags, start=1)])
        with st.container(horizontal=True):
            st.metric("Latencia Qdrant", f"{ms:.0f} ms", border=True)
            st.metric("Similitud top-1", f"{df['similitud'].iloc[0]:.3f}", border=True)
            st.metric("Similitud media", f"{df['similitud'].mean():.3f}", border=True)
        st.bar_chart(df, x="rank", y="similitud", color="#0f6e8c", height=220,
                     alt="Similitud coseno de cada resultado por posición")
        st.dataframe(df, hide_index=True,
                     column_config={"similitud": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                                    "fragmento": st.column_config.TextColumn(width="large")},
                     alt="Resultados de la búsqueda semántica")
