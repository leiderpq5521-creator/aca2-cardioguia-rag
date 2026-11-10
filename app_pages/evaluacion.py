"""Página 4 · Evaluación: batería de casos de prueba (fidelidad, robustez fuera de contexto y latencia)."""
import pandas as pd
import streamlit as st

import evaluacion
import servicios

st.header("Validación integral del sistema", anchor=False)
st.caption("Casos en dominio (deben responder con citas) y fuera de contexto / inyección de prompt (deben rechazarse).")

st.dataframe(pd.DataFrame(evaluacion.CASOS)[["id", "categoria", "esperado", "pregunta"]], hide_index=True,
             alt="Casos de prueba definidos")

if st.button("Ejecutar batería de pruebas", icon=":material/play_arrow:", type="primary"):
    pipe = servicios.pipeline()
    a = servicios.ajustes()
    if pipe.llm is None:
        st.error(st.session_state.get("error_llm", "LLM no configurado"))
        st.stop()
    with st.spinner(f"Ejecutando {len(evaluacion.CASOS)} casos con {a['modelo']}…"):
        st.session_state.eval_df = evaluacion.ejecutar(pipe, k_docs=a["top_k"], umbral=a["umbral"])

if "eval_df" in st.session_state:
    df = st.session_state.eval_df
    res = evaluacion.resumen(df)
    with st.container(horizontal=True):
        st.metric("Comportamiento correcto", f"{res['aciertos']}/{res['casos']}", border=True)
        st.metric("Respuestas con cita", f"{res['respuestas_con_cita_%']}%", border=True)
        st.metric("Tasa de citación media", f"{res['tasa_citacion_media']:.0%}", border=True)
        st.metric("Citas inventadas", res["citas_inventadas"], border=True)
        st.metric("Latencia media", f"{res['latencia_media_total_ms']/1000:.1f} s", border=True)
    st.dataframe(df.drop(columns=["respuesta"]), hide_index=True,
                 column_config={"OK": st.column_config.CheckboxColumn()}, alt="Resultados por caso")
    st.bar_chart(df.set_index("id")[["qdrant_ms", "llm_ms"]], stack=True, height=240,
                 alt="Latencia por caso separada en búsqueda y generación")
    for _, f in df.iterrows():
        with st.expander(f"{f['id']} · {f['categoria']} · {'✅' if f['OK'] else '❌'}"):
            st.markdown(f"**Pregunta:** {f['pregunta']}")
            st.markdown(f["respuesta"])
    st.download_button("Descargar resultados (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "evaluacion_rag.csv", icon=":material/download:")
