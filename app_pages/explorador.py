"""Página 3 · Explorador vectorial: búsqueda semántica directa (sin LLM) con filtros por metadatos."""
import html
import altair as alt
import pandas as pd
import streamlit as st

import servicios

r = servicios.recuperador()
a = servicios.ajustes()

servicios.encabezado_pagina(
    titulo="Explorador del Espacio Vectorial (384d)",
    subtitulo="Consulta semántica directa en Qdrant Cloud sin pasar por el LLM: inspecciona la similitud coseno y el umbral de evidencia.",
    resaltar="Espacio Vectorial",
    chips=[
        ("Embeddings", "MiniLM-L12-v2 (384d)"),
        ("Métrica", "Coseno"),
        ("Umbral activo", f"{a['umbral']:.2f}"),
    ],
)

with st.form("busqueda"):
    consulta = st.text_input("Consulta en lenguaje natural",
                             value="¿Cuál es el esquema de tratamiento farmacológico inicial de la hipertensión?")
    c1, c2, c3, c4 = st.columns(4)
    tipo = c1.selectbox("Tipo de contenido", [None, "documento", "registro_clinico"],
                        format_func=lambda v: "Todos" if v is None else v)
    audiencia = c2.selectbox("Audiencia", [None, "profesionales", "pacientes"],
                             format_func=lambda v: "Todas" if v is None else v.capitalize())
    fuente = c3.selectbox("Fuente documental", [None] + sorted(servicios.estado_indice()["por_fuente"]),
                          format_func=lambda v: "Todas" if v is None else v)
    k = c4.slider("Top-k resultados", 1, 20, 8)
    buscar = st.form_submit_button("Ejecutar búsqueda vectorial", icon=":material/search:", type="primary")

if consulta:
    frags, ms = r.buscar(consulta, k, r.filtro(tipo, audiencia, fuente))
    if not frags:
        st.info("Sin resultados para los filtros elegidos.", icon=":material/info:")
    else:
        umbral = a["umbral"]
        df = pd.DataFrame([{
            "pos": f"#{i}", "rank": i, "similitud": round(f.score, 4), "origen": f.origen,
            "estado": "Supera umbral" if f.score >= umbral else "Bajo umbral",
            "tipo": f.payload.get("tipo_contenido"), "apa": f.cita_apa, "fragmento": f.texto,
        } for i, f in enumerate(frags, start=1)])

        superan = int((df["similitud"] >= umbral).sum())
        with st.container(horizontal=True):
            st.metric("Latencia Qdrant Cloud", f"{ms:.0f} ms", border=True)
            st.metric("Similitud Top-1", f"{df['similitud'].iloc[0]:.3f}", border=True)
            st.metric("Similitud media", f"{df['similitud'].mean():.3f}", border=True)
            st.metric("Superan umbral", f"{superan} / {len(df)}", border=True)

        with st.container(border=True):
            st.subheader(":material/analytics: Perfil de similitud coseno vs. umbral anti-alucinación", anchor=False)
            barras = alt.Chart(df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=36).encode(
                x=alt.X("pos:N", sort=alt.EncodingSortField(field="rank", order="ascending"),
                        title="Posición (Top-k)", axis=alt.Axis(labelAngle=0, labelFontWeight="bold")),
                y=alt.Y("similitud:Q", title="Similitud coseno", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "estado:N",
                    scale=alt.Scale(domain=["Supera umbral", "Bajo umbral"], range=["#0f6e8c", "#e76f51"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["pos:N", "origen:N", "similitud:Q", "estado:N"],
            )
            etiq = alt.Chart(df).mark_text(dy=-8, fontWeight="bold", color="#14232e").encode(
                x=alt.X("pos:N", sort=alt.EncodingSortField(field="rank", order="ascending")),
                y="similitud:Q",
                text=alt.Text("similitud:Q", format=".3f"),
            )
            regla = alt.Chart(pd.DataFrame([{"umbral": umbral}])).mark_rule(
                color="#d9534f", strokeDash=[6, 4], strokeWidth=2
            ).encode(y="umbral:Q")
            st.altair_chart((barras + etiq + regla).properties(height=240), width="stretch")

        st.dataframe(
            df[["rank", "similitud", "estado", "origen", "tipo", "fragmento"]],
            hide_index=True,
            column_config={
                "similitud": st.column_config.ProgressColumn("Similitud coseno", min_value=0, max_value=1, format="%.3f"),
                "fragmento": st.column_config.TextColumn("Fragmento recuperado", width="large"),
            },
            alt="Resultados de la búsqueda semántica",
        )

        with st.expander("Inspeccionar tarjetas detalladas de los fragmentos recuperados", icon=":material/Description:"):
            for fila in df.to_dict("records"):
                pasa = fila["similitud"] >= umbral
                clase = "cg-source-card citada" if pasa else "cg-source-card"
                badge = '<span class="cg-badge-ok">≥ Umbral</span>' if pasa else '<span class="cg-badge-neutral">&lt; Umbral</span>'
                st.markdown(
                    f"""
                    <div class="{clase}">
                        <div class="cg-source-header">
                            <div>{badge} &nbsp; <b>[{fila['pos']}] {html.escape(fila['origen'])}</b></div>
                            <div class="cg-badge-neutral">Similitud: <b>{fila['similitud']:.4f}</b></div>
                        </div>
                        <div class="cg-apa">📖 {html.escape(fila['apa'])}</div>
                        <div class="cg-frag-text">{html.escape(fila['fragmento'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
