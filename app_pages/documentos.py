"""Página 2 · Gestión documental: estado del índice, exploración de documentos/fragmentos, carga y eliminación."""
import html
import altair as alt
import pandas as pd
import streamlit as st

import servicios
from ingesta import pdf_a_chunks

r = servicios.recuperador()
est = servicios.estado_indice()
n_pdfs = len([f for f in est["por_fuente"] if f.lower().endswith(".pdf")])

servicios.encabezado_pagina(
    titulo="Gestión Documental y del Índice Vectorial",
    subtitulo="Supervisión en tiempo real de la colección serverless en Qdrant Cloud (ACA 1) y administración de guías clínicas.",
    resaltar="Índice Vectorial",
    chips=[
        ("Colección", r.cfg.collection),
        ("Vectores", f"{est['puntos']:,}".replace(",", ".")),
        ("Estado", est["estado"].upper()),
    ],
)

# ---------------- KPIs del índice ----------------
with st.container(horizontal=True):
    st.metric("Puntos indexados", f"{est['puntos']:,}".replace(",", "."), border=True)
    st.metric("Guías clínicas (PDF)", f"{n_pdfs} docs", border=True)
    st.metric("Dimensión embedding", f"{est['dimension']} d", border=True)
    st.metric("Métrica de distancia", est["metrica"].capitalize(), border=True)
    st.metric("Estado del clúster", est["estado"].upper(), border=True)

if est["dimension"] != est["dimension_modelo"]:
    st.error("La dimensión del modelo de embeddings no coincide con la colección.", icon=":material/error:")

col1, col2 = st.columns([3, 2])
with col1:
    with st.container(border=True):
        st.subheader(":material/bar_chart: Fragmentos indexados por fuente", anchor=False)
        df_f = (pd.DataFrame([
            {"fuente": k, "fragmentos": v, "clase": "Guía PDF" if k.lower().endswith(".pdf") else "Dataset clínico"}
            for k, v in est["por_fuente"].items()
        ]).sort_values("fragmentos", ascending=False))
        if not df_f.empty:
            base = alt.Chart(df_f).encode(
                y=alt.Y("fuente:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260, labelFontWeight="bold")),
                x=alt.X("fragmentos:Q", title="Fragmentos (chunks)"),
                color=alt.Color(
                    "clase:N",
                    scale=alt.Scale(domain=["Guía PDF", "Dataset clínico"], range=["#0f6e8c", "#2a9d8f"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["fuente:N", "fragmentos:Q", "clase:N"],
            )
            barras = base.mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, height=28)
            textos = base.mark_text(align="left", dx=6, fontWeight="bold", color="#14232e").encode(text="fragmentos:Q")
            st.altair_chart((barras + textos).properties(height=210), width="stretch")

with col2:
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(":material/donut_small: Composición", anchor=False)
            st.space("stretch")
            if st.button("Actualizar", icon=":material/refresh:"):
                servicios.limpiar_cache_indice()
                st.rerun()
        df_t = pd.DataFrame([{"tipo": k, "puntos": v} for k, v in est["por_tipo"].items()])
        if not df_t.empty:
            donut = alt.Chart(df_t).mark_arc(innerRadius=48, outerRadius=78, cornerRadius=4).encode(
                theta=alt.Theta("puntos:Q"),
                color=alt.Color("tipo:N", scale=alt.Scale(range=["#0f6e8c", "#f4a261"]),
                                legend=alt.Legend(orient="right", title="Tipo")),
                tooltip=["tipo:N", "puntos:Q"],
            ).properties(height=165)
            st.altair_chart(donut, width="stretch")
        st.caption("Índices de payload activos: " + ", ".join(f"`{c}`" for c in est["indices_payload"]))

st.divider()

# ---------------- Explorar un documento ----------------
st.subheader(":material/library_books: Explorar documento y metadatos", anchor=False)
fuentes_ordenadas = sorted(est["por_fuente"], key=lambda f: (not f.endswith(".pdf"), f))
fuente = st.selectbox("Selecciona una fuente documental para inspeccionar sus vectores", fuentes_ordenadas) if fuentes_ordenadas else None
if fuente:
    filas = servicios.fragmentos_fuente(fuente)
    if filas:
        p0 = filas[0]
        if p0.get("tipo_contenido") == "documento":
            st.markdown(
                f"""
                <div class="cg-card-grid">
                    <div class="cg-info-card"><h4>📄 Título</h4><p>{html.escape(str(p0.get('titulo', fuente)))}</p></div>
                    <div class="cg-info-card"><h4>🏛️ Entidad emisora</h4><p>{html.escape(str(p0.get('entidad', '—')))} ({html.escape(str(p0.get('anio', 's.f.')) )})</p></div>
                    <div class="cg-info-card"><h4>👥 Audiencia y Tipo</h4><p>Audiencia: <b>{html.escape(str(p0.get('audiencia', '—')))}</b> · Tipo: <code>{html.escape(str(p0.get('tipo_documento', '—')))}</code></p></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        df = pd.DataFrame(filas)
        cols = [c for c in ("pagina", "chunk_en_pagina", "id_registro", "edad", "sexo", "colesterol",
                            "presion_reposo", "diagnostico", "texto") if c in df.columns]
        busq = st.text_input("Filtrar fragmentos por palabra clave o cifra clínica",
                             placeholder="p. ej. amlodipino, losartán, sodio, 140/90, MAPA…",
                             icon=":material/search:")
        if busq:
            df = df[df["texto"].str.contains(busq, case=False, na=False, regex=False)]
        st.dataframe(df[cols], hide_index=True, height=360,
                     column_config={"texto": st.column_config.TextColumn("Fragmento indexado (chunk)", width="large")},
                     alt=f"Fragmentos indexados de {fuente}")
        st.caption(f"Mostrando **{len(df)}** de **{len(filas)}** fragmentos recuperados para `{fuente}`.")

st.divider()

# ---------------- Cargar / eliminar ----------------
c_up, c_del = st.columns(2)
with c_up:
    with st.container(border=True):
        st.subheader(":material/upload_file: Indexar nuevo PDF", anchor=False)
        with st.form("carga", clear_on_submit=True):
            archivo = st.file_uploader("Documento PDF", type=["pdf"])
            titulo = st.text_input("Título")
            entidad = st.text_input("Entidad emisora", value="Ministerio de Salud y Protección Social (Colombia)")
            c1, c2, c3 = st.columns(3)
            anio = c1.number_input("Año", 1990, 2030, 2025)
            audiencia = c2.selectbox("Audiencia", ["profesionales", "pacientes"])
            tipo = c3.selectbox("Tipo", ["guia_clinica", "guia_pacientes", "protocolo_internacional", "otro"])
            enviar = st.form_submit_button("Vectorizar e indexar", icon=":material/cloud_upload:", type="primary")
        if enviar:
            if not archivo or not titulo.strip():
                st.warning("Selecciona un PDF y escribe su título.")
            else:
                with st.status("Procesando documento…") as s:
                    chunks, resumen = pdf_a_chunks(archivo.name, archivo.getvalue(), titulo.strip(), entidad.strip(),
                                                   anio, audiencia, tipo)
                    st.write(resumen)
                    if chunks:
                        n = r.indexar(chunks)
                        s.update(label=f"{n} fragmentos indexados en Qdrant Cloud", state="complete")
                        servicios.limpiar_cache_indice()
                    else:
                        s.update(label="El PDF no tiene texto extraíble (¿es escaneado?)", state="error")

with c_del:
    with st.container(border=True):
        st.subheader(":material/delete: Eliminar documento del índice", anchor=False)
        pdfs = sorted(f for f in est["por_fuente"] if f.lower().endswith(".pdf"))
        borrar = st.selectbox("Documento", pdfs, key="borrar") if pdfs else None
        confirmar = st.checkbox("Confirmo que deseo eliminar todos sus vectores (acción irreversible)",
                                disabled=not borrar)
        if st.button("Eliminar", icon=":material/delete_forever:", disabled=not (borrar and confirmar)):
            n = r.eliminar_fuente(borrar)
            servicios.limpiar_cache_indice()
            st.success(f"Se eliminaron {n} fragmentos de «{borrar}». Puede volver a indexarlo desde el panel izquierdo.")
