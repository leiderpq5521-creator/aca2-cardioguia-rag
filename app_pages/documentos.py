"""Página 2 · Gestión documental: estado del índice, exploración de documentos/fragmentos, carga y eliminación."""
import pandas as pd
import streamlit as st

import servicios
from ingesta import pdf_a_chunks

st.header("Gestión documental y del índice vectorial", anchor=False)
st.caption("Supervisión de la colección de Qdrant Cloud construida en el ACA 1 y administración del corpus.")

r = servicios.recuperador()
est = servicios.estado_indice()

# ---------------- KPIs del índice ----------------
with st.container(horizontal=True):
    st.metric("Puntos indexados", f"{est['puntos']:,}".replace(",", "."), border=True)
    st.metric("Documentos PDF", len([f for f in est["por_fuente"] if f.lower().endswith(".pdf")]), border=True)
    st.metric("Dimensión", f"{est['dimension']} d", border=True)
    st.metric("Métrica", est["metrica"].capitalize(), border=True)
    st.metric("Estado", est["estado"].upper(), border=True)

if est["dimension"] != est["dimension_modelo"]:
    st.error("La dimensión del modelo de embeddings no coincide con la colección.", icon=":material/error:")

col1, col2 = st.columns([3, 2])
with col1:
    df_f = (pd.DataFrame([{"fuente": k, "fragmentos": v} for k, v in est["por_fuente"].items()])
            .sort_values("fragmentos", ascending=False))
    st.subheader("Fragmentos por fuente", anchor=False)
    st.bar_chart(df_f, x="fuente", y="fragmentos", horizontal=True, color="#0f6e8c", height=220,
                 alt="Número de fragmentos indexados por cada fuente documental")
with col2:
    st.subheader("Composición", anchor=False)
    st.dataframe(pd.DataFrame([{"tipo de contenido": k, "puntos": v} for k, v in est["por_tipo"].items()]),
                 hide_index=True, alt="Puntos por tipo de contenido")
    st.caption("Índices de payload: " + ", ".join(f"`{c}`" for c in est["indices_payload"]))
    if st.button("Actualizar estado", icon=":material/refresh:"):
        servicios.estado_indice.clear()
        st.rerun()

st.divider()

# ---------------- Explorar un documento ----------------
st.subheader("Explorar documento y metadatos", anchor=False)
fuente = st.selectbox("Fuente", sorted(est["por_fuente"], key=lambda f: (not f.endswith(".pdf"), f)))
if fuente:
    filas = r.fragmentos_de(fuente)
    if filas:
        meta = {k: v for k, v in filas[0].items()
                if k in ("titulo", "entidad", "anio", "audiencia", "tipo_documento", "tipo_contenido", "fuente")}
        st.table(pd.Series({k: str(v) for k, v in meta.items()}, name="valor").to_frame())
        df = pd.DataFrame(filas)
        cols = [c for c in ("pagina", "chunk_en_pagina", "id_registro", "edad", "sexo", "colesterol",
                            "presion_reposo", "diagnostico", "texto") if c in df.columns]
        busq = st.text_input("Filtrar fragmentos por texto", placeholder="p. ej. amlodipino, sodio, 140/90…")
        if busq:
            df = df[df["texto"].str.contains(busq, case=False, na=False)]
        st.dataframe(df[cols], hide_index=True, height=360,
                     column_config={"texto": st.column_config.TextColumn("fragmento", width="large")},
                     alt=f"Fragmentos indexados de {fuente}")
        st.caption(f"{len(df)} fragmentos mostrados (máx. 500 por fuente).")

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
            if not archivo or not titulo:
                st.warning("Selecciona un PDF y escribe su título.")
            else:
                with st.status("Procesando documento…") as s:
                    chunks, resumen = pdf_a_chunks(archivo.name, archivo.getvalue(), titulo, entidad, anio,
                                                   audiencia, tipo)
                    st.write(resumen)
                    if chunks:
                        n = r.indexar(chunks)
                        s.update(label=f"{n} fragmentos indexados en Qdrant Cloud", state="complete")
                        servicios.estado_indice.clear()
                    else:
                        s.update(label="El PDF no tiene texto extraíble (¿es escaneado?)", state="error")

with c_del:
    with st.container(border=True):
        st.subheader(":material/delete: Eliminar documento del índice", anchor=False)
        pdfs = sorted(f for f in est["por_fuente"] if f.lower().endswith(".pdf"))
        borrar = st.selectbox("Documento", pdfs, key="borrar")
        confirmar = st.checkbox("Confirmo que deseo eliminar todos sus vectores (acción irreversible)")
        if st.button("Eliminar", icon=":material/delete_forever:", disabled=not confirmar):
            n = r.eliminar_fuente(borrar)
            servicios.estado_indice.clear()
            st.success(f"Se eliminaron {n} fragmentos de «{borrar}». Puede volver a indexarlo desde el panel izquierdo.")
