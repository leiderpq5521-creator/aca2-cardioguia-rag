"""Página 1 · Chatbot RAG con historial conversacional y fuentes citadas."""
import html
import json
import time

import streamlit as st

import servicios
from rag_core import SYSTEM_PROMPT

SUGERENCIAS = {
    ":material/monitor_heart: Diagnóstico": "¿A partir de qué cifras de presión arterial se diagnostica hipertensión?",
    ":material/medication: Tratamiento": "¿Cuáles son los medicamentos de primera línea para tratar la hipertensión?",
    ":material/directions_run: Estilo de vida": "¿Qué cambios en el estilo de vida ayudan a bajar la presión arterial?",
    ":material/stethoscope: Medición": "¿Cómo se debe medir correctamente la presión arterial en el consultorio?",
    ":material/clinical_notes: Caso HEARTS": ("Un hombre de 60 años tiene 160/100 mm Hg y colesterol alto: ¿qué tratamiento "
                                              "inicial propone el protocolo HEARTS y cada cuánto hacer seguimiento?"),
}

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []


def mostrar_fuentes(meta: dict):
    """Panel de evidencia: fragmentos recuperados, cuáles se citaron y métricas de la respuesta."""
    frags = meta.get("fragmentos", [])
    lat = meta.get("latencia", {})
    citadas = set(meta.get("citas_validas", []))
    etiqueta = (f"Evidencia clínica · {len(citadas)} citadas de {len(frags)} recuperadas · {lat.get('total_ms', 0)/1000:.1f} s"
                if not meta.get("sin_evidencia") else f"Guarda activada (sin evidencia suficiente) · {len(frags)} fragmentos revisados")
    with st.expander(etiqueta, icon=":material/menu_book:"):
        with st.container(horizontal=True):
            st.metric("Búsqueda Qdrant", f"{lat.get('busqueda_qdrant_ms', 0)} ms")
            st.metric("Reformulación", f"{lat.get('condensacion_ms', 0)} ms")
            st.metric("Generación LLM", f"{lat.get('llm_ms', 0)} ms")
            st.metric("Tasa de citación", f"{meta.get('tasa_citacion', 0):.0%}")

        if meta.get("modelo"):
            st.caption(f"Modelo generador: `{meta['modelo']}`")
        if meta.get("pregunta_busqueda") and meta["pregunta_busqueda"] != meta.get("pregunta"):
            st.info(f"**Consulta autónoma reformulada con el historial:** «{meta['pregunta_busqueda']}»",
                    icon=":material/history:")
        if meta.get("citas_invalidas"):
            st.warning(f"Citas sin respaldo detectadas: {', '.join(meta['citas_invalidas'])}", icon=":material/warning:")

        for f in frags:
            usado = f["ref"] in citadas
            clase = "cg-source-card citada" if usado else "cg-source-card"
            badge = '<span class="cg-badge-ok">✓ Citada en respuesta</span>' if usado else '<span class="cg-badge-neutral">Contexto de apoyo</span>'
            st.markdown(
                f"""
                <div class="{clase}">
                    <div class="cg-source-header">
                        <div>{badge} &nbsp; <b>[{html.escape(f['ref'])}] {html.escape(f['origen'])}</b></div>
                        <div class="cg-badge-neutral">Similitud coseno: <b>{f['score']:.3f}</b></div>
                    </div>
                    <div class="cg-apa">📖 {html.escape(f['apa'])}</div>
                    <div class="cg-frag-text">{html.escape(f['texto'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------- Encabezado Hero ----------------
pipe = servicios.pipeline()
a = servicios.ajustes()
modelo_activo = getattr(pipe.llm, "modelo", None) or a["modelo"]

servicios.encabezado_pagina(
    titulo="Chatbot RAG · Hipertensión Arterial y Riesgo Cardiovascular",
    subtitulo="Respuestas clínicas fundamentadas exclusivamente en las guías indexadas en Qdrant Cloud, con verificación automática de citas y página.",
    resaltar="RAG",
    chips=[
        ("LLM", f"{a['proveedor'].upper()} · {modelo_activo}"),
        ("Top-k", str(a["top_k"])),
        ("Umbral", f"{a['umbral']:.2f}"),
        ("Audiencia", "Todas" if not a["audiencia"] else a["audiencia"].capitalize()),
    ],
)

if "error_llm" in st.session_state:
    st.error(st.session_state["error_llm"], title="LLM no disponible", icon=":material/error:")

prompt_sugerido = None
if st.session_state.mensajes:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(f"💬 **{len(st.session_state.mensajes) // 2}** consultas en esta sesión conversacional")
        st.space("stretch")
        with st.popover("Consultas de ejemplo", icon=":material/lightbulb:"):
            for etq, preg in SUGERENCIAS.items():
                if st.button(etq, key=f"sug_{etq}", width="stretch"):
                    prompt_sugerido = preg
        st.download_button("Exportar historial (JSON)",
                           json.dumps(st.session_state.mensajes, ensure_ascii=False, indent=2),
                           file_name="conversacion_cardioguia.json", icon=":material/download:")
        if st.button("Nueva conversación", icon=":material/restart_alt:", type="primary"):
            st.session_state.mensajes = []
            st.rerun()

# ---------------- Estado inicial (tarjetas informativas) ----------------
if not st.session_state.mensajes:
    st.markdown(
        """
        <div class="cg-card-grid">
            <div class="cg-info-card">
                <h4>📚 Corpus Oficial Indexado (Qdrant Cloud)</h4>
                <p>GPC de Hipertensión Arterial Primaria <b>MinSalud Colombia (2025)</b> para profesionales y para pacientes, protocolo <b>HEARTS (OPS/OMS)</b> y casos ilustrativos <b>UCI Heart Disease</b>.</p>
            </div>
            <div class="cg-info-card">
                <h4>🛡️ Mitigación Multicapa de Alucinaciones</h4>
                <p>Ventana de contexto con chunks vecinos, guarda por umbral coseno (≥ 0.40), temperatura baja (0.2) y bloqueo de inyección de prompt o temas fuera de dominio.</p>
            </div>
            <div class="cg-info-card">
                <h4>📑 Trazabilidad Clínica y Citas APA 7</h4>
                <p>Cada afirmación incluye su identificador <b>[F#]</b> verificado contra los fragmentos recuperados, indicando documento, página exacta y referencia bibliográfica.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Historial ----------------
for m in st.session_state.mensajes:
    with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else ":material/cardiology:"):
        contenido = servicios.embellecer_citas(m["content"]) if m["role"] == "assistant" else m["content"]
        st.markdown(contenido, unsafe_allow_html=m["role"] == "assistant")
        if m.get("meta"):
            mostrar_fuentes(m["meta"])

# ---------------- Entrada ----------------
prompt = st.chat_input("Escribe tu pregunta clínica sobre diagnóstico, medición o tratamiento de HTA…",
                       submit_mode="disable") or prompt_sugerido
if not st.session_state.mensajes:
    chips = st.empty()
    elegido = chips.pills(" Selecciona una consulta clínica de prueba:", list(SUGERENCIAS))
    if elegido:
        prompt = SUGERENCIAS[elegido]
    if prompt:
        chips.empty()

if prompt:
    historial = [{"role": m["role"], "content": m["content"]} for m in st.session_state.mensajes]
    st.session_state.mensajes.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=":material/cardiology:"):
        try:
            with st.status(":shimmer[Buscando evidencia en las guías clínicas…]", type="compact") as estado:
                res, mensajes_llm = pipe.preparar(prompt, historial, k_docs=a["top_k"],
                                                  k_registros=a["k_registros"], audiencia=a["audiencia"],
                                                  umbral=a["umbral"])
                if res.pregunta_busqueda != prompt:
                    with st.status("Consulta reformulada con el historial", type="step", state="complete"):
                        st.write(res.pregunta_busqueda)
                with st.status(f"Recuperados {len(res.fragmentos)} fragmentos de Qdrant Cloud",
                               type="step", state="complete"):
                    for f in res.fragmentos:
                        st.caption(f"[{f.ref}] {f.origen} · sim {f.score:.3f}")
                estado.update(label=f"Evidencia recuperada en {res.latencia['busqueda_qdrant_ms']} ms",
                              state="complete")

            if res.sin_evidencia:
                st.markdown(res.respuesta)
                res = pipe.finalizar(res, res.respuesta, 0)
            elif pipe.llm is None:
                raise RuntimeError(st.session_state.get("error_llm", "LLM no configurado"))
            else:
                t0 = time.perf_counter()
                contenedor_stream = st.empty()
                texto = contenedor_stream.write_stream(pipe.llm.stream(SYSTEM_PROMPT, mensajes_llm))
                res = pipe.finalizar(res, texto, (time.perf_counter() - t0) * 1000)
                contenedor_stream.markdown(servicios.embellecer_citas(res.respuesta), unsafe_allow_html=True)

            meta = {
                "pregunta": res.pregunta, "pregunta_busqueda": res.pregunta_busqueda,
                "sin_evidencia": res.sin_evidencia, "latencia": res.latencia,
                "citas_validas": res.citas_validas, "citas_invalidas": res.citas_invalidas,
                "tasa_citacion": res.tasa_citacion,
                "modelo": None if res.sin_evidencia or pipe.llm is None else pipe.llm.modelo,
                "fragmentos": [{"ref": f.ref, "score": f.score, "origen": f.origen, "apa": f.cita_apa,
                                "texto": f.texto} for f in res.fragmentos],
            }
            mostrar_fuentes(meta)
            st.session_state.mensajes.append({"role": "assistant", "content": res.respuesta, "meta": meta})
        except Exception as e:
            st.error(f"No fue posible generar la respuesta: {e}", icon=":material/error:")
            st.session_state.mensajes.pop()  # se descarta la pregunta sin respuesta
