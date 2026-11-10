"""Página 1 · Chatbot RAG con historial conversacional y fuentes citadas."""
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
    etiqueta = (f"Fuentes · {len(citadas)} citadas de {len(frags)} recuperadas · {lat.get('total_ms', 0)/1000:.1f} s"
                if not meta.get("sin_evidencia") else f"Sin evidencia suficiente · {len(frags)} fragmentos revisados")
    with st.expander(etiqueta, icon=":material/menu_book:"):
        if meta.get("modelo"):
            st.caption(f"Respuesta generada por `{meta['modelo']}`")
        if meta.get("pregunta_busqueda") and meta["pregunta_busqueda"] != meta.get("pregunta"):
            st.caption(f"Consulta reformulada con el historial: «{meta['pregunta_busqueda']}»")
        for f in frags:
            usado = f["ref"] in citadas
            with st.container(border=True):
                st.markdown(f"{':green-badge[citada]' if usado else ':gray-badge[no citada]'} "
                            f"**[{f['ref']}] {f['origen']}** · similitud `{f['score']:.3f}`")
                st.caption(f["apa"])
                st.text(f["texto"])
        if meta.get("citas_invalidas"):
            st.warning(f"Citas sin respaldo detectadas: {', '.join(meta['citas_invalidas'])}", icon=":material/warning:")
        with st.container(horizontal=True):
            st.metric("Búsqueda Qdrant", f"{lat.get('busqueda_qdrant_ms', 0)} ms")
            st.metric("Reformulación", f"{lat.get('condensacion_ms', 0)} ms")
            st.metric("Generación LLM", f"{lat.get('llm_ms', 0)} ms")
            st.metric("Tasa de citación", f"{meta.get('tasa_citacion', 0):.0%}")


# ---------------- Encabezado ----------------
st.header("Chatbot RAG · hipertensión arterial", anchor=False)
st.caption("Respuestas fundamentadas exclusivamente en las guías indexadas en Qdrant Cloud, con cita de fuente y página.")

pipe = servicios.pipeline()
a = servicios.ajustes()
if "error_llm" in st.session_state:
    st.error(st.session_state["error_llm"], title="LLM no disponible", icon=":material/error:")

with st.container(horizontal=True, vertical_alignment="center"):
    st.badge(f"{a['proveedor']} · {a['modelo']}", icon=":material/smart_toy:", color="blue")
    st.badge(f"Top-k {a['top_k']} · umbral {a['umbral']:.2f}", icon=":material/manage_search:", color="gray")
    st.space("stretch")
    if st.session_state.mensajes:
        st.download_button("Exportar chat", json.dumps(st.session_state.mensajes, ensure_ascii=False, indent=2),
                           file_name="conversacion_cardioguia.json", icon=":material/download:")
        if st.button("Nueva conversación", icon=":material/restart_alt:"):
            st.session_state.mensajes = []
            st.rerun()

# ---------------- Historial ----------------
for m in st.session_state.mensajes:
    with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else ":material/cardiology:"):
        st.markdown(m["content"])
        if m.get("meta"):
            mostrar_fuentes(m["meta"])

# ---------------- Entrada ----------------
prompt = st.chat_input("Escribe tu pregunta sobre hipertensión arterial…", submit_mode="disable")
if not st.session_state.mensajes:
    chips = st.empty()
    elegido = chips.pills("Prueba con:", list(SUGERENCIAS), label_visibility="collapsed")
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
            with st.status(":shimmer[Buscando evidencia en las guías]", type="compact") as estado:
                with st.status("Reformulando la consulta con el historial", type="step"):
                    res, mensajes_llm = pipe.preparar(prompt, historial, k_docs=a["top_k"],
                                                      k_registros=a["k_registros"], audiencia=a["audiencia"],
                                                      umbral=a["umbral"])
                    st.write(res.pregunta_busqueda)
                with st.status(f"Recuperados {len(res.fragmentos)} fragmentos de Qdrant Cloud", type="step"):
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
                texto = st.write_stream(pipe.llm.stream(SYSTEM_PROMPT, mensajes_llm))
                res = pipe.finalizar(res, texto, (time.perf_counter() - t0) * 1000)

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
