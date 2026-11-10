"""servicios.py — Recursos compartidos (cacheados) entre las páginas de la app Streamlit."""
from __future__ import annotations

import streamlit as st

from rag_core import LLM, Config, PipelineRAG, Recuperador


@st.cache_resource(show_spinner="Conectando con Qdrant Cloud y cargando el modelo de embeddings…")
def recuperador() -> Recuperador:
    return Recuperador(Config())


@st.cache_resource(max_entries=6, show_spinner=False)
def llm(proveedor: str, modelo: str, temperature: float, top_p: float, max_tokens: int) -> LLM:
    return LLM(proveedor, modelo, temperature, top_p, max_tokens)


@st.cache_data(ttl=60, show_spinner=False)
def estado_indice() -> dict:
    r = recuperador()
    est = r.estado()
    est["por_fuente"] = r.conteo_por("fuente")
    est["por_tipo"] = r.conteo_por("tipo_contenido")
    return est


def ajustes() -> dict:
    """Hiperparámetros elegidos en la barra lateral (con valores por defecto del .env)."""
    cfg = Config()
    ss = st.session_state
    return {
        "proveedor": ss.get("proveedor") or cfg.llm_provider,
        "modelo": ss.get("modelo") or cfg.llm_model,
        "temperature": ss.get("temperature", cfg.temperature),
        "top_p": cfg.top_p,
        "max_tokens": ss.get("max_tokens", cfg.max_tokens),
        "top_k": ss.get("top_k", cfg.top_k),
        "umbral": ss.get("umbral", cfg.umbral_similitud),
        "k_registros": ss.get("k_registros", 0),
        "audiencia": ss.get("audiencia", None),
    }


def pipeline() -> PipelineRAG:
    a = ajustes()
    cfg = Config()
    cfg.top_k, cfg.umbral_similitud = a["top_k"], a["umbral"]
    modelo_llm = None
    try:
        modelo_llm = llm(a["proveedor"], a["modelo"], a["temperature"], a["top_p"], a["max_tokens"])
    except Exception as e:  # la app sigue funcionando (gestión documental y explorador) sin LLM
        st.session_state["error_llm"] = str(e)
    else:
        st.session_state.pop("error_llm", None)
    return PipelineRAG(recuperador(), modelo_llm, cfg)
