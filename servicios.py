"""servicios.py — Recursos compartidos (cacheados) y sistema de diseño visual de la app Streamlit."""
from __future__ import annotations

import html
import re
import streamlit as st

from rag_core import LLM, Config, PipelineRAG, Recuperador

ESTILOS_CSS = """
<style>
/* Tipografía y contenedor principal */
.block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1220px !important;
}

/* Encabezado Hero clínico (identidad visual alineada con el póster ACA 2) */
.cg-hero {
    background: linear-gradient(125deg, #0b2f3d 0%, #0f6e8c 65%, #1b9aaa 100%);
    color: #ffffff;
    border-radius: 16px;
    padding: 1.35rem 1.65rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 24px rgba(11, 47, 61, 0.14);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1.2rem;
    flex-wrap: wrap;
}
.cg-hero-kicker {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    color: #bfe9f2;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.cg-hero-title {
    font-size: 1.65rem;
    font-weight: 800;
    line-height: 1.18;
    margin: 0 0 0.3rem 0;
    color: #ffffff !important;
}
.cg-hero-title span {
    color: #ffd166;
}
.cg-hero-sub {
    font-size: 0.92rem;
    color: #e3f6fa;
    margin: 0;
    line-height: 1.4;
}
.cg-hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: center;
}
.cg-chip {
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.32);
    backdrop-filter: blur(6px);
    color: #ffffff;
    border-radius: 999px;
    padding: 0.3rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
}
.cg-chip b {
    color: #ffd166;
}

/* Tarjetas de métricas (KPIs) */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #dde8ee !important;
    border-top: 3px solid #0f6e8c !important;
    border-radius: 12px !important;
    padding: 0.85rem 1rem !important;
    box-shadow: 0 3px 12px rgba(15, 60, 80, 0.05) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(15, 60, 80, 0.09) !important;
}
[data-testid="stMetricLabel"] {
    font-weight: 600 !important;
    color: #4a6572 !important;
}
[data-testid="stMetricValue"] {
    color: #0b4f66 !important;
    font-weight: 800 !important;
}

/* Tarjetas informativas de bienvenida y metadatos */
.cg-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.85rem;
    margin-bottom: 1rem;
}
.cg-info-card {
    background: #ffffff;
    border: 1px solid #dde8ee;
    border-top: 3px solid #1b9aaa;
    border-radius: 12px;
    padding: 0.9rem 1.05rem;
    box-shadow: 0 2px 10px rgba(15, 60, 80, 0.04);
}
.cg-info-card h4 {
    font-size: 0.92rem;
    color: #0b4f66;
    margin: 0 0 0.35rem 0;
    font-weight: 700;
}
.cg-info-card p {
    font-size: 0.83rem;
    color: #3b5360;
    margin: 0;
    line-height: 1.42;
}

/* Mensajes del chat */
[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #e1ecf1;
    border-radius: 14px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 8px rgba(15, 60, 80, 0.04);
}

/* Insignia de cita clínica [F#] */
.cg-cite {
    display: inline-block;
    background: #0f6e8c;
    color: #ffffff !important;
    font-size: 0.76rem;
    font-weight: 700;
    padding: 0.05rem 0.42rem;
    border-radius: 6px;
    margin: 0 0.1rem;
    vertical-align: baseline;
    letter-spacing: 0.3px;
}

/* Tarjeta de fragmento de evidencia */
.cg-source-card {
    background: #ffffff;
    border: 1px solid #dce7ed;
    border-left: 4px solid #94a3b8;
    border-radius: 10px;
    padding: 0.75rem 0.95rem;
    margin-bottom: 0.6rem;
}
.cg-source-card.citada {
    border-left-color: #198754;
    background: #f8fdfb;
}
.cg-source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.3rem;
}
.cg-badge-ok {
    background: #d1e7dd;
    color: #0f5132;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.12rem 0.5rem;
    border-radius: 999px;
    text-transform: uppercase;
}
.cg-badge-neutral {
    background: #e2e8f0;
    color: #475569;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.12rem 0.5rem;
    border-radius: 999px;
}
.cg-apa {
    font-size: 0.78rem;
    color: #526b77;
    font-style: italic;
    margin-bottom: 0.4rem;
}
.cg-frag-text {
    font-size: 0.83rem;
    color: #1e293b;
    background: #f1f6f9;
    border-radius: 8px;
    padding: 0.55rem 0.75rem;
    line-height: 1.45;
    white-space: pre-wrap;
}

/* Barra lateral */
[data-testid="stSidebar"] {
    border-right: 1px solid #dbe7ed;
}
.cg-sidebar-brand {
    background: linear-gradient(135deg, #0b2f3d 0%, #0f6e8c 100%);
    color: #ffffff;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.85rem;
}
.cg-sidebar-brand h3 {
    color: #ffffff !important;
    font-size: 1.2rem;
    font-weight: 800;
    margin: 0 0 0.2rem 0;
}
.cg-sidebar-brand p {
    color: #cbeaf2;
    font-size: 0.78rem;
    margin: 0;
    line-height: 1.35;
}
</style>
"""


def inyectar_estilos() -> None:
    st.markdown(ESTILOS_CSS, unsafe_allow_html=True)


def encabezado_pagina(titulo: str, subtitulo: str, resaltar: str = "RAG",
                      chips: list[tuple[str, str]] | None = None) -> None:
    """Renderiza un banner hero clínico unificado en la parte superior de cada página."""
    titulo_html = html.escape(titulo)
    if resaltar and resaltar in titulo_html:
        titulo_html = titulo_html.replace(resaltar, f"<span>{html.escape(resaltar)}</span>", 1)
    chips_html = ""
    if chips:
        items = "".join(
            f'<div class="cg-chip">{html.escape(k)}: <b>{html.escape(str(v))}</b></div>' for k, v in chips
        )
        chips_html = f'<div class="cg-hero-chips">{items}</div>'
    st.markdown(
        f"""
        <div class="cg-hero">
            <div>
                <div class="cg-hero-kicker">CardioGuía · Asistente Clínico Basado en Evidencia · MinSalud 2025 &amp; OPS HEARTS</div>
                <div class="cg-hero-title">{titulo_html}</div>
                <p class="cg-hero-sub">{html.escape(subtitulo)}</p>
            </div>
            {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def embellecer_citas(texto: str) -> str:
    """Transforma referencias [F1] o [F2, F5] en insignias visuales dentro del Markdown."""
    def _reemplazar(m: re.Match) -> str:
        refs = re.findall(r"F\d+", m.group(1))
        return " ".join(f'<span class="cg-cite">[{r}]</span>' for r in refs)

    return re.sub(r"\[(F\d+(?:\s*[,;]\s*F\d+)*)\]", _reemplazar, texto)


@st.cache_resource(show_spinner="Conectando con Qdrant Cloud y cargando el modelo de embeddings…")
def _recuperador() -> Recuperador:
    return Recuperador(Config())


def recuperador() -> Recuperador:
    """Recuperador compartido; si faltan credenciales muestra un aviso claro en lugar de un error técnico."""
    try:
        return _recuperador()
    except ValueError as e:
        st.error(f"{e} En Streamlit Cloud agréguelas en *Manage app → Settings → Secrets* "
                 "(ver `.streamlit/secrets.toml.example`); en local, en el archivo `.env`.",
                 title="Faltan credenciales de Qdrant", icon=":material/key_off:")
        st.stop()


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


@st.cache_data(ttl=120, show_spinner=False)
def fragmentos_fuente(fuente: str, limite: int = 500) -> list[dict]:
    return recuperador().fragmentos_de(fuente, limite=limite)


def limpiar_cache_indice() -> None:
    estado_indice.clear()
    fragmentos_fuente.clear()


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
