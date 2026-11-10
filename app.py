"""
app.py — CardioGuía · Chatbot RAG de hipertensión arterial (ACA 2 · Computación en la Nube · CUN)

Ejecutar:  streamlit run app.py
Páginas:   Chatbot RAG · Gestión documental · Explorador vectorial · Evaluación
"""
import streamlit as st

from rag_core import MODELOS_POR_DEFECTO, Config

st.set_page_config(page_title="CardioGuía · RAG Salud", page_icon=":material/cardiology:", layout="wide")

cfg = Config()

# ---------------- Barra lateral: estado del sistema + hiperparámetros ----------------
with st.sidebar:
    st.title(":material/cardiology: CardioGuía")
    st.caption("Chatbot RAG sobre guías de hipertensión arterial · MinSalud 2025 · OPS HEARTS")

    with st.expander("Configuración del modelo", icon=":material/tune:", expanded=False):
        proveedores = ["gemini", "groq", "ollama"]
        if "modelo" not in st.session_state:
            st.session_state.modelo = cfg.llm_model

        def _cambiar_proveedor():
            p = st.session_state.proveedor or cfg.llm_provider
            st.session_state.modelo = cfg.llm_model if p == cfg.llm_provider else MODELOS_POR_DEFECTO[p]

        st.segmented_control("Proveedor LLM", proveedores, key="proveedor", on_change=_cambiar_proveedor,
                             default=cfg.llm_provider if cfg.llm_provider in proveedores else "gemini",
                             help="gemini/groq = API cloud · ollama = modelo abierto local (Gemma/Llama)")
        st.text_input("Modelo", key="modelo")
        st.slider("Temperatura", 0.0, 1.0, cfg.temperature, 0.05, key="temperature",
                  help="Baja (0.1–0.3) = respuestas deterministas y fieles al contexto clínico.")
        st.slider("Máx. tokens de salida", 512, 4096, cfg.max_tokens, 256, key="max_tokens",
                  help="En Gemini 3 incluye los tokens de razonamiento interno; 2048 evita respuestas truncadas.")

    with st.expander("Recuperación (Top-k)", icon=":material/manage_search:", expanded=False):
        st.slider("Fragmentos de guías (k)", 1, 10, cfg.top_k, key="top_k")
        st.slider("Umbral mínimo de similitud", 0.0, 0.8, cfg.umbral_similitud, 0.05, key="umbral",
                  help="Si ningún fragmento supera este valor, el sistema no llama al LLM y declara que no hay evidencia.")
        st.slider("Casos del dataset UCI a incluir", 0, 5, 0, key="k_registros",
                  help="Agrega registros anónimos de pacientes (UCI Heart Disease) como ejemplos ilustrativos.")
        st.selectbox("Audiencia de las guías", [None, "profesionales", "pacientes"], key="audiencia",
                     format_func=lambda v: "Todas" if v is None else v.capitalize())

    with st.expander("Credenciales (.env)", icon=":material/key:"):
        for k, v in cfg.resumen_seguro().items():
            st.markdown(f"**{k}:** `{v}`")
        st.caption("Las claves se leen de variables de entorno o `st.secrets`; nunca se muestran ni se guardan en el código.")

pages = [
    st.Page("app_pages/chat.py", title="Chatbot RAG", icon=":material/chat:", default=True),
    st.Page("app_pages/documentos.py", title="Gestión documental", icon=":material/folder_open:"),
    st.Page("app_pages/explorador.py", title="Explorador vectorial", icon=":material/travel_explore:"),
    st.Page("app_pages/evaluacion.py", title="Evaluación", icon=":material/fact_check:"),
]
st.navigation(pages, position="top").run()
