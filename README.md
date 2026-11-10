# CardioGuía · Chatbot RAG sobre hipertensión arterial

**ACA 2 · Computación en la Nube para Sistemas Inteligentes · Especialización en IA · CUN**
Docente: Dr. Hamilton Rivera Flor · Integrantes: Kevin David Guzmán Payares, Leider Antonio Pacheco Quinto, Joaquín Pablo Redondo Tapias, José Luis Ruiz Ospino

Chatbot con **RAG (Retrieval-Augmented Generation)** que responde preguntas sobre diagnóstico, medición, tratamiento y control de la hipertensión arterial **citando documento y página** de las guías oficiales. Continúa el **ACA 1**, que construyó la base vectorial en **Qdrant Cloud**.

🎥 Video de sustentación: _enlace_ · 📄 Póster: `Poster_ACA2_RAG_Salud.pdf` · 📓 Notebook: `ACA2_Computacion_Nube_RAG_Salud.ipynb`

---

## Arquitectura

```
Usuario ─► Streamlit (app.py) ─► PipelineRAG (rag_core.py)
                                   │ 1. reformula la pregunta con el historial
                                   │ 2. embedding (MiniLM multilingüe, 384 d)
                                   ├─► Qdrant Cloud (colección del ACA 1) ─► Top-k fragmentos [F1..Fk]
                                   │      + ventana de contexto (chunk anterior y siguiente de la misma página)
                                   │ 3. si la similitud < umbral → "sin evidencia" (no llama al LLM)
                                   │ 4. system prompt + contexto numerado
                                   ├─► LLM: Gemini 3.5 Flash (API) | Groq Llama 3.3 | Ollama Gemma 3 (local)
                                   │ 5. verificación de citas [F#]
                                   ▼
                     respuesta en streaming + fuentes citadas + métricas
```

| Capa | Tecnología |
|---|---|
| Base vectorial | Qdrant Cloud Free Tier (coseno, 384 d) — 3 guías PDF (MinSalud 2025, OPS HEARTS) + 303 registros UCI Heart Disease |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local) |
| LLM | Google Gemini API (`gemini-3.5-flash`) · alternativas: Groq, Ollama |
| Orquestación | Python nativo (`rag_core.py`) |
| Interfaz | Streamlit multipágina |

## Estructura del repositorio

```
├── app.py                  # Entrada de Streamlit: barra lateral (hiperparámetros) + navegación
├── app_pages/
│   ├── chat.py             # Chatbot con historial, streaming y panel de fuentes
│   ├── documentos.py       # Gestión documental: KPIs del índice, metadatos, carga y eliminación de PDF
│   ├── explorador.py       # Búsqueda semántica directa con filtros
│   └── evaluacion.py       # Batería de pruebas y métricas
├── rag_core.py             # Configuración, recuperador Qdrant, cliente LLM, prompts y pipeline RAG
├── servicios.py            # Recursos cacheados de Streamlit
├── ingesta.py              # Chunking de PDFs nuevos (misma canalización del ACA 1)
├── evaluacion.py           # Casos de prueba y métricas (PASO 5)
├── ACA2_Computacion_Nube_RAG_Salud.ipynb
├── requirements.txt
├── .env.example            # Plantilla de variables (sin secretos)
└── .streamlit/
    ├── config.toml         # Tema visual
    └── secrets.toml.example
```

## Ejecución local

**Requisitos:** Python 3.10+ y la colección del ACA 1 ya cargada en Qdrant Cloud.

```bash
git clone https://github.com/leiderpq5521-creator/aca2-cardioguia-rag.git
cd aca2-cardioguia-rag
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # Linux/Mac: cp .env.example .env  → luego edite .env con sus claves
streamlit run app.py
```

Abra http://localhost:8501.

### Claves necesarias
| Variable | Dónde obtenerla |
|---|---|
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant Cloud → Clusters → API Keys (las mismas del ACA 1) |
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey (gratuita) |
| `GROQ_API_KEY` (opcional) | https://console.groq.com/keys |

### Usar un modelo local (Ollama)
```bash
ollama pull gemma3:4b
```
y en `.env`: `LLM_PROVIDER=ollama` y `LLM_MODEL=gemma3:4b`.

## Despliegue en Streamlit Community Cloud
1. Suba el repositorio a GitHub (el `.env` **no** se sube; está en `.gitignore`).
2. En https://share.streamlit.io cree la app apuntando a `app.py`.
3. En *Settings → Secrets* pegue el contenido de `.streamlit/secrets.toml.example` con sus claves reales.

## Seguridad de credenciales
- Las claves solo se leen desde `.env` o `st.secrets`; nunca están en el código ni se imprimen (la app muestra únicamente "cargada"/"FALTA").
- `.env` y `.streamlit/secrets.toml` están excluidos en `.gitignore`.
- Si una clave se expone por error, revóquela de inmediato en la consola del proveedor.

## Mitigación de alucinaciones
1. *System prompt* con reglas estrictas: responder solo con el contexto, citar `[F#]`, frase fija cuando no hay evidencia, no inventar dosis.
2. Umbral de similitud: sin evidencia suficiente no se invoca el LLM.
3. Verificación automática de citas contra los fragmentos recuperados (detecta citas inventadas).
4. Temperatura baja (0.2).
5. Ventana de contexto: los fragmentos se amplían con sus vecinos para no perder definiciones cortadas entre dos chunks.

## Cuotas de la capa gratuita de Gemini
La capa gratuita limita las solicitudes **por día y por modelo** (unas 20 en el modelo principal). Si se agota, el cliente pasa automáticamente a `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite` y `gemini-2.5-flash-lite` (configurable con `GEMINI_FALLBACK_MODELS`). Para la grabación del video, evite gastar la cuota en pruebas previas.

> ⚕️ Prototipo académico con fines educativos; no reemplaza la valoración médica profesional.
