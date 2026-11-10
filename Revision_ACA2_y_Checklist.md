# Revisión del ACA 2 · Lineamientos, continuidad con el ACA 1 y checklist de entrega

## 1. ¿Qué pide el ACA 2? (resumen de la guía)
El ACA 2 cierra el ciclo **end-to-end** del RAG: la base vectorial del ACA 1 se conecta a un **LLM** (local o por API) y se construye una **interfaz gráfica** con chatbot y gestión documental.

| Paso | Requisito de la guía | Cómo se cumplió en CardioGuía |
|---|---|---|
| 1 · Conexión base vectorial | Recuperar Top-k desde la base del ACA 1 | `Recuperador` en `rag_core.py` reutiliza el clúster, la colección y el modelo de embeddings del ACA 1; verifica estado, dimensión (384) e inventario |
| 2 · Configuración LLM | API key segura (.env) o servidor local | Gemini 3.5 Flash por API (elegido), Groq y Ollama como alternativas con la misma interfaz; claves en `.env`/`st.secrets` |
| 3 · Pipeline RAG + prompt | Plantilla que combine pregunta + fragmentos, responda solo con los documentos, cite fuentes y advierta si no hay información | `SYSTEM_PROMPT` con 8 reglas, contexto numerado `[F#]`, umbral de evidencia, verificación de citas, memoria conversacional |
| 4 · Interfaz gráfica | (a) chat con historial; (b) exploración de documentos, metadatos y fragmentos | Streamlit multipágina: Chatbot, Gestión documental (incluye carga y eliminación de PDF), Explorador vectorial, Evaluación |
| 5 · Validación | Precisión, tiempo de respuesta, robustez fuera de contexto | `evaluacion.py`: 9 casos (6 dominio + 3 robustez), métricas de citación, rechazo y latencia; calibración del umbral en el notebook |

## 2. Continuidad con el ACA 1 (revisión del entregable anterior)
| Aspecto del ACA 1 | Estado | Uso en el ACA 2 |
|---|---|---|
| Qdrant Cloud, colección `salud_cardiaca_rag`, coseno, 384 d | ✅ Correcto | Se consulta directamente; no hace falta re-indexar |
| Embeddings `paraphrase-multilingual-MiniLM-L12-v2` normalizados | ✅ Correcto | **Obligatorio usar el mismo modelo** en la consulta; el código lo verifica |
| Payload con `fuente`, `titulo`, `entidad`, `anio`, `pagina`, `texto` | ✅ Muy bueno | Permite citar documento y página y construir la referencia APA de cada fragmento |
| Índices de payload (`tipo_contenido`, `audiencia`, `fuente`…) | ✅ | Filtros del chat (audiencia) y del explorador; *facet* para el inventario |
| IDs determinísticos `uuid5` | ✅ | La carga de PDF desde la app usa el mismo esquema (re-subir un PDF actualiza, no duplica) |
| Registros UCI convertidos a texto | ⚠️ Útiles pero riesgosos para un chatbot clínico | Se excluyen del contexto por defecto; se pueden incluir bajo demanda y el prompt los marca como "caso ilustrativo" |
| Conclusión "siguiente paso: conectar a un LLM con citación de fuente y página" | ➜ | Es exactamente lo que implementa el ACA 2 |

**Observaciones de mejora detectadas en el ACA 1 (ya resueltas o mitigadas en el ACA 2):**
1. El ACA 1 no tenía un umbral de similitud: cualquier consulta devolvía Top-k aunque fuera irrelevante → el ACA 2 calibra un umbral con datos (sección 1.1 del notebook).
2. El ACA 1 solo validaba la recuperación; el ACA 2 agrega métricas de **fidelidad** (citas válidas/inventadas, tasa de citación) y de **robustez**.
3. El código del ACA 1 vivía solo en el notebook; ahora la lógica está en módulos reutilizables (`rag_core.py`, `ingesta.py`) que comparten el notebook y la app.

## 3. Checklist frente a la rúbrica
| Peso | Criterio | Evidencia | Pendiente del equipo |
|---|---|---|---|
| 20 % | Integración RAG | Notebook §1 y §3; `rag_core.py` | — |
| 20 % | Configuración LLM | Notebook §2 (justificación + tabla de hiperparámetros); `.env.example` | ✅ Hecho (Gemini 3.5 Flash con respaldo Flash-Lite) |
| 20 % | Interfaz gráfica | `app.py` + `app_pages/`; capturas en `capturas/` | ✅ Hecho |
| 15 % | Fidelidad | Notebook §3.3–3.5 y §5 (ejecutado: 9/9, 0 citas inventadas); página Evaluación | ✅ Hecho |
| 15 % | Póster PDF | `Poster_ACA2_RAG_Salud.pdf` | Nombres, enlaces reales (GitHub, video, app) y re-exportar |
| 10 % | Video 3–5 min | `Guion_Video_ACA2.md` | Grabar siguiendo el guion |

## 4. Pasos finales para el equipo
1. ✅ API key de Gemini configurada, notebook ejecutado con salidas y capturas tomadas.
2. Tener en cuenta la cuota gratuita (~20 solicitudes/día por modelo) antes de la grabación del video.
3. (Opcional) Agregar `GROQ_API_KEY` como respaldo adicional.
4. Crear el repositorio en GitHub y subir todo **excepto** `.env` (ya está en `.gitignore`). Verificar en GitHub que el `.env` NO aparece.
5. (Opcional, suma puntos) Desplegar en Streamlit Community Cloud con los *Secrets*.
6. ✅ Nombres de integrantes puestos. Falta reemplazar `USUARIO` (GitHub), el enlace del video y el de la app.
7. Grabar el video con el guion y verificar que dure entre 3 y 5 minutos.

## 5. Seguridad: alerta sobre el ACA 1
En la carpeta del curso hay un archivo `aca1-salud-rag_api_key.txt` y un `.env` dentro de la carpeta del entregable del ACA 1 y del ZIP `Entregable aca 1 RAG_salud.zip`. Si ese ZIP o carpeta se entregó o publicó, **la API key de Qdrant quedó expuesta**: se recomienda **rotarla** en Qdrant Cloud (crear una nueva, actualizar `.env` y borrar la anterior) y no incluir `.env` en los ZIP de entrega.
