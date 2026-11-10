# Guion del video de sustentación · ACA 2 (objetivo: 4 min 30 s, máximo 5 min)

| Tiempo | Quién | En pantalla | Qué decir (resumen) |
|---|---|---|---|
| 0:00–0:25 | Kevin David Guzmán Payares | Póster (encabezado) | Presentación del equipo, pilar Salud y pregunta orientadora. "En el ACA 1 construimos la base vectorial en Qdrant Cloud; en el ACA 2 la convertimos en un chatbot que responde citando las guías." |
| 0:25–1:05 | Kevin David Guzmán Payares | Póster (diagrama de arquitectura) | Flujo end-to-end: pregunta → reformulación con historial → embedding 384 d → Top-k en Qdrant → umbral de evidencia → prompt con contexto [F#] → Gemini → verificación de citas → interfaz. |
| 1:05–1:45 | Leider Antonio Pacheco Quinto | Notebook, sección 2 | **Por qué Gemini 3.5 Flash por API**: capa gratuita, baja latencia, buen español, cero infraestructura; alternativas intercambiables (Groq/Llama, Ollama/Gemma local). Hiperparámetros: temperatura 0.2 (fidelidad), top_p 0.9, 2048 tokens, *thinking* bajo; respaldo automático a Flash-Lite si se agota la cuota diaria. Claves en `.env`, nunca en GitHub. |
| 1:45–3:15 | Joaquín Pablo Redondo Tapias | **App en vivo** · Chatbot | 1) Pregunta sugerida "Diagnóstico" → abrir panel de fuentes (citada/no citada, página, APA, latencias). 2) Caso complejo HEARTS (hombre 60 años, 160/100). 3) Seguimiento elíptico "¿y si no se logra la meta?" → mostrar la consulta reformulada. 4) Pregunta fuera de dominio "capital de Australia" → rechazo sin llamar al LLM. |
| 3:15–3:50 | José Luis Ruiz Ospino | App · Gestión documental y Explorador | KPIs del índice (puntos, 384 d, coseno, estado), fragmentos por fuente, explorar metadatos y fragmentos de una guía; mostrar el formulario de carga de PDF; búsqueda directa con filtro *audiencia = pacientes*. |
| 3:50–4:15 | José Luis Ruiz Ospino | App · Evaluación | Resultados de la batería: 9/9 casos correctos, 100 % de respuestas con cita, 0 citas inventadas, latencia media 3,7 s; mencionar la mejora de 6/9 a 9/9 (ventana de contexto + regla de evidencia parcial). **Ojo con la cuota:** la capa gratuita permite ~20 solicitudes/día por modelo; no ejecute la batería completa justo antes de grabar. |
| 4:15–4:35 | Kevin David Guzmán Payares | Póster (conclusiones y QR/enlaces) | Conclusiones, limitaciones y trabajo futuro (re-ranking, RAGAS, más guías). Cierre. |

**Consejos de grabación**
- Abrir la app y ejecutar una consulta *antes* de grabar (el modelo de embeddings ya queda cargado en caché y la demo es fluida).
- Ocultar el archivo `.env` y la barra de direcciones si muestra datos sensibles; nunca abrir el `.env` en cámara.
- Ensayar con cronómetro: el criterio de la rúbrica exige **ajuste estricto de 3 a 5 minutos**.
