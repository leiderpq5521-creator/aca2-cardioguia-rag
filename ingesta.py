"""
ingesta.py — Ingesta de nuevos PDF a la base vectorial desde la interfaz (gestión documental).

Replica EXACTAMENTE la canalización del ACA 1 para que los documentos nuevos sean coherentes
con los ya indexados: extracción por página -> limpieza -> chunking recursivo (500/75) ->
filtro de ruido -> payload con los mismos metadatos.
"""
from __future__ import annotations

import io
import re

CHUNK_SIZE, CHUNK_OVERLAP = 500, 75   # justificado en el ACA 1 (límite de 128 tokens del modelo MiniLM)


def limpiar(t: str) -> str:
    t = re.sub(r"-\n(?=\w)", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def pdf_a_chunks(nombre: str, contenido: bytes, titulo: str, entidad: str, anio: int,
                 audiencia: str, tipo_documento: str) -> tuple[list[dict], dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader

    lector = PdfReader(io.BytesIO(contenido))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                              separators=["\n\n", "\n", ". ", " ", ""])
    chunks, descartados, utiles = [], 0, 0
    for n_pag, pg in enumerate(lector.pages, start=1):
        txt = limpiar(pg.extract_text() or "")
        if len(txt) < 80:
            continue
        utiles += 1
        for k, frag in enumerate(splitter.split_text(txt)):
            frag = frag.strip()
            if len(frag) < 80:
                continue
            if re.search(r"\.{5,}|https?://|doi\.org", frag):
                descartados += 1
                continue
            chunks.append({"texto": frag, "payload": {
                "tipo_contenido": "documento", "fuente": nombre, "titulo": titulo, "entidad": entidad,
                "anio": int(anio), "audiencia": audiencia, "tipo_documento": tipo_documento,
                "pagina": n_pag, "chunk_en_pagina": k, "texto": frag,
            }})
    resumen = {"paginas": len(lector.pages), "paginas_con_texto": utiles,
               "fragmentos": len(chunks), "descartados_ruido": descartados}
    return chunks, resumen
