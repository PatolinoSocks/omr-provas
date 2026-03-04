import os
import io
import re
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st

from omr_engine import corrigir_prova, OMRConfig


st.set_page_config(page_title="OMR - Correção em Lote", layout="centered")
st.title("📄 OMR - Raphael Alvim")
st.title("Aplicativo para Correção de Babarito de Provas em Lote")
st.caption("Envie várias fotos e baixe um Excel/CSV consolidado.")

ALT_OK = {"A", "B", "C", "D"}


def parse_turma_nome_from_filename(filename: str):
    # remove caminho e extensão
    base = os.path.splitext(os.path.basename(filename))[0].strip()

    # remove espaços extras
    base = base.replace(" ", "")

    if "_" in base:
        turma, nome = base.split("_", 1)
        turma = turma.strip()
        nome = nome.strip()
        return turma, nome

    # fallback caso não tenha _
    return "SEM_TURMA", base


def parse_gabarito_text(text: str, n_questions: int = 22):
    text = (text or "").strip().upper()
    if not text:
        return {}

    # caso 1: "ABCD..." (contínuo)
    compact = re.sub(r"[^ABCD]", "", text)
    if len(compact) >= n_questions:
        compact = compact[:n_questions]
        return {i + 1: compact[i] for i in range(n_questions)}

    # caso 2: linhas "1:A"
    g = {}
    for line in text.splitlines():
        line = line.strip().upper()
        if not line:
            continue
        m = re.match(r"^\s*(\d+)\s*[:=\-\s]\s*([ABCD])\s*$", line)
        if m:
            q = int(m.group(1))
            a = m.group(2)
            if 1 <= q <= n_questions:
                g[q] = a

    # caso 3: tokens separados
    if not g:
        tokens = re.split(r"[\s,;]+", text)
        tokens = [t for t in tokens if t]
        if len(tokens) >= n_questions and all(t in ALT_OK for t in tokens[:n_questions]):
            return {i + 1: tokens[i] for i in range(n_questions)}

    return g


def answers_to_wide_row(answers):
    row = {}
    for q, a in answers:
        row[f"Q{q:02d}"] = a if a is not None else ""
    return row


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("⚙️ Configuração")

cfg = OMRConfig()
n_questions = getattr(cfg, "n_questions_used", 22)

st.sidebar.subheader("Gabarito (1–22)")
gabarito_text = st.sidebar.text_area(
    "Cole o gabarito (22 letras ou linhas 1:A etc.)",
    value="ABCDABCDABCDABCDABCDAB",
    height=120,
)
gabarito = parse_gabarito_text(gabarito_text, n_questions=n_questions)

if len(gabarito) != n_questions:
    st.sidebar.warning(f"Gabarito incompleto: {len(gabarito)}/{n_questions}. A nota pode ficar errada.")

debug = st.sidebar.checkbox("Mostrar detalhes por aluno (thr, erros/brancos)", value=False)

st.sidebar.divider()
st.sidebar.caption("Dica: nomeie arquivos como Turma_Nome.ext (ex.: 7Verde_Aline.png)")


# ---------------------------
# Upload múltiplo
# ---------------------------
st.subheader("1) Enviar imagens (múltiplas)")
uploads = st.file_uploader(
    "Envie várias imagens (PNG/JPG).",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if not uploads:
    st.stop()

st.subheader("2) Rodar correção em lote")
run = st.button("✅ Corrigir todas", type="primary")
if not run:
    st.stop()


# ---------------------------
# Processamento em lote
# ---------------------------
resultados = []
erros = []

progress = st.progress(0)
status = st.empty()

for i, up in enumerate(uploads, start=1):
    status.write(f"Processando {i}/{len(uploads)}: **{up.name}**")

    suffix = os.path.splitext(up.name)[1].lower() or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(up.read())
        tmp_path = tmp.name

    turma_sug, nome_sug = parse_turma_nome_from_filename(up.name)

    try:
        r = corrigir_prova(tmp_path, gabarito, cfg=cfg)

        r["turma"] = turma_sug
        r["nome"] = nome_sug

        row = {
            "turma": r.get("turma", ""),
            "nome": r.get("nome", ""),
            "imagem": up.name,
            "nota": r.get("nota", None),
            "percentual": r.get("percentual", None),
            "acertos": r.get("acertos", None),
            "erros": ",".join(map(str, r.get("erros", []))) if r.get("erros") else "",
        }

        row.update(answers_to_wide_row(r.get("respostas", [])))
        resultados.append(row)

    except Exception as e:
        erros.append({"imagem": up.name, "erro": str(e)})

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # barra de progresso corrigida
    progress.progress(int((i / len(uploads)) * 100))

status.success("✅ Lote finalizado!")


# ---------------------------
# Tabelas de saída
# ---------------------------
st.subheader("3) Resultado consolidado")

df = pd.DataFrame(resultados) if resultados else pd.DataFrame()
if not df.empty:
    # ordena por turma/nome
    cols_first = ["turma", "nome", "imagem", "nota", "percentual", "acertos", "erros"]
    other_cols = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + other_cols]
    df = df.sort_values(["turma", "nome", "imagem"], na_position="last")

    st.dataframe(df, use_container_width=True)
else:
    st.warning("Nenhum resultado gerado.")

if erros:
    st.subheader("⚠️ Imagens com erro")
    st.dataframe(pd.DataFrame(erros), use_container_width=True)


# ---------------------------
# Downloads (CSV/Excel)
# ---------------------------
st.subheader("4) Baixar Excel/CSV consolidado")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = f"resultados_lote_{timestamp}"

# CSV
csv_bytes = df.to_csv(index=False).encode("utf-8") if not df.empty else b""
st.download_button(
    "⬇️ Baixar CSV",
    data=csv_bytes,
    file_name=f"{base_name}.csv",
    mime="text/csv",
    disabled=df.empty,
)

# Excel com 2 abas: resultados + erros
excel_buffer = io.BytesIO()
if not df.empty:
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="resultados")
        if erros:
            pd.DataFrame(erros).to_excel(writer, index=False, sheet_name="erros")

st.download_button(
    "⬇️ Baixar Excel (.xlsx)",
    data=excel_buffer.getvalue() if not df.empty else b"",
    file_name=f"{base_name}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    disabled=df.empty,
)