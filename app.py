import os
import io
import re
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st

from omr_engine import corrigir_prova, OMRConfig


st.set_page_config(page_title="OMR - Correção em Lote", layout="centered")
st.title("📄 OMR - Raphael Alvim Apps")
st.title("Correção de Gabarito de Provas")
st.caption("Envie várias fotos e baixe um Excel/CSV consolidado.")

ALT_OK = {"A", "B", "C", "D"}
MAX_QUESTIONS = 40  # formulário físico suporta até 40


def parse_turma_nome_from_filename(filename: str):
    base = os.path.splitext(os.path.basename(filename))[0].strip()
    base = base.replace(" ", "")

    if "_" in base:
        turma, nome = base.split("_", 1)
        turma = turma.strip()
        nome = nome.strip()
        return turma, nome

    return "SEM_TURMA", base


def parse_gabarito_text(text: str, n_questions: int = 40):
    text = (text or "").strip().upper()
    if not text:
        return {}

    # caso 1: "ABCD..." contínuo
    compact = re.sub(r"[^ABCD]", "", text)
    if len(compact) >= 1:
        compact = compact[:n_questions]
        return {i + 1: compact[i] for i in range(len(compact))}

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
        if len(tokens) >= 1 and all(t in ALT_OK for t in tokens[: min(len(tokens), n_questions)]):
            return {i + 1: tokens[i] for i in range(min(len(tokens), n_questions))}

    return g


def answers_to_wide_row(answers, n_questions: int):
    m = {q: a for q, a in answers}
    return {
        f"Q{i:02d}": (m.get(i) if m.get(i) is not None else "")
        for i in range(1, n_questions + 1)
    }


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("⚙️ Configuração")

cfg = OMRConfig()

st.sidebar.subheader("Gabarito (até 40 questões)")
gabarito_text = st.sidebar.text_area(
    "Cole o gabarito (até 40 letras ou linhas 1:A etc.)",
    value="",
    height=120,
)

gabarito = parse_gabarito_text(gabarito_text, n_questions=MAX_QUESTIONS)

if len(gabarito) == 0:
    st.sidebar.warning("Cole o gabarito da prova.")
    n_questions_used = MAX_QUESTIONS
else:
    n_questions_used = len(gabarito)
    st.sidebar.success(f"Gabarito carregado: {n_questions_used} questões.")
    st.sidebar.info(
        f"""
Formulário suporta até: **{MAX_QUESTIONS} questões**

Questões da prova: **{n_questions_used}**
"""
    )

# engine sempre usa layout físico de 40
cfg.n_rows_per_col = 10
cfg.n_questions_used = n_questions_used

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

if len(gabarito) == 0:
    st.warning("Cole o gabarito antes de processar as imagens.")
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

        respostas_detectadas = sum(
            1 for _, a in r.get("respostas", []) if a is not None
        )

        r["turma"] = turma_sug
        r["nome"] = nome_sug

        row = {
            "turma": r.get("turma", ""),
            "nome": r.get("nome", ""),
            "imagem": up.name,
            "nota": r.get("nota", None),
            "percentual": r.get("percentual", None),
            "acertos": r.get("acertos", None),
            "respondidas": respostas_detectadas,
            "erros": ",".join(map(str, r.get("erros", []))) if r.get("erros") else "",
        }

        row.update(answers_to_wide_row(r.get("respostas", []), n_questions_used))

        if debug:
            row["thr"] = r.get("thr", None)
            row["brancos"] = ",".join(map(str, r.get("brancos", []))) if r.get("brancos") else ""

        resultados.append(row)

    except Exception as e:
        erros.append({"imagem": up.name, "erro": str(e)})

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    progress.progress(int((i / len(uploads)) * 100))

status.success("✅ Lote finalizado!")

st.success(
    f"""
📊 Resumo da correção

Formulário: **{MAX_QUESTIONS} posições**

Questões corrigidas: **{n_questions_used}**
"""
)


# ---------------------------
# Tabelas de saída
# ---------------------------
st.subheader("3) Resultado consolidado")

df = pd.DataFrame(resultados) if resultados else pd.DataFrame()
if not df.empty:
    cols_first = ["turma", "nome", "imagem", "nota", "percentual", "acertos", "respondidas", "erros"]
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

csv_bytes = df.to_csv(index=False).encode("utf-8") if not df.empty else b""
st.download_button(
    "⬇️ Baixar CSV",
    data=csv_bytes,
    file_name=f"{base_name}.csv",
    mime="text/csv",
    disabled=df.empty,
)

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