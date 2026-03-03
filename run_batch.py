# run_batch.py
import os
import glob
import pandas as pd

from omr_engine import corrigir_prova, OMRConfig


PASTA_IMAGENS = "imagens"
PASTA_SAIDA = "resultados"
PASTA_DEBUG = "debug"          # deixe None se não quiser debug
EXTENSOES = ("*.png", "*.jpg", "*.jpeg")

# >>>> TROQUE PELO SEU GABARITO <<<<
gabarito = {
    1: "B", 2: "C", 3: "C", 4: "C", 5: "C", 6: "A", 7: "B", 8: "A",
    9: "C", 10: "D", 11: "C", 12: "B", 13: "C", 14: "B", 15: "C", 16: "A",
    17: "B", 18: "C", 19: "B", 20: "C", 21: "D", 22: "B"
}

cfg = OMRConfig()

os.makedirs(PASTA_SAIDA, exist_ok=True)
if PASTA_DEBUG:
    os.makedirs(PASTA_DEBUG, exist_ok=True)

# coletar arquivos (inclui subpastas)
arquivos = []
for ext in EXTENSOES:
    arquivos.extend(glob.glob(os.path.join(PASTA_IMAGENS, "**", ext), recursive=True))
arquivos = sorted(arquivos)

if not arquivos:
    raise FileNotFoundError(f"Nenhuma imagem encontrada em '{PASTA_IMAGENS}'")

linhas = []

for i, path in enumerate(arquivos, start=1):
    base = os.path.basename(path)
    try:
        debug_dir = PASTA_DEBUG if PASTA_DEBUG else None

        r = corrigir_prova(path, gabarito, cfg=cfg, debug_dir=debug_dir)

        linhas.append({
            "turma": r["turma"],
            "nome": r["nome"],
            "imagem": r["imagem"],
            "nota": r["nota"],
            "percentual": r["percentual"],
            "acertos": r["acertos"],
            "erros": ",".join(map(str, r["erros"])),
        })

        print(f"[OK] {i:03d} - {r['turma']}_{r['nome']} -> nota {r['nota']}")

    except Exception as e:
        linhas.append({
            "turma": "ERRO",
            "nome": base,
            "imagem": base,
            "nota": None,
            "percentual": None,
            "acertos": None,
            "erros": None,
            "brancos": None,
            "thr": None,
            "erro_msg": str(e),
        })
        print(f"[ERRO] {i:03d} - {base}: {e}")

df = pd.DataFrame(linhas)

# geral
saida_xlsx = os.path.join(PASTA_SAIDA, "resultados_geral.xlsx")
saida_csv = os.path.join(PASTA_SAIDA, "resultados_geral.csv")

df.to_excel(saida_xlsx, index=False)
df.to_csv(saida_csv, index=False)

# por turma
ok_df = df[df["turma"] != "ERRO"].copy()
for turma, df_turma in ok_df.groupby("turma"):
    out_xlsx = os.path.join(PASTA_SAIDA, f"resultados_{turma}.xlsx")
    out_csv = os.path.join(PASTA_SAIDA, f"resultados_{turma}.csv")
    df_turma.to_excel(out_xlsx, index=False)
    df_turma.to_csv(out_csv, index=False)

print("\nArquivos gerados em:", PASTA_SAIDA)
print(" -", saida_xlsx)
print(" -", saida_csv)