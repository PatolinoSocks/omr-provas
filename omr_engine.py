# omr_engine.py
# Motor OMR para prova objetiva (A-D) com layout: 4 colunas x 10 linhas (40 posições), usando até 40 questões.
#
# Requisitos:
#   pip install opencv-python numpy scikit-learn pandas openpyxl
#
# Uso:
#   from omr_engine import corrigir_prova, OMRConfig
#   resultado = corrigir_prova("imagens/7Branco_Aline.png", gabarito)
#   print(resultado)

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

Answer = Optional[str]  # "A"/"B"/"C"/"D" ou None
AnswersList = List[Tuple[int, Answer]]


@dataclass
class OMRConfig:
    # --- pré-processamento ---
    background_blur_ksize: int = 101  # ímpar
    use_otsu: bool = True

    # --- ROI (marcadores quadrados) ---
    marker_min_area: int = 200
    marker_ar_min: float = 0.80
    marker_ar_max: float = 1.20
    marker_min_side: int = 10
    roi_pad: int = 20
    prefer_bottom_fraction_y: float = 0.35  # prioriza marcadores abaixo disso (folhas com capa)

    # --- detecção de bolhas (CANNY na ROI) ---
    canny_blur_ksize: int = 7
    canny_t1: int = 50
    canny_t2: int = 150
    edges_dilate_ksize: int = 3
    edges_dilate_iter: int = 1

    # filtros por círculo mínimo
    min_edge_contour_area: int = 25
    min_r: float = 8.0
    max_r: float = 25.0
    circ_min_edge: float = 0.10
    r_hist_bins: int = 40
    r_tol_rel: float = 0.25

    # --- preenchimento ---
    close_ksize: int = 5
    fill_radius_factor: float = 0.70
    otsu_min_thr: float = 0.20

    # --- layout ---
    n_cols: int = 4
    n_rows_per_col: int = 10
    n_alts: int = 4
    n_questions_used: int = 40

    # --- KMeans ---
    kmeans_n_init: int = 10

    # --- decisão ---
    thr_abs_floor: float = 0.18


def parse_turma_nome(image_path: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(image_path))[0]
    if "_" in base:
        turma, nome = base.split("_", 1)
        return turma, nome
    return "SEM_TURMA", base


def preprocess(image: np.ndarray, cfg: OMRConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna:
      gray, thresh_raw (invertido), thresh_fill (close para medir preenchimento)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    k = cfg.background_blur_ksize
    if k % 2 == 0:
        k += 1

    background = cv2.GaussianBlur(gray, (k, k), 0)
    normalized = cv2.divide(gray, background, scale=255)

    if cfg.use_otsu:
        _, thresh_raw = cv2.threshold(
            normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, thresh_raw = cv2.threshold(normalized, 150, 255, cv2.THRESH_BINARY_INV)

    kc = cfg.close_ksize
    if kc % 2 == 0:
        kc += 1
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
    thresh_fill = cv2.morphologyEx(thresh_raw, cv2.MORPH_CLOSE, k_close)

    return gray, thresh_raw, thresh_fill


def find_roi_by_markers(
    thresh_raw: np.ndarray, orig: np.ndarray, cfg: OMRConfig
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Acha 4 marcadores quadrados e recorta ROI.
    Retorna: orig_roi, thresh_raw_roi, (x1,y1,x2,y2)
    """
    cnts, _ = cv2.findContours(thresh_raw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = thresh_raw.shape[:2]
    cands = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < cfg.marker_min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)

        if not (cfg.marker_ar_min <= ar <= cfg.marker_ar_max):
            continue
        if w < cfg.marker_min_side or h < cfg.marker_min_side:
            continue

        cands.append((area, x, y, w, h))

    cands_bottom = [t for t in cands if t[2] > H * cfg.prefer_bottom_fraction_y]
    cands_use = cands_bottom if len(cands_bottom) >= 4 else cands
    cands_use = sorted(cands_use, key=lambda t: t[0], reverse=True)[:12]

    if len(cands_use) < 4:
        raise RuntimeError(f"Não consegui achar 4 marcadores. Achei só {len(cands_use)}.")

    centers = []
    for area, x, y, w, h in cands_use:
        cx = x + w / 2
        cy = y + h / 2
        centers.append((cx, cy, x, y, w, h, area))

    tl = min(centers, key=lambda t: t[0] + t[1])
    tr = max(centers, key=lambda t: t[0] - t[1])
    bl = min(centers, key=lambda t: t[0] - t[1])
    br = max(centers, key=lambda t: t[0] + t[1])

    xs = [tl[2], tr[2] + tr[4], bl[2], br[2] + br[4]]
    ys = [tl[3], tr[3], bl[3] + bl[5], br[3] + br[5]]

    x1 = int(max(0, min(xs) - cfg.roi_pad))
    x2 = int(min(W, max(xs) + cfg.roi_pad))
    y1 = int(max(0, min(ys) - cfg.roi_pad))
    y2 = int(min(H, max(ys) + cfg.roi_pad))

    orig_roi = orig[y1:y2, x1:x2]
    thresh_raw_roi = thresh_raw[y1:y2, x1:x2]

    return orig_roi, thresh_raw_roi, (x1, y1, x2, y2)


def detect_bubbles_canny(orig_roi: np.ndarray, cfg: OMRConfig) -> List[Tuple[int, int, float]]:
    """
    Detecta bolhas por Canny na ROI e usa:
      - minEnclosingCircle para (cx,cy,r)
      - filtro por raio mais comum (histograma)
    Retorna lista de círculos: [(cx,cy,r_float), ...]
    """
    gray_roi = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2GRAY)

    kb = cfg.canny_blur_ksize
    if kb % 2 == 0:
        kb += 1
    gray_roi = cv2.GaussianBlur(gray_roi, (kb, kb), 0)

    edges = cv2.Canny(gray_roi, cfg.canny_t1, cfg.canny_t2)

    kd = cfg.edges_dilate_ksize
    if kd % 2 == 0:
        kd += 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
    edges = cv2.dilate(edges, k, iterations=cfg.edges_dilate_iter)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands: List[Tuple[int, int, float]] = []
    r_list: List[float] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.min_edge_contour_area:
            continue

        (x0, y0), r = cv2.minEnclosingCircle(c)
        r = float(r)

        if r < cfg.min_r or r > cfg.max_r:
            continue

        peri = cv2.arcLength(c, True)
        if peri <= 1e-6:
            continue

        circ = 4 * np.pi * area / (peri * peri)
        if circ < cfg.circ_min_edge:
            continue

        cands.append((int(x0), int(y0), r))
        r_list.append(r)

    if not r_list:
        return []

    r_arr = np.array(r_list, dtype=float)
    hist, bins = np.histogram(r_arr, bins=cfg.r_hist_bins)
    i_peak = int(np.argmax(hist))
    r_mode = float((bins[i_peak] + bins[i_peak + 1]) / 2.0)

    tol = cfg.r_tol_rel * r_mode
    circles = [(cx, cy, r) for (cx, cy, r) in cands if abs(r - r_mode) <= tol]

    return circles


def otsu_1d_threshold(values: List[float], min_thr: float) -> float:
    if not values:
        return min_thr

    hist, _ = np.histogram(values, bins=256, range=(0, 1))
    total = int(hist.sum())
    if total == 0:
        return min_thr

    sum_total = sum(i * int(hist[i]) for i in range(256))
    sumB = 0
    wB = 0
    max_var = 0
    thr_bin = 0

    for i in range(256):
        wB += int(hist[i])
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break

        sumB += i * int(hist[i])
        mB = sumB / wB
        mF = (sum_total - sumB) / wF

        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            thr_bin = i

    thr = thr_bin / 255.0
    return max(float(thr), float(min_thr))


def measure_fill_from_circles(
    circles: List[Tuple[int, int, float]],
    thresh_fill_roi: np.ndarray,
    cfg: OMRConfig
) -> Tuple[List[Tuple[int, int, int, float]], float]:
    """
    Para cada círculo (cx,cy,r_float), mede preenchimento em thresh_fill_roi.
    Retorna bubble_info = [(cx,cy,r_int,fill), ...] e thr (Otsu 1D no fill).
    """
    fills: List[float] = []
    bubble_info: List[Tuple[int, int, int, float]] = []

    for (cx, cy, r) in circles:
        rr = int(max(1, r * cfg.fill_radius_factor))
        mask = np.zeros(thresh_fill_roi.shape, dtype="uint8")
        cv2.circle(mask, (int(cx), int(cy)), rr, 255, -1)

        total = cv2.countNonZero(cv2.bitwise_and(thresh_fill_roi, thresh_fill_roi, mask=mask))
        area_mask = float(np.pi * (rr ** 2))
        fill = float(total / area_mask) if area_mask > 0 else 0.0

        fills.append(fill)
        bubble_info.append((int(cx), int(cy), int(round(r)), fill))

    thr = otsu_1d_threshold(fills, min_thr=cfg.otsu_min_thr)
    return bubble_info, thr


def extract_answers_grid(
    bubble_info: List[Tuple[int, int, int, float]],
    thr: float,
    cfg: OMRConfig
) -> AnswersList:
    """
    Extrai respostas usando grade fixa do formulário:
    - 4 colunas
    - 10 linhas por coluna
    - 4 alternativas A-D
    Usa as posições detectadas para ajustar a grade e evitar deslocamentos.
    """
    if not bubble_info:
        return [(q, None) for q in range(1, cfg.n_questions_used + 1)]

    alt_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    thr_abs = max(float(thr), float(cfg.thr_abs_floor))

    xs = np.array([b[0] for b in bubble_info], dtype=float)

    # 1) encontrar centros das 4 colunas
    k_cols = KMeans(n_clusters=cfg.n_cols, random_state=42, n_init=cfg.kmeans_n_init)
    _ = k_cols.fit_predict(xs.reshape(-1, 1))
    col_centers = np.sort(k_cols.cluster_centers_.flatten())

    answers: List[Tuple[int, Answer]] = []

    for c in range(cfg.n_cols):
        col_center = col_centers[c]
        col_group = [b for b in bubble_info if abs(b[0] - col_center) < 80]

        if len(col_group) < cfg.n_rows_per_col * 2:
            for r in range(cfg.n_rows_per_col):
                q = c * cfg.n_rows_per_col + r + 1
                answers.append((q, None))
            continue

        col_xs = np.array([b[0] for b in col_group], dtype=float)
        col_ys = np.array([b[1] for b in col_group], dtype=float)

        # centros A-D globais da coluna
        x_min = col_xs.min()
        x_max = col_xs.max()

        x_margin = (x_max - x_min) * 0.03
        x_min -= x_margin
        x_max += x_margin

        alt_centers = np.linspace(x_min, x_max, cfg.n_alts)

        # topo/base com margem
        y_min = col_ys.min()
        y_max = col_ys.max()
        y_margin = (y_max - y_min) * 0.04
        y_min -= y_margin
        y_max += y_margin

        row_centers = np.linspace(y_min, y_max, cfg.n_rows_per_col)
        row_step = row_centers[1] - row_centers[0]

        for r in range(cfg.n_rows_per_col):
            q = c * cfg.n_rows_per_col + r + 1
            y_target = row_centers[r]

            near_row = [b for b in col_group if abs(b[1] - y_target) < 0.45 * row_step]

            slots: List[Optional[Tuple[int, int, int, float]]] = [None] * cfg.n_alts

            for b in near_row:
                a_idx = int(np.argmin(np.abs(alt_centers - b[0])))

                if slots[a_idx] is None:
                    slots[a_idx] = b
                else:
                    old = slots[a_idx]
                    d_new = abs(b[0] - alt_centers[a_idx]) + abs(b[1] - y_target)
                    d_old = abs(old[0] - alt_centers[a_idx]) + abs(old[1] - y_target)
                    if d_new < d_old:
                        slots[a_idx] = b

            if any(s is None for s in slots):
                answers.append((q, None))
                continue

            fills = [float(s[3]) for s in slots]  # type: ignore[arg-type]
            best_i = int(np.argmax(fills))
            best_val = fills[best_i]
            second_val = sorted(fills, reverse=True)[1]

            if best_val >= thr_abs:
                answers.append((q, alt_map[best_i]))
            elif best_val > 0.12 and (best_val - second_val) > 0.05:
                answers.append((q, alt_map[best_i]))
            else:
                answers.append((q, None))

    answers = sorted(answers, key=lambda x: x[0])

    # fallback de deslocamento raro
    if answers and answers[0][1] is None:
        filled = [a for _, a in answers if a is not None]
        if len(filled) >= int(0.7 * len(answers)):
            shifted = []
            for i in range(len(answers)):
                if i + 1 < len(answers):
                    shifted.append((answers[i][0], answers[i + 1][1]))
                else:
                    shifted.append((answers[i][0], None))
            answers = shifted

    return answers[: cfg.n_questions_used]


def grade_answers(answers: AnswersList, gabarito: Dict[int, str]) -> Dict[str, object]:
    acertos = 0
    erros: List[int] = []
    brancos: List[int] = []

    total_ref = len(gabarito) if gabarito else len(answers)
    total_ref = max(total_ref, 1)

    for q, resp in answers:
        correta = gabarito.get(q)
        if resp is None:
            brancos.append(q)
        elif correta is not None and resp == correta:
            acertos += 1
        else:
            erros.append(q)

    nota = acertos
    percentual = (acertos / total_ref) * 100.0

    return {
        "acertos": acertos,
        "erros": erros,
        "brancos": brancos,
        "nota": nota,
        "percentual": percentual,
    }


def corrigir_prova(
    image_path: str,
    gabarito: Dict[int, str],
    cfg: Optional[OMRConfig] = None,
    debug_dir: Optional[str] = None
) -> Dict[str, object]:
    """
    Produção:
    - lê imagem
    - preprocess
    - ROI por marcadores
    - detect bubbles (CANNY)
    - measure fill (thr)
    - extract answers
    - grade
    """
    cfg = cfg or OMRConfig()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Não consegui ler a imagem: {image_path}")

    turma, nome = parse_turma_nome(image_path)

    _, thresh_raw, thresh_fill = preprocess(image, cfg)
    orig = image.copy()

    # ROI
    orig_roi, thresh_raw_roi, (x1, y1, x2, y2) = find_roi_by_markers(thresh_raw, orig, cfg)
    thresh_fill_roi = thresh_fill[y1:y2, x1:x2]

    # bubbles (CANNY)
    circles = detect_bubbles_canny(orig_roi, cfg)

    # fill
    bubble_info, thr = measure_fill_from_circles(circles, thresh_fill_roi, cfg)

    # answers
    answers = extract_answers_grid(bubble_info, thr, cfg)

    if gabarito:
        max_q = max(gabarito.keys())
        answers = [a for a in answers if a[0] <= max_q]

    # grade
    result_grade = grade_answers(answers, gabarito)

    out = {
        "turma": turma,
        "nome": nome,
        "imagem": os.path.basename(image_path),
        "thr": float(thr),
        "respostas": answers,
        **result_grade,
    }

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

        dbg = orig.copy()
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.imwrite(os.path.join(debug_dir, f"{turma}_{nome}_01_roi_rect.png"), dbg)

        cv2.imwrite(os.path.join(debug_dir, f"{turma}_{nome}_02_roi.png"), orig_roi)

        img_bubbles = orig_roi.copy()
        for (cx, cy, r) in circles:
            cv2.circle(img_bubbles, (int(cx), int(cy)), int(round(r)), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, f"{turma}_{nome}_03_bubbles.png"), img_bubbles)

        img_mark = orig_roi.copy()
        thr_abs = max(float(thr), float(cfg.thr_abs_floor))
        for (cx, cy, r, fill) in bubble_info:
            color = (0, 255, 0) if float(fill) >= thr_abs else (0, 0, 255)
            cv2.circle(img_mark, (int(cx), int(cy)), int(r), color, 2)
        cv2.imwrite(os.path.join(debug_dir, f"{turma}_{nome}_04_marked.png"), img_mark)

    return out