"""
FeS_HKE_plot.py
================

Given the fitted spin-wave parameters (read from ``fit_results.csv``),
compute the closed contour in the (H,0,L)-(K,-2K,L) plane (parameterised by
(h,k)) that satisfies ``calculate_lambda(H(h,k), K(h,k), L) == TARGET_ENERGY``.
The corresponding reciprocal-space coordinates are H = h + k, K = -2k.  Inside the
contour ``lambda < TARGET_ENERGY``; outside the contour ``lambda > TARGET_ENERGY``.  We exploit the monotonicity of
``calculate_lambda`` with respect to the radial distance from the origin:
inside the contour ``lambda < TARGET_ENERGY``; outside the contour
``lambda > TARGET_ENERGY``.  Along each ray we therefore use a bracketing
and bisection strategy to locate the intersection efficiently.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fitting_FeS import (
    calculate_Aq,
    calculate_Bq,
    calculate_Lq1,
    calculate_Lq2,
    calculate_lambda,
)

# ---------------------------------------------------------------------
# User-tunable constants (edit here as needed)
# ---------------------------------------------------------------------

TARGET_ENERGY = 90.0  # 計算したいエネルギー値 E （単位はデータと揃える）
FIXED_L = 3.0  # L 固定値

# (H,0,L)-(K,-2K,L) 平面で探索するパラメータ領域 (h,k)
H_PARAM_RANGE = (-0.5, 0.5)
K_PARAM_RANGE = (-0.5, 0.5)

# 数値計算パラメータ
ANGLE_SAMPLES = 720        # 閉曲線の分解能（角度分割数）
RADIUS_INITIAL = 0.02      # ブラケット探索の初期半径
RADIUS_GROWTH = 1.8        # ブラケット拡大係数
RADIUS_MAX = 5.0           # 最大探索半径（物理的に十分大きく設定）
BISECTION_TOL = 1e-6       # 二分法終了条件（半径差）
LAMBDA_TOL = 1e-6          # 二分法終了条件（関数値差）

# ---------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------

PARAMETER_NAMES = ["J1", "J2", "J3", "J_alt", "J_prime_alt", "S", "D"]


def read_fit_results(csv_path: Path) -> Dict[str, float]:
    """
    fit_results.csv からパラメータを辞書形式で読み込む。
    フォーマットは ``fitting_FeS.py`` が出力するものを想定。
    """
    params: Dict[str, float] = {}
    with csv_path.open(newline="") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            if name.startswith("#"):
                continue
            if name in PARAMETER_NAMES:
                params[name] = float(row[1])
            elif name == "residual_sum":
                break  # パラメータ表の終了

    missing = [name for name in PARAMETER_NAMES if name not in params]
    if missing:
        raise ValueError(f"Missing parameters in fit_results.csv: {missing}")
    return params


def evaluate_lambda(params: Dict[str, float], h_param: float, k_param: float, L: float) -> float:
    """
    (H,0,L)-(K,-2K,L) 平面上のパラメータ (h_param, k_param) に対応する点で
    calculate_lambda を評価し、実数部を返す。
    """
    H_coord = h_param + k_param
    K_coord = -2.0 * k_param

    J1 = params["J1"]
    J2 = params["J2"]
    J3 = params["J3"]
    J_alt = params["J_alt"]
    J_prime_alt = params["J_prime_alt"]
    S = params["S"]
    D = params["D"]

    Aq = calculate_Aq(J1, J2, J3, S, D, H_coord, K_coord, L)
    Bq = calculate_Bq(J1, J2, J3, S, H_coord, K_coord, L)
    Lq1 = calculate_Lq1(S, J_alt, J_prime_alt, H_coord, K_coord, L)
    Lq2 = calculate_Lq2(S, J_alt, J_prime_alt, H_coord, K_coord, L)
    lam = calculate_lambda(Aq, Bq, Lq1, Lq2)
    return float(np.real_if_close(lam))


def direction_vectors(angle: float) -> np.ndarray:
    """
    (h,k) パラメータ平面で角度 ``angle`` (rad) に対応する単位ベクトル。
    """
    return np.array([math.cos(angle), math.sin(angle)], dtype=float)


def in_domain(h_param: float, k_param: float) -> bool:
    return (
        H_PARAM_RANGE[0] <= h_param <= H_PARAM_RANGE[1]
        and K_PARAM_RANGE[0] <= k_param <= K_PARAM_RANGE[1]
    )


def max_radius_for_direction(direction: np.ndarray) -> float:
    """
    指定方向でパラメータ領域を越えない最大半径を計算する。
    """
    limits = []
    for value, (lower, upper) in zip(direction, (H_PARAM_RANGE, K_PARAM_RANGE)):
        if abs(value) < 1e-12:
            limits.append(math.inf)
            continue
        if value > 0:
            limits.append((upper - 0.0) / value)
        else:
            limits.append((lower - 0.0) / value)
    positive_limits = [limit for limit in limits if limit > 0]
    if not positive_limits:
        raise ValueError("Direction does not permit positive radius within domain.")
    max_radius = min(positive_limits)
    if max_radius <= 0:
        raise ValueError("Direction does not permit positive radius within domain.")
    return max_radius


def param_to_cartesian(h_param: float, k_param: float) -> Tuple[float, float]:
    """
    (h,k) パラメータを実際の逆格子座標 (H,K) に変換する。
    """
    return h_param + k_param, -2.0 * k_param


def lambda_on_ray(params: Dict[str, float], direction: np.ndarray, radius: float) -> float:
    """半径 ``radius`` の位置 (h,k) = radius * direction で λ を計算。"""
    h_param = radius * direction[0]
    k_param = radius * direction[1]
    if not in_domain(h_param, k_param):
        raise ValueError("Point lies outside the specified (h,k) domain.")
    return evaluate_lambda(params, h_param, k_param, FIXED_L)


def find_radius_for_direction(
    params: Dict[str, float],
    direction: np.ndarray,
    lambda_center: float,
) -> float:
    """
    ある方向ベクトルについて ``calculate_lambda == TARGET_ENERGY`` を満たす半径を
    ブラケット探索＋二分法で求める。
    """
    if lambda_center >= TARGET_ENERGY:
        raise ValueError(
            "lambda at the origin is already >= TARGET_ENERGY. "
            "Please adjust TARGET_ENERGY or verify the monotonicity assumption."
        )

    low = 0.0
    max_allowed = min(RADIUS_MAX, max_radius_for_direction(direction))
    high = min(RADIUS_INITIAL, max_allowed)
    lambda_high = lambda_on_ray(params, direction, high)

    # 外側へ向かって λ が TARGET を超えるまで半径を拡大
    while lambda_high < TARGET_ENERGY:
        if high >= max_allowed - 1e-12:
            raise RuntimeError(
                "Failed to bracket TARGET_ENERGY before hitting domain boundary. "
                "Consider enlarging H_PARAM_RANGE/K_PARAM_RANGE or lowering TARGET_ENERGY."
            )
        new_high = min(high * RADIUS_GROWTH, max_allowed)
        if abs(new_high - high) < 1e-12:
            raise RuntimeError(
                "Unable to enlarge radius within domain while bracketing TARGET_ENERGY."
            )
        high = new_high
        lambda_high = lambda_on_ray(params, direction, high)

    # 二分法
    for _ in range(100):
        mid = 0.5 * (low + high)
        lambda_mid = lambda_on_ray(params, direction, mid)

        if abs(lambda_mid - TARGET_ENERGY) < LAMBDA_TOL or (high - low) < BISECTION_TOL:
            return mid

        if lambda_mid < TARGET_ENERGY:
            low = mid
        else:
            high = mid

    # 収束しない場合でも最良の推定値を返す
    return 0.5 * (low + high)


def compute_contour(params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    角度 0〜2π を走査し、各方向で半径を求めて (H, K) 座標を返す。
    """
    if not in_domain(0.0, 0.0):
        raise ValueError("Origin is outside the specified parameter domain.")
    lambda_center = evaluate_lambda(params, 0.0, 0.0, FIXED_L)
    if lambda_center >= TARGET_ENERGY:
        raise ValueError(
            f"lambda(0,0,{FIXED_L})={lambda_center:.6f} "
            f"is not smaller than TARGET_ENERGY={TARGET_ENERGY:.6f}."
        )

    angles = np.linspace(0.0, 2.0 * math.pi, ANGLE_SAMPLES, endpoint=False)
    H_points: List[float] = []
    K_points: List[float] = []

    for angle in angles:
        direction = direction_vectors(angle)
        radius = find_radius_for_direction(params, direction, lambda_center)
        h_param = radius * direction[0]
        k_param = radius * direction[1]
        if not in_domain(h_param, k_param):
            raise RuntimeError("Computed contour point lies outside specified domain.")
        H_coord, K_coord = param_to_cartesian(h_param, k_param)
        H_points.append(H_coord)
        K_points.append(K_coord)

    # 閉曲線にするため最初の点を最後に追加
    H_points.append(H_points[0])
    K_points.append(K_points[0])
    return np.asarray(H_points), np.asarray(K_points)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    fit_csv = base_dir / "fit_results.csv"
    if not fit_csv.exists():
        raise FileNotFoundError(f"fit_results.csv not found: {fit_csv}")

    params = read_fit_results(fit_csv)
    H_curve, K_curve = compute_contour(params)

    fig, ax = plt.subplots()
    ax.plot(H_curve, K_curve, color="tab:orange", label=f"lambda = {TARGET_ENERGY}")
    ax.scatter([0], [0], color="tab:blue", label="Origin")
    ax.set_xlabel("(H,0,0)")
    ax.set_ylabel("(K,-2K,0)")
    ax.set_title(f"Contour of lambda=E (E={TARGET_ENERGY}, L={FIXED_L})")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    output_path = base_dir / f"HK_lambda_{TARGET_ENERGY:.3f}_L_{FIXED_L:.3f}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Contour saved to: {output_path}")


if __name__ == "__main__":
    main()
