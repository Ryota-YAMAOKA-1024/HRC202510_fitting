import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:  # SciPy is optional; main() falls back to Gauss-Newton
    least_squares = None


LM_MAX_ITER = 1000
PARAM_ORDER = ["J1", "J2", "J3", "J_alt", "J_prime_alt", "S", "D"]
INITIAL_CONFIG_FILENAME = "initial_list.txt"
RANDOM_GUESS_COUNT = 10
RANDOM_SEED = 12345
LOGISTIC_EPS = 1e-9
PLOT_ASPECT = {
    # 軸ごとのアスペクト比。Noneなら自動、数値や'equal'などMatplotlibが受け取れる値を指定。
    "H": None,
    "K": None,
    "L": None,
}


@dataclass
class ParamSetting:
    name: str
    initial: float
    locked: bool
    lower: float
    upper: float


def value_to_internal(value: float, lower: float, upper: float) -> float:
    span = upper - lower
    if span <= 0:
        raise ValueError(f"Invalid bounds: [{lower}, {upper}]")
    ratio = (value - lower) / span
    ratio = np.clip(ratio, LOGISTIC_EPS, 1 - LOGISTIC_EPS)
    return np.log(ratio / (1 - ratio))


def logistic_sigmoid(values: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr_flat = np.atleast_1d(arr)
    result = np.empty_like(arr_flat, dtype=float)

    positive_mask = arr_flat >= 0
    if np.any(positive_mask):
        result[positive_mask] = 1.0 / (1.0 + np.exp(-arr_flat[positive_mask]))
    if np.any(~positive_mask):
        exp_vals = np.exp(arr_flat[~positive_mask])
        result[~positive_mask] = exp_vals / (1.0 + exp_vals)

    if arr.ndim == 0:
        return result[0]
    return result.reshape(arr.shape)


def internal_to_value(internal_value: float, lower: float, upper: float) -> float:
    span = upper - lower
    if span <= 0:
        raise ValueError(f"Invalid bounds: [{lower}, {upper}]")
    sigmoid = logistic_sigmoid(internal_value)
    return lower + span * sigmoid


def load_initial_config(config_path: Path) -> list[ParamSetting]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} が見つかりません。")

    parsed: dict[str, ParamSetting] = {}
    with config_path.open() as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.replace("\t", ",").split(",") if part.strip()]
            if len(parts) < 3:
                raise ValueError(f"不正な設定行です: {raw_line}")

            name = parts[0]
            if name not in PARAM_ORDER:
                raise ValueError(f"未知のパラメータ名です: {name}")

            if name in parsed:
                raise ValueError(f"{name} が重複定義されています。")

            try:
                initial = float(parts[1])
            except ValueError as exc:
                raise ValueError(f"{name} の初期値が不正です: {parts[1]}") from exc

            status = parts[2].lower()
            if status == "lock":
                parsed[name] = ParamSetting(name, initial, True, initial, initial)
            elif status == "free":
                if len(parts) < 5:
                    raise ValueError(f"{name} は free ですが下限・上限が不足しています。")
                try:
                    lower = float(parts[3])
                    upper = float(parts[4])
                except ValueError as exc:
                    raise ValueError(f"{name} の下限/上限が不正です: {parts[3:5]}") from exc
                if upper <= lower:
                    raise ValueError(f"{name} の下限と上限が矛盾しています: [{lower}, {upper}]")
                if not (lower <= initial <= upper):
                    raise ValueError(f"{name} の初期値 {initial} が範囲 [{lower}, {upper}] 外です。")
                parsed[name] = ParamSetting(name, initial, False, lower, upper)
            else:
                raise ValueError(f"{name} の状態が不正です (lock/free): {parts[2]}")

    settings: list[ParamSetting] = []
    for name in PARAM_ORDER:
        if name not in parsed:
            raise ValueError(f"{name} の設定が initial_list に存在しません。")
        settings.append(parsed[name])
    return settings


def unpack_params(internal_vector: Sequence[float], settings: Sequence[ParamSetting]) -> np.ndarray:
    params = np.empty(len(settings), dtype=float)
    internal_vector = np.asarray(internal_vector, dtype=float)
    free_index = 0
    for idx, setting in enumerate(settings):
        if setting.locked:
            params[idx] = setting.initial
        else:
            params[idx] = internal_to_value(internal_vector[free_index], setting.lower, setting.upper)
            free_index += 1
    if free_index != internal_vector.size:
        raise ValueError("内部パラメータの次元が設定と一致しません。")
    return params


def build_initial_internal_vector(settings: Sequence[ParamSetting]) -> np.ndarray:
    internal_values = []
    for setting in settings:
        if setting.locked:
            continue
        internal_values.append(value_to_internal(setting.initial, setting.lower, setting.upper))
    return np.array(internal_values, dtype=float)


def sample_random_internal(settings: Sequence[ParamSetting], rng: np.random.Generator) -> np.ndarray:
    samples = []
    for setting in settings:
        if setting.locked:
            continue
        span = setting.upper - setting.lower
        margin = span * LOGISTIC_EPS
        value = rng.uniform(setting.lower + margin, setting.upper - margin)
        samples.append(value_to_internal(value, setting.lower, setting.upper))
    return np.array(samples, dtype=float)


def residuals_internal(
    internal_vector: Sequence[float],
    settings: Sequence[ParamSetting],
    H_vals: np.ndarray,
    K_vals: np.ndarray,
    L_vals: np.ndarray,
    E_vals: np.ndarray,
) -> np.ndarray:
    params = unpack_params(internal_vector, settings)
    return residuals(params, H_vals, K_vals, L_vals, E_vals)


def calculate_Aq(J1, J2, J3, S, D, H, K, L):
    """
    Diagonal term (Aq) を計算します。

    Args:
        J1, J2, J3 (float): 交換相互作用パラメータ
        S (float): スピン量子数
        D (float): 一軸異方性などの追加項
        H, K, L (float): 逆格子空間の座標

    Returns:
        float: 計算されたAqの値
    """
    term1 = 2 * J1 * S
    cos_term = np.cos(2 * np.pi * H) + np.cos(2 * np.pi * K) + np.cos(2 * np.pi * (H + K))
    term2 = 2 * J2 * S * (cos_term - 3)
    term3 = 12 * J3 * S

    return term1 + term2 + term3 + D


def calculate_Bq(J1, J2, J3, S, H, K, L):
    """
    Off-diagonal term (Bq) を計算します。

    Args:
        J1, J3 (float): 交換相互作用パラメータ
        S (float): スピン量子数
        H, K, L (float): 逆格子空間の座標

    Returns:
        float: 計算されたBqの値
    """
    term1 = 2 * J1 * S * np.cos(np.pi * L)

    cos_sum = (
        np.cos(2 * np.pi * (H + L / 2))
        + np.cos(2 * np.pi * (K + L / 2))
        + np.cos(2 * np.pi * (H + K + L / 2))
        + np.cos(2 * np.pi * (H - L / 2))
        + np.cos(2 * np.pi * (K - L / 2))
        + np.cos(2 * np.pi * (H + K - L / 2))
    )
    term2 = 2 * J3 * S * cos_sum

    return term1 + term2


def calculate_Lq1(S, J_alt, J_prime_alt, H, K, L):
    """
    Lq(1) を計算します。

    Args:
        S (float): スピン量子数
        J_alt (float): 交換相互作用パラメータ
        J_prime_alt (float): もう一つの交換相互作用パラメータ
        H, K, L (float): 逆格子空間の座標

    Returns:
        float: 計算されたLq(1)の値
    """
    cos_sum_1 = (
        np.cos(2 * np.pi * (2 * H + K + L))
        + np.cos(2 * np.pi * (-H + K + L))
        + np.cos(2 * np.pi * (-H - 2 * K + L))
    )
    cos_sum_2 = (
        np.cos(2 * np.pi * (H + 2 * K + L))
        + np.cos(2 * np.pi * (H - K + L))
        + np.cos(2 * np.pi * (-2 * H - K + L))
    )

    return 2 * S * (J_prime_alt * (-3 + cos_sum_1) + J_alt * (-3 + cos_sum_2))


def calculate_Lq2(S, J_alt, J_prime_alt, H, K, L):
    """
    Lq(2) を計算します。

    Args:
        S (float): スピン量子数
        J_alt (float): 交換相互作用パラメータ
        J_prime_alt (float): もう一つの交換相互作用パラメータ
        H, K, L (float): 逆格子空間の座標

    Returns:
        float: 計算されたLq(2)の値
    """
    cos_term1 = (
        np.cos(2 * np.pi * (2 * H + K + L))
        + np.cos(2 * np.pi * (-H + K + L))
        + np.cos(2 * np.pi * (-H - 2 * K + L))
    )
    term1 = J_alt * (-3 + cos_term1)

    cos_term2 = (
        np.cos(2 * np.pi * (H + 2 * K + L))
        + np.cos(2 * np.pi * (H - K + L))
        + np.cos(2 * np.pi * (-2 * H - K + L))
    )
    term2 = J_prime_alt * (-3 + cos_term2)

    return 2 * S * (term1 + term2)


def calculate_lambda(Aq, Bq, Lq1, Lq2):
    """
    固有値λを計算します。

    Args:
        Aq (float): 対角項
        Bq (float): 非対角項
        Lq1 (float): Lq(1)
        Lq2 (float): Lq(2)

    Returns:
        float: 正の固有値（λ+）
    """
    diff_term_half = (Lq1 - Lq2) / 2
    inside_sqrt = diff_term_half**2 + (Aq + Lq1) * (Aq + Lq2) - Bq**2

    if np.isscalar(inside_sqrt):
        sqrt_part = np.sqrt(inside_sqrt + 0j) if inside_sqrt < 0 else np.sqrt(inside_sqrt)
    else:
        inside_array = np.asarray(inside_sqrt)
        sqrt_part = np.sqrt(inside_array + 0j) if np.any(inside_array < 0) else np.sqrt(inside_array)

    lambda_plus = diff_term_half + sqrt_part

    return np.real_if_close(lambda_plus)


def resolve_input_file(base_dir: Path) -> Path:
    """
    list.csv または list.txt を探して返します。
    優先順位: list.csv -> list.txt
    """
    for name in ("list.csv", "list.txt"):
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("list.csv または list.txt が見つかりません。")


def load_dataset(csv_path):
    """
    list.csvからデータを読み込みます。

    Args:
        csv_path (Path): 読み込み対象のCSVファイルパス

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: H, K, L, Eの配列
    """
    H_vals, K_vals, L_vals, E_vals = [], [], [], []

    with Path(csv_path).open(newline="") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row:
                continue
            if row[0].lstrip().startswith("#"):
                continue
            if len(row) < 4:
                raise ValueError(f"Invalid row (expected at least 4 columns): {row}")
            H_vals.append(float(row[0]))
            K_vals.append(float(row[1]))
            L_vals.append(float(row[2]))
            E_vals.append(float(row[3]))

    if not H_vals:
        raise ValueError("list.csvに有効なデータがありません。")

    return np.array(H_vals), np.array(K_vals), np.array(L_vals), np.array(E_vals)


def model_lambda(params, H_vals, K_vals, L_vals):
    """
    パラメータからλを計算します。

    Args:
        params (Sequence[float]): [J1, J2, J3, J_alt, J_prime_alt, S, D]
        H_vals, K_vals, L_vals (np.ndarray): 逆格子空間の座標

    Returns:
        np.ndarray: λの配列
    """
    J1, J2, J3, J_alt, J_prime_alt, S, D = params
    lambdas = []

    for H, K, L in zip(H_vals, K_vals, L_vals):
        Aq = calculate_Aq(J1, J2, J3, S, D, H, K, L)
        Bq = calculate_Bq(J1, J2, J3, S, H, K, L)
        Lq1 = calculate_Lq1(S, J_alt, J_prime_alt, H, K, L)
        Lq2 = calculate_Lq2(S, J_alt, J_prime_alt, H, K, L)
        lambdas.append(calculate_lambda(Aq, Bq, Lq1, Lq2))

    return np.asarray(lambdas, dtype=np.complex128)


def residuals(params, H_vals, K_vals, L_vals, E_vals):
    """
    λと測定量Eとの差を計算します。

    Returns:
        np.ndarray: 残差ベクトル
    """
    lambda_vals = model_lambda(params, H_vals, K_vals, L_vals)
    # 測定値Eは実数と仮定し、λの実部で残差を構築
    return np.real(lambda_vals) - E_vals


def numerical_jacobian(func, params, args, eps: float = 1e-6):
    """
    数値微分でヤコビアンを計算します。
    """
    x = np.asarray(params, dtype=float)
    base_residuals = func(x, *args)
    jac = np.empty((base_residuals.size, x.size), dtype=float)

    for idx in range(x.size):
        step = eps * max(1.0, abs(x[idx]))
        delta = np.zeros_like(x)
        delta[idx] = step
        shifted = func(x + delta, *args)
        jac[:, idx] = (shifted - base_residuals) / step

    return jac


def gauss_newton(func, initial_guess, args, max_iter: int = LM_MAX_ITER, tol: float = 1e-8):
    """
    SciPyが利用できない場合のLevenberg-Marquardt型最適化。
    """
    x = np.asarray(initial_guess, dtype=float)
    lam = 1e-3

    for _ in range(max_iter):
        res = func(x, *args)
        cost = 0.5 * float(np.dot(res, res))
        jac = numerical_jacobian(func, x, args)

        try:
            jtj = jac.T @ jac
            grad = jac.T @ res
            diag = np.diag(jtj)
            damping = lam * np.diag(np.maximum(diag, 1e-8))
            step = np.linalg.solve(jtj + damping, -grad)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        if np.linalg.norm(step) <= tol * (1.0 + np.linalg.norm(x)):
            return x, res, True, "収束しました。"

        candidate = x + step
        candidate_res = func(candidate, *args)
        candidate_cost = 0.5 * float(np.dot(candidate_res, candidate_res))

        if candidate_cost < cost:
            x = candidate
            lam = max(lam / 3.0, 1e-7)
            if abs(cost - candidate_cost) <= tol * (1.0 + candidate_cost):
                return x, candidate_res, True, "コスト変化が許容範囲内です。"
        else:
            lam *= 3.0

    return x, func(x, *args), False, "最大反復回数に到達しました。"


def fit_parameters(
    initial_internal: np.ndarray,
    settings: Sequence[ParamSetting],
    H_vals: np.ndarray,
    K_vals: np.ndarray,
    L_vals: np.ndarray,
    E_vals: np.ndarray,
):
    """
    最小二乗法でパラメータをフィットします。

    Returns:
        tuple: (result_params, residual_sum, success, message)
    """
    initial_internal = np.asarray(initial_internal, dtype=float)
    free_param_count = initial_internal.size

    if free_param_count == 0:
        params = unpack_params(initial_internal, settings)
        residual_vec = residuals(params, H_vals, K_vals, L_vals, E_vals)
        residual_sum = float(np.sum(residual_vec**2))
        return params, residual_sum, True, "全パラメータが固定されています。"

    args = (settings, H_vals, K_vals, L_vals, E_vals)

    if least_squares is not None:
        result = least_squares(
            residuals_internal,
            initial_internal,
            args=args,
        )
        final_internal = result.x
        residual_vec = result.fun
        params = unpack_params(final_internal, settings)
        residual_sum = float(np.sum(residual_vec**2))
        return params, residual_sum, bool(result.success), result.message

    final_internal, residual_vec, success, message = gauss_newton(
        residuals_internal, initial_internal, args, max_iter=LM_MAX_ITER
    )
    params = unpack_params(final_internal, settings)
    residual_sum = float(np.sum(residual_vec**2))
    return params, residual_sum, success, message


def save_fit_results(
    output_path: Path,
    settings: Sequence[ParamSetting],
    params: Sequence[float],
    residual_sum: float,
    success: bool,
    message: str,
    initial_guess: Sequence[float],
):
    """
    フィット結果をCSVに保存します。
    """
    bounds_text = []
    for setting in settings:
        if setting.locked:
            bounds_text.append(f"[{setting.initial}, {setting.initial}]")
        else:
            bounds_text.append(f"[{setting.lower}, {setting.upper}]")

    with Path(output_path).open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["parameter", "value", "status", "bounds"])
        for setting, value, bound in zip(settings, params, bounds_text):
            status = "locked" if setting.locked else "free"
            writer.writerow([setting.name, value, status, bound])
        writer.writerow([])
        writer.writerow(["residual_sum", residual_sum])
        writer.writerow(["success", success])
        writer.writerow(["message", message])
        writer.writerow(
            [
                "initial_guess",
                " ".join(f"{value:.6g}" for value in initial_guess),
            ]
        )


def generate_plots(
    H_vals: Sequence[float],
    K_vals: Sequence[float],
    L_vals: Sequence[float],
    measured_E: Sequence[float],
    fitted_E: Sequence[float],
    output_dir: Path,
    params: Sequence[float],
):
    """
    H-E, K-E, L-E の散布図とフィット結果を保存します。
    """
    measured_array = np.asarray(measured_E, dtype=float)
    fitted_array = np.asarray(fitted_E, dtype=float)

    def compute_curve(axis_key: str, x_vals: np.ndarray) -> np.ndarray:
        x_vals = np.asarray(x_vals, dtype=float)
        zeros = np.zeros_like(x_vals)
        if axis_key == "H":
            H_arr, K_arr, L_arr = x_vals, zeros, zeros
        elif axis_key == "K":
            H_arr, K_arr, L_arr = zeros, x_vals, zeros
        elif axis_key == "L":
            H_arr, K_arr, L_arr = zeros, zeros, x_vals
        else:
            raise ValueError(f"Unknown axis key: {axis_key}")
        return np.real_if_close(model_lambda(params, H_arr, K_arr, L_arr))

    def plot_axis(
        x_vals: Iterable[float],
        file_name: str,
        xlabel: str,
        axis_key: str,
        xlim=None,
    ):
        x_array = np.asarray(x_vals, dtype=float)

        fig, ax = plt.subplots()
        ax.scatter(x_array, measured_array, label="experimental data", color="tab:blue")
        if xlim is not None:
            xmin, xmax = xlim
        else:
            xmin, xmax = float(np.min(x_array)), float(np.max(x_array))
        curve_x = np.linspace(xmin, xmax, 400)
        curve_y = compute_curve(axis_key, curve_x)
        ax.plot(curve_x, curve_y, label="Fit curve", color="tab:orange")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("E")
        aspect = PLOT_ASPECT.get(axis_key)
        if aspect is not None:
            ax.set_aspect(aspect)
        ax.set_ylim(bottom=0)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / file_name, dpi=300)
        plt.close(fig)

    plot_axis(H_vals, "H00_E.png", "(H,0,3)", axis_key="H", xlim=(0, 1))
    plot_axis(K_vals, "K-K0_E.png", "(K,-2K,3)", axis_key="K", xlim=(0, 1))
    plot_axis(L_vals, "00L_E.png", "(0,0,L) ", axis_key="L", xlim=(3, 4))


def main():
    base_dir = Path(__file__).resolve().parent
    input_csv = resolve_input_file(base_dir)
    output_csv = base_dir / "fit_results.csv"
    config_path = base_dir / INITIAL_CONFIG_FILENAME

    H_vals, K_vals, L_vals, E_vals = load_dataset(input_csv)

    settings = load_initial_config(config_path)

    initial_internal = build_initial_internal_vector(settings)
    initial_vectors = [initial_internal]

    free_param_count = initial_internal.size
    if RANDOM_GUESS_COUNT > 0 and free_param_count > 0:
        rng = np.random.default_rng(RANDOM_SEED)
        for _ in range(RANDOM_GUESS_COUNT):
            initial_vectors.append(sample_random_internal(settings, rng))

    best_result = None

    for internal_guess in initial_vectors:
        params_guess = np.array(unpack_params(internal_guess, settings), dtype=float)
        params, residual_sum, success, message = fit_parameters(
            internal_guess, settings, H_vals, K_vals, L_vals, E_vals
        )
        if best_result is None or residual_sum < best_result["residual_sum"]:
            best_result = {
                "params": params,
                "residual_sum": residual_sum,
                "success": success,
                "message": message,
                "initial_guess": params_guess,
            }

    if best_result is None:
        raise RuntimeError("フィットに失敗しました。初期条件を見直してください。")

    params = best_result["params"]
    residual_sum = best_result["residual_sum"]
    success = best_result["success"]
    message = best_result["message"]
    initial_guess_used = best_result["initial_guess"]

    save_fit_results(
        output_csv,
        settings,
        params,
        residual_sum,
        success,
        message,
        initial_guess_used,
    )

    fitted_E = np.real(model_lambda(params, H_vals, K_vals, L_vals))
    generate_plots(H_vals, K_vals, L_vals, E_vals, fitted_E, base_dir, params)


if __name__ == "__main__":
    main()
