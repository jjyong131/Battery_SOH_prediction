"""
oxford_preprocess.py

Oxford Battery Degradation Dataset 1 전처리 파이프라인:

1) .mat -> long-format CSV
2) long CSV -> cell 별 CSV 분리
3) cell별 IC curve 계산 + 600pt 보간 (wide 형식)
4) 보간된 IC curve로 슬라이딩 윈도우 입력(X), 전체 커브(Y) 생성
   + train/valid/test split 및 StandardScaler 적용

Colab에서 쓰던 코드를 프로젝트용 함수 구조로 정리한 버전.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================
# 공통 경로 설정
# =========================
DATA_ROOT = Path(".")  # 프로젝트 루트 기준
MAT_FILE  = DATA_ROOT / "data" / "raw" / "Oxford_Battery_Degradation_Dataset_1.mat"

LONG_CSV  = DATA_ROOT / "data" / "interim" / "Oxford_Battery_Degradation_Dataset_1_long.csv"
CELL_DIR  = DATA_ROOT / "data" / "interim" / "oxford_by_cell"
IC_DIR    = DATA_ROOT / "data" / "interim" / "oxford_ic_interpolated"

PROCESSED_DIR = DATA_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1) .mat → long-format CSV
# =========================
def convert_oxford_mat_to_csv(
    mat_filepath: Path,
    save_filepath: Path
) -> None:
    """
    Oxford Battery Degradation Dataset 1 .mat 파일을 long-format CSV로 변환.
    columns: [Cell_ID, Cycle, Type, Time_s, Voltage_V, Charge_mAh, Temp_C]
    """
    try:
        print(f"[MAT→CSV] '{mat_filepath}' 파일을 로드합니다...")
        mat_data = scipy.io.loadmat(mat_filepath, simplify_cells=True)
        print("파일 로드 완료.")

        all_dfs = []

        # 1단계: Cell1, Cell2, ...
        cell_ids = [k for k in mat_data.keys() if k.startswith("Cell")]
        print(f"파일에서 {len(cell_ids)}개의 배터리 셀을 찾았습니다: {cell_ids}")

        for cell_id_str in cell_ids:
            print(f"\n'{cell_id_str}' 데이터 처리 중...")
            cell_id_num = int("".join(filter(str.isdigit, cell_id_str)))

            # Cell1 자체가 cycle들을 담고 있으므로 .cycle 접근 제거
            cycle_struct = mat_data[cell_id_str]

            # 2단계: cyc0100, cyc0200, ...
            cycle_names = list(cycle_struct.keys())

            for cycle_name_str in tqdm(cycle_names, desc=f"Processing {cell_id_str}"):
                cycle_num = int("".join(filter(str.isdigit, cycle_name_str)))

                # 3단계: C1ch, C1dc, ...
                op_struct = cycle_struct[cycle_name_str]
                op_names = list(op_struct.keys())

                for op_name_str in op_names:
                    meas = op_struct[op_name_str]

                    df_temp = pd.DataFrame(
                        {
                            "Time_s": meas["t"],
                            "Voltage_V": meas["v"],
                            "Charge_mAh": meas["q"],
                            "Temp_C": meas["T"],
                        }
                    )
                    df_temp["Cell_ID"] = cell_id_num
                    df_temp["Cycle"] = cycle_num
                    df_temp["Type"] = op_name_str

                    all_dfs.append(df_temp)

        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df = final_df[
            ["Cell_ID", "Cycle", "Type", "Time_s", "Voltage_V", "Charge_mAh", "Temp_C"]
        ]

        save_filepath.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(save_filepath, index=False)
        print(
            f"\n✔ 변환 완료! 총 {len(final_df)}개 행의 데이터가 '{save_filepath}'에 저장되었습니다."
        )

        print("\n--- 변환된 데이터 샘플 ---")
        print(final_df.head())

    except FileNotFoundError:
        print(f"\n[오류] 파일을 찾을 수 없습니다: '{mat_filepath}'")
    except Exception as e:
        print(f"\n[오류] 데이터 변환 중 문제가 발생했습니다: {e}")


# =========================
# 2) long CSV → cell별 CSV 분리
# =========================
def split_long_csv_by_cell(
    input_csv: Path,
    output_dir: Path,
    filename_prefix: str = "oxford_cell",
    cell_id_col: str = "Cell_ID",
) -> None:
    """
    long-format CSV를 Cell_ID로 그룹화하여
    각 셀마다 별도의 CSV 파일 생성.
    예: oxford_cell_1.csv, oxford_cell_2.csv, ...
    """
    try:
        print(f"[Split] '{input_csv}' 파일을 불러옵니다...")
        df = pd.read_csv(input_csv)
        print("로드 완료.")

        output_dir.mkdir(parents=True, exist_ok=True)

        unique_cell_ids = sorted(df[cell_id_col].unique())
        print(f"다음 Cell ID들을 찾았습니다: {unique_cell_ids}")

        print("\n각 Cell ID별로 CSV 파일을 분리하여 저장합니다...")
        for cell_id in unique_cell_ids:
            cell_df = df[df[cell_id_col] == cell_id].copy()
            out_name = f"{filename_prefix}_{cell_id}.csv"
            out_path = output_dir / out_name
            cell_df.to_csv(out_path, index=False)
            print(f"✔ '{out_name}' ({len(cell_df)} 행) 저장 완료.")

        print("\n모든 파일 분리가 완료되었습니다.")

    except FileNotFoundError:
        print(f"\n[오류] 원본 파일을 찾을 수 없습니다: '{input_csv}'")
    except KeyError:
        print(f"\n[오류] '{cell_id_col}' 컬럼을 찾을 수 없습니다. 컬럼 이름을 확인해주세요.")
    except Exception as e:
        print(f"\n작업 중 오류가 발생했습니다: {e}")


# =========================
# 3) cell별 IC curve 계산 + 보간 (wide)
# =========================

def _sigma_pts_from_volt(width_v: float, dV: float, min_sigma: float = 0.5) -> float:
    if dV <= 0 or width_v is None or width_v <= 0:
        return 1.0
    return max(width_v / dV, min_sigma)


def calculate_ic_from_vq(
    voltage: np.ndarray,
    charge_mAh: np.ndarray,
    n_points: int = 800,
    sigma_v: float = 0.015,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oxford 데이터(Voltage, Charge_mAh)로부터
    가우시안 스무딩 + 미분 기반 IC(dQ/dV) 계산.
    """
    voltage = np.asarray(voltage, float)
    charge = np.asarray(charge_mAh, float) / 1000.0  # mAh → Ah

    if len(voltage) < 10:
        return np.array([]), np.array([])

    mask = np.isfinite(voltage) & np.isfinite(charge)
    v_m, q_m = voltage[mask], charge[mask]
    order = np.argsort(v_m)
    v_sorted, q_sorted = v_m[order], q_m[order]

    uniq, idx = np.unique(v_sorted, return_index=True)
    v_unique, q_unique = v_sorted[idx], q_sorted[idx]
    if len(v_unique) < 2:
        return np.array([]), np.array([])

    v_grid = np.linspace(v_unique.min(), v_unique.max(), n_points)
    q_grid = np.interp(v_grid, v_unique, q_unique)

    dV = float(np.diff(v_grid).mean())
    sigma_pts = _sigma_pts_from_volt(sigma_v, dV)
    dQdIdx = gaussian_filter1d(q_grid, sigma=sigma_pts, order=1, mode="nearest")
    ic = dQdIdx / dV

    return v_grid, np.abs(ic)


def build_ic_interpolated_per_cell(
    cell_dir: Path,
    out_dir: Path,
    cells_to_process: List[int],
    operation_type: str = "C1ch",
    interpolation_v_min: float = 3.4,
    interpolation_v_max: float = 4.0,
    interpolation_n_points: int = 600,
    sigma_v: float = 0.015,
) -> None:
    """
    Cell별 long-format 파일에서:
      - 각 cycle에 대해 IC curve 계산
      - (interpolation_v_min ~ interpolation_v_max) 구간을 N_POINTS로 보간
      - wide 형식으로 저장: ['cycle', 'v_0'...'v_{N-1}']
    """
    if not cell_dir.exists():
        print(f"[오류] 입력 폴더를 찾을 수 없습니다: '{cell_dir}'")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for cell_id in cells_to_process:
        file_path = cell_dir / f"oxford_cell_{cell_id}.csv"
        try:
            df = pd.read_csv(file_path)
            df_op = df[df["Type"] == operation_type].copy()

            if df_op.empty:
                print(
                    f"\n[{file_path}] 파일에 '{operation_type}' 타입 데이터가 없습니다. 건너뜁니다."
                )
                continue

            unique_cycles = sorted(df_op["Cycle"].unique())
            interpolated_results = []

            print(f"\n--- Cell #{cell_id} ({file_path.name}) 처리 중 ---")

            for cycle_num in tqdm(unique_cycles, desc=f"Cell {cell_id}"):
                cycle_df = df_op[df_op["Cycle"] == cycle_num]

                v_native, ic_native = calculate_ic_from_vq(
                    voltage=cycle_df["Voltage_V"].values,
                    charge_mAh=cycle_df["Charge_mAh"].values,
                    n_points=800,
                    sigma_v=sigma_v,
                )
                if len(v_native) == 0:
                    continue

                v_target = np.linspace(
                    interpolation_v_min, interpolation_v_max, interpolation_n_points
                )
                ic_interp = np.interp(
                    v_target, v_native, ic_native, left=0.0, right=0.0
                )

                row = {"cycle": int(cycle_num)}
                for i in range(interpolation_n_points):
                    row[f"v_{i}"] = ic_interp[i]
                interpolated_results.append(row)

            if interpolated_results:
                df_cell_interpolated = pd.DataFrame(interpolated_results)
                out_name = f"oxford_cell_{cell_id}_ic_interpolated.csv"
                out_path = out_dir / out_name
                df_cell_interpolated.to_csv(out_path, index=False)
                print(f"✔ Cell #{cell_id}의 보간된 IC Curve → '{out_path}'")
            else:
                print(f"Cell #{cell_id}: 처리할 유효 사이클 없음.")

        except FileNotFoundError:
            print(f"\n[오류] 파일을 찾을 수 없습니다: '{file_path}'")
        except Exception as e:
            print(f"\n[오류] '{file_path}' 처리 중 문제가 발생했습니다: {e}")


# =========================
# 4) 슬라이딩 윈도우 + 스케일링
# =========================
def build_sliding_window_dataset(
    ic_dir: Path,
    cells_to_process: List[int],
    voltage_min: float = 3.4,
    voltage_max: float = 4.0,
    n_points: int = 600,
    window_v_min: float = 3.7,
    window_v_max: float = 3.9,
    n_windows: int = 10,
    window_width_v: float = 0.14,
) -> Dict[str, np.ndarray]:
    """
    보간된 IC Curve(wide, 600pt)를 사용해:

      X: 전압 구간 [window_v_min, window_v_min+window_width_v] 를
         여러 위치로 슬라이딩하며 입력 윈도우로 사용
      Y: 전체 600포인트 IC curve

    -> X, Y 생성 후 train/valid/test split + StandardScaler 적용.
    반환:
      {
        'X_train_scaled': ... (N_train, input_len, 1),
        'X_valid_scaled': ...,
        'X_test_scaled' : ...,
        'Y_train_scaled': ...,
        'Y_valid_scaled': ...,
        'Y_test_scaled' : ...,
        'scaler_x': scaler_x, 'scaler_y': scaler_y
      }
    """
    all_df_list = []
    print("보간된 IC Curve 파일들을 로드합니다...")
    for cell_id in cells_to_process:
        fpath = ic_dir / f"oxford_cell_{cell_id}_ic_interpolated.csv"
        try:
            df = pd.read_csv(fpath)
            df["battery"] = f"Cell{cell_id}"
            all_df_list.append(df)
        except FileNotFoundError:
            print(f"[경고] 파일 없음: {fpath}")

    if not all_df_list:
        raise RuntimeError("[오류] 사용할 IC interpolated 파일이 없습니다.")

    all_df = pd.concat(all_df_list, ignore_index=True)
    print("모든 Cell 데이터 통합 완료.")

    # 전압축 + 입력 창 포인트 개수 계산
    v_ref = np.linspace(voltage_min, voltage_max, n_points)
    dv_per_point = (voltage_max - voltage_min) / (n_points - 1)
    input_points_len = int(np.round(window_width_v / dv_per_point))
    print(
        f"\n입력 윈도우 전압 폭 {window_width_v:.2f}V → 약 {input_points_len} 포인트."
    )

    sliding_start_idx = np.argmin(np.abs(v_ref - window_v_min))
    sliding_end_idx = np.argmin(np.abs(v_ref - window_v_max))

    window_start_indices = np.linspace(
        sliding_start_idx,
        sliding_end_idx - input_points_len,
        n_windows,
        dtype=int,
    )

    # X, Y 생성
    print("\n슬라이딩 윈도우로 학습 데이터를 생성합니다...")
    curve_cols = [f"v_{i}" for i in range(n_points)]
    all_curves = all_df[curve_cols].values
    all_meta = all_df[["battery", "cycle"]].to_dict("records")

    new_X, new_Y, new_meta = [], [], []

    for i in tqdm(range(len(all_curves)), desc="Generating Samples"):
        full_curve = all_curves[i]
        meta_info = all_meta[i]

        for start_idx in window_start_indices:
            end_idx = start_idx + input_points_len

            x_window = full_curve[start_idx:end_idx]  # 입력 부분
            new_X.append(x_window)

            # 출력: 전체 600포인트 커브
            new_Y.append(full_curve)

            new_meta.append(meta_info)

    X = np.array(new_X, dtype=np.float32)  # (N_samples, input_len)
    Y = np.array(new_Y, dtype=np.float32)  # (N_samples, 600)
    meta_df = pd.DataFrame(new_meta)

    print(
        f"\n[데이터] 총 {len(all_df)}개 사이클로부터 {len(X)}개 샘플 생성."
    )
    print(f"최종 X={X.shape}, Y={Y.shape}")

    # train/valid/test 분할 + 스케일링
    print("\n데이터를 분할하고 스케일링합니다...")
    X_temp, X_test, Y_temp, Y_test, meta_temp, meta_test = train_test_split(
        X, Y, meta_df, test_size=0.20, random_state=42, shuffle=True
    )
    X_train, X_valid, Y_train, Y_valid, meta_train, meta_valid = train_test_split(
        X_temp, Y_temp, meta_temp, test_size=0.25, random_state=42, shuffle=True
    )

    scaler_x = StandardScaler().fit(X_train)
    X_train_scaled = scaler_x.transform(X_train)
    X_valid_scaled = scaler_x.transform(X_valid)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_y = StandardScaler().fit(Y_train)
    Y_train_scaled = scaler_y.transform(Y_train)
    Y_valid_scaled = scaler_y.transform(Y_valid)
    Y_test_scaled = scaler_y.transform(Y_test)

    # Conv1D/Transformer 입력용 3D로 reshape
    X_train_scaled = X_train_scaled.reshape(
        (X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    )
    X_valid_scaled = X_valid_scaled.reshape(
        (X_valid_scaled.shape[0], X_valid_scaled.shape[1], 1)
    )
    X_test_scaled = X_test_scaled.reshape(
        (X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    )

    print("\n--- 데이터 분할 및 전처리 결과 ---")
    print(f"Train: X={X_train_scaled.shape}, Y={Y_train_scaled.shape}")
    print(f"Valid: X={X_valid_scaled.shape}, Y={Y_valid_scaled.shape}")
    print(f"Test : X={X_test_scaled.shape},  Y={Y_test_scaled.shape}")

    return {
        "X_train_scaled": X_train_scaled,
        "X_valid_scaled": X_valid_scaled,
        "X_test_scaled": X_test_scaled,
        "Y_train_scaled": Y_train_scaled,
        "Y_valid_scaled": Y_valid_scaled,
        "Y_test_scaled": Y_test_scaled,
        "meta_train": meta_train,
        "meta_valid": meta_valid,
        "meta_test": meta_test,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
    }


# =========================
# 전체 파이프라인 실행 entrypoint
# =========================
def run_full_oxford_preprocess(
    cells_to_process: List[int] = list(range(1, 9))
) -> Dict[str, np.ndarray]:
    """
    1) .mat -> long CSV
    2) long -> cell별 CSV
    3) cell별 IC curve 보간
    4) 슬라이딩 윈도우 + split + 스케일링

    이미 중간 산출물이 있으면, 필요한 부분만 다시 실행하도록
    사용자가 직접 조절해도 됨.
    """
    # 1) mat → long
    if not LONG_CSV.exists():
        convert_oxford_mat_to_csv(MAT_FILE, LONG_CSV)
    else:
        print(f"[Skip] long CSV 이미 존재: {LONG_CSV}")

    # 2) long → cell별
    if not CELL_DIR.exists() or not any(CELL_DIR.glob("oxford_cell_*.csv")):
        split_long_csv_by_cell(LONG_CSV, CELL_DIR, filename_prefix="oxford_cell")
    else:
        print(f"[Skip] cell별 CSV 이미 존재: {CELL_DIR}")

    # 3) cell별 IC 보간
    if not IC_DIR.exists() or not any(IC_DIR.glob("oxford_cell_*_ic_interpolated.csv")):
        build_ic_interpolated_per_cell(
            cell_dir=CELL_DIR,
            out_dir=IC_DIR,
            cells_to_process=cells_to_process,
            operation_type="C1ch",
            interpolation_v_min=3.4,
            interpolation_v_max=4.0,
            interpolation_n_points=600,
            sigma_v=0.015,
        )
    else:
        print(f"[Skip] 보간된 IC CSV 이미 존재: {IC_DIR}")

    # 4) 슬라이딩 윈도우 + split + 스케일링
    out = build_sliding_window_dataset(
        ic_dir=IC_DIR,
        cells_to_process=cells_to_process,
        voltage_min=3.4,
        voltage_max=4.0,
        n_points=600,
        window_v_min=3.7,
        window_v_max=3.9,
        n_windows=10,
        window_width_v=0.14,
    )

    # 필요하다면 npy로도 저장 가능
    np.save(PROCESSED_DIR / "X_train_scaled.npy", out["X_train_scaled"])
    np.save(PROCESSED_DIR / "X_valid_scaled.npy", out["X_valid_scaled"])
    np.save(PROCESSED_DIR / "X_test_scaled.npy", out["X_test_scaled"])
    np.save(PROCESSED_DIR / "Y_train_scaled.npy", out["Y_train_scaled"])
    np.save(PROCESSED_DIR / "Y_valid_scaled.npy", out["Y_valid_scaled"])
    np.save(PROCESSED_DIR / "Y_test_scaled.npy", out["Y_test_scaled"])
    print(f"\n[저장] 전처리된 데이터들을 {PROCESSED_DIR}에 npy로 저장했습니다.")

    return out


if __name__ == "__main__":
    # Colab처럼 전체 파이프라인 한 번에 돌리고 싶을 때:
    run_full_oxford_preprocess(cells_to_process=[1, 2, 3, 4, 5, 6, 7, 8])
