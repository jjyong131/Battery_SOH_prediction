"""
oxford_ic_reconstruct_model.py

Oxford IC curve (슬라이딩 윈도우) → 전체 600pt IC curve 재구성 Transformer 모델 학습 & 시각화 스크립트.

구성:
1) 슬라이딩 윈도우 기반 데이터셋 생성 + 스케일링
2) Pure Transformer 모델 정의/학습
3) 윈도우별·셀별 재구성 결과 시각화
"""

import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# =====================================
# 1) 설정
# =====================================

INPUT_DIR = "oxford_ic_interpolated"     # 보간된 IC curve CSV들이 있는 폴더
CELLS_TO_PROCESS = [1, 2, 3, 4, 5, 6, 7, 8]

# IC Curve 전체 정의
VOLTAGE_MIN = 3.4
VOLTAGE_MAX = 4.0
N_POINTS = 600

# 슬라이딩 윈도우 설정
WINDOW_V_MIN = 3.7
WINDOW_V_MAX = 3.9
N_WINDOWS = 10
WINDOW_WIDTH_V = 0.14  # 입력창의 전압 폭 → 약 140포인트

# 시각화 설정
CELLS_TO_PLOT = [1, 2, 3, 4, 5, 6, 7, 8]
CYCLE_INTERVAL = 3  # 몇 사이클마다 하나씩 그릴지


# =====================================
# 2) 데이터 생성 + 스케일링
# =====================================

def build_sliding_window_dataset() -> Dict[str, Any]:
    """
    1) oxford_ic_interpolated 안의 각 cell 파일 로드
    2) 슬라이딩 윈도우 기반 X(윈도우), Y(전체 600pt)를 생성
    3) train/valid/test split + StandardScaler 적용
    4) Conv1D/Transformer용 3D 입력으로 reshape

    반환:
        {
            'X_train_scaled', 'X_valid_scaled', 'X_test_scaled',
            'Y_train_scaled', 'Y_valid_scaled', 'Y_test_scaled',
            'all_df', 'curve_cols', 'v_ref',
            'window_start_indices', 'input_points_len',
            'scaler_x', 'scaler_y'
        }
    """
    # --- 2) 모든 Cell 데이터 로드 및 통합 ---
    all_df_list = []
    print("보간된 IC Curve 파일들을 로드합니다...")
    for cell_id in CELLS_TO_PROCESS:
        file_path = os.path.join(INPUT_DIR, f"oxford_cell_{cell_id}_ic_interpolated.csv")
        try:
            df = pd.read_csv(file_path)
            df["battery"] = f"Cell{cell_id}"
            all_df_list.append(df)
        except FileNotFoundError:
            print(f"[경고] 파일 없음: {file_path}")

    if not all_df_list:
        raise RuntimeError("[에러] 사용할 IC interpolated 파일이 없습니다.")

    all_df = pd.concat(all_df_list, ignore_index=True)
    print("모든 Cell 데이터 통합 완료.")

    # --- 3) 슬라이딩 윈도우 인덱스 계산 ---
    v_ref = np.linspace(VOLTAGE_MIN, VOLTAGE_MAX, N_POINTS)
    dv_per_point = (VOLTAGE_MAX - VOLTAGE_MIN) / (N_POINTS - 1)
    input_points_len = int(np.round(WINDOW_WIDTH_V / dv_per_point))
    print(
        f"\n입력 윈도우의 전압 폭 {WINDOW_WIDTH_V:.2f}V는 "
        f"{input_points_len}개의 포인트에 해당합니다."
    )

    sliding_start_idx = np.argmin(np.abs(v_ref - WINDOW_V_MIN))
    sliding_end_idx = np.argmin(np.abs(v_ref - WINDOW_V_MAX))

    window_start_indices = np.linspace(
        sliding_start_idx,
        sliding_end_idx - input_points_len,
        N_WINDOWS,
        dtype=int,
    )

    # --- 4) 슬라이딩 윈도우를 이용한 X, Y 데이터 생성 ---
    print("\n슬라이딩 윈도우로 학습 데이터를 생성합니다...")
    curve_cols = [f"v_{i}" for i in range(N_POINTS)]
    all_curves = all_df[curve_cols].values
    all_meta = all_df[["battery", "cycle"]].to_dict("records")

    new_X = []
    new_Y = []
    new_meta = []

    for i in tqdm(range(len(all_curves)), desc="Generating Samples"):
        full_curve = all_curves[i]
        meta_info = all_meta[i]

        for start_idx in window_start_indices:
            end_idx = start_idx + input_points_len

            # X: 윈도우로 잘라낸 입력 부분
            x_window = full_curve[start_idx:end_idx]
            new_X.append(x_window)

            # Y: 전체 600포인트 커브
            new_Y.append(full_curve)

            new_meta.append(meta_info)

    X = np.array(new_X, dtype=np.float32)  # (N_samples, input_len)
    Y = np.array(new_Y, dtype=np.float32)  # (N_samples, 600)
    meta_df = pd.DataFrame(new_meta)

    print(f"\n[데이터] 총 {len(all_df)}개 사이클로부터 {len(X)}개의 학습 샘플 생성 완료.")
    print(f"최종 X={X.shape}, Y={Y.shape}")  # Y shape = (N, 600)

    # --- 5) 랜덤 분할 및 스케일링 ---
    print("\n데이터를 분할하고 스케일링합니다...")
    (X_temp, X_test, Y_temp, Y_test, meta_temp, meta_test) = train_test_split(
        X, Y, meta_df, test_size=0.20, random_state=42, shuffle=True
    )
    (X_train, X_valid, Y_train, Y_valid, meta_train, meta_valid) = train_test_split(
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

    # Conv1D/Transformer 입력용 3D로 변환
    X_train_scaled = X_train_scaled.reshape(
        (X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    )
    X_valid_scaled = X_valid_scaled.reshape(
        (X_valid_scaled.shape[0], X_valid_scaled.shape[1], 1)
    )
    X_test_scaled = X_test_scaled.reshape(
        (X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    )

    # 요약 출력
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
        "all_df": all_df,
        "curve_cols": curve_cols,
        "v_ref": v_ref,
        "window_start_indices": window_start_indices,
        "input_points_len": input_points_len,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
    }


# =====================================
# 3) Transformer 모델 정의
# =====================================

# -------- [CLS] 토큰 레이어 --------
class CLSToken(L.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.d_model),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        b = tf.shape(x)[0]
        cls = tf.tile(self.cls, [b, 1, 1])
        return tf.concat([cls, x], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])


# -------- 포지셔널 임베딩 --------
class PositionalEmbedding(L.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.emb = L.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        T = tf.shape(x)[1]
        pos = tf.range(0, T)[tf.newaxis, :]
        return x + self.emb(pos)

    def compute_output_shape(self, input_shape):
        return input_shape


# -------- Transformer Encoder 블록 --------
class TransformerBlock(L.Layer):
    def __init__(self, d_model, num_heads, mlp_ratio=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = L.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.drop1 = L.Dropout(dropout)
        self.norm1 = L.LayerNormalization(epsilon=1e-5)

        self.ffn = tf.keras.Sequential(
            [
                L.Dense(d_model * mlp_ratio, activation="gelu"),
                L.Dropout(dropout),
                L.Dense(d_model),
            ]
        )
        self.drop2 = L.Dropout(dropout)
        self.norm2 = L.LayerNormalization(epsilon=1e-5)

    def call(self, x, training=False):
        attn_out = self.mha(x, x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x, training=training)
        ffn_out = self.drop2(ffn_out, training=training)
        return self.norm2(x + ffn_out)


def build_pure_transformer(
    input_len=140,
    output_len=600,
    d_model=128,
    num_layers=3,
    num_heads=4,
    mlp_ratio=4,
    dropout=0.1,
    dense_units=256,
    lr=1e-3,
) -> Model:
    """
    슬라이딩 윈도우 입력 → 전체 600pt 곡선 재구성용 Transformer 모델.
    """
    inp = L.Input(shape=(input_len, 1), name="ic_input")
    x = L.Dense(d_model)(inp)
    x = CLSToken(d_model)(x)
    x = PositionalEmbedding(max_len=input_len + 1, d_model=d_model)(x)

    for i in range(num_layers):
        x = TransformerBlock(d_model, num_heads, mlp_ratio, dropout, name=f"enc_{i}")(
            x
        )

    cls = L.Lambda(lambda t: t[:, 0, :], name="cls_take")(x)
    h = L.Dense(dense_units, activation="relu")(cls)
    h = L.Dropout(dropout)(h)
    out = L.Dense(output_len, activation="linear", name="ic_out")(h)

    model = Model(inp, out, name="PureTransformer_IC_Reconstruct")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


# =====================================
# 4) 학습 루프
# =====================================

def train_ic_transformer(
    data_dict: Dict[str, Any],
    epochs: int = 200,
    batch_size: int = 32,
) -> Model:
    """
    앞에서 만든 데이터(dict)를 받아 Transformer를 학습하고,
    학습된 모델을 반환.
    """
    X_train_scaled = data_dict["X_train_scaled"]
    X_valid_scaled = data_dict["X_valid_scaled"]
    Y_train_scaled = data_dict["Y_train_scaled"]
    Y_valid_scaled = data_dict["Y_valid_scaled"]

    input_len = X_train_scaled.shape[1]
    output_len = Y_train_scaled.shape[1]

    model = build_pure_transformer(
        input_len=input_len,
        output_len=output_len,
        d_model=128,
        num_layers=3,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        dense_units=256,
        lr=1e-3,
    )
    model.summary()

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=15, restore_best_weights=True
    )
    plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", mode="min", patience=6, factor=0.5, min_lr=1e-5
    )

    history = model.fit(
        X_train_scaled,
        Y_train_scaled,
        validation_data=(X_valid_scaled, Y_valid_scaled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early, plateau],
        verbose=1,
    )

    # 검증셋 역스케일 평가
    scaler_y = data_dict["scaler_y"]
    Y_pred_scaled = model.predict(X_valid_scaled)
    Y_pred = scaler_y.inverse_transform(Y_pred_scaled)
    Y_true = scaler_y.inverse_transform(Y_valid_scaled)

    print("Valid MAE(real):", np.mean(np.abs(Y_pred - Y_true)))
    print("Valid MSE(real):", np.mean((Y_pred - Y_true) ** 2))

    return model


# =====================================
# 5) 시각화 (윈도우/셀별 재구성 결과)
# =====================================

def plot_ic_reconstruction_per_window(
    model: Model,
    data_dict: Dict[str, Any],
    cells_to_plot: List[int] = CELLS_TO_PLOT,
    cycle_interval: int = CYCLE_INTERVAL,
):
    """
    각 입력 윈도우(#1~#N)마다, 선택한 Cell들의 일부 Cycle에서
    - Ground truth IC curve
    - 재구성된 예측 curve
    를 voltage 축(v_ref) 위에 함께 플로팅.
    """

    all_df = data_dict["all_df"].copy()
    curve_cols = data_dict["curve_cols"]
    v_ref = data_dict["v_ref"]
    window_start_indices = data_dict["window_start_indices"]
    input_points_len = data_dict["input_points_len"]
    scaler_x = data_dict["scaler_x"]
    scaler_y = data_dict["scaler_y"]

    print("Window별 -> Cell별 IC Curve 예측 결과 시각화를 시작합니다...")

    # 바깥 루프: 윈도우 index
    for i, start_idx in enumerate(window_start_indices):
        window_num = i + 1
        print("\n" + "=" * 49)
        print(f"--- Processing for Input Window #{window_num} ---")
        print("=" * 49)

        end_idx = start_idx + input_points_len

        # 각 Cell을 순회
        for cell_id in cells_to_plot:
            try:
                cell_df = all_df[all_df["battery"] == f"Cell{cell_id}"].copy()
                if cell_df.empty:
                    continue

                cell_df["cycle"] = (
                    pd.to_numeric(cell_df["cycle"], errors="coerce")
                    .dropna()
                    .astype(int)
                )
                unique_cycles = sorted(cell_df["cycle"].unique())
                if not unique_cycles:
                    continue

                cycles_to_visualize = [
                    c
                    for c in unique_cycles
                    if (c - unique_cycles[0]) % cycle_interval == 0
                ]
                if not cycles_to_visualize:
                    cycles_to_visualize = [unique_cycles[0]]

                plt.figure(figsize=(12, 8))
                norm = mcolors.Normalize(
                    vmin=min(cycles_to_visualize), vmax=max(cycles_to_visualize)
                )
                mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

                # 각 cycle에 대해 GT vs Pred plot
                for cycle_num in cycles_to_visualize:
                    original_full_curve = (
                        cell_df[cell_df["cycle"] == cycle_num][curve_cols]
                        .values.flatten()
                    )

                    # 입력 윈도우 추출
                    x_window = original_full_curve[start_idx:end_idx].reshape(1, -1)

                    # 스케일링 + 모델 입력 모양으로 변환
                    x_window_scaled = scaler_x.transform(x_window)
                    x_window_final = x_window_scaled[..., np.newaxis]

                    # 예측
                    y_pred_scaled = model.predict(x_window_final, verbose=0)
                    reconstructed_curve = scaler_y.inverse_transform(y_pred_scaled).flatten()

                    color = mapper.to_rgba(cycle_num)
                    plt.plot(v_ref, reconstructed_curve, color=color, alpha=0.8)
                    plt.plot(
                        v_ref,
                        original_full_curve,
                        color=color,
                        linestyle="--",
                        alpha=0.4,
                    )

                # 범례/하이라이트
                plt.plot([], [], "k-", label="Model Prediction")
                plt.plot([], [], "k--", label="Ground Truth", alpha=0.5)

                input_v_min_plot = v_ref[start_idx]
                input_v_max_plot = v_ref[end_idx - 1]
                plt.axvspan(
                    input_v_min_plot,
                    input_v_max_plot,
                    color="gray",
                    alpha=0.2,
                    label=f"Input Window #{window_num}",
                )

                cbar = plt.colorbar(mapper, ax=plt.gca())
                cbar.set_label("Cycle Number", fontsize=12)

                plt.title(
                    f"IC Curve Predictions from Window #{window_num} - Cell #{cell_id}",
                    fontsize=16,
                )
                plt.xlabel("Voltage (V)")
                plt.ylabel("IC Value")
                plt.grid(True, linestyle="--")
                plt.legend()
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"\n[오류] Cell {cell_id} 처리 중 문제가 발생했습니다: {e}")


# =====================================
# 6) main
# =====================================

if __name__ == "__main__":
    # 1. 데이터 생성 & 스케일링
    data = build_sliding_window_dataset()

    # 2. Transformer 학습
    model = train_ic_transformer(data, epochs=200, batch_size=32)

    # 3. 윈도우/셀별 재구성 결과 시각화
    plot_ic_reconstruction_per_window(model, data)
