"""
oxford_ic_soh_model.py

Oxford Incremental Capacity 멀티-윈도우 재구성 결과를 이용한
SOH(State of Health) 예측 CNN+LSTM 모델 학습 및 평가 스크립트.

입력:
    predictions_by_window/Window_k/Cell_i_pred_from_Wk.csv
        (각 파일: long format, columns=['battery','cycle','voltage','ic_true','ic_pred','SOH'])

출력:
    - 학습된 best_model_soh (메모리 상)
    - 콘솔에 Test MAE / RMSE
    - Window별 · Cell별 SOH True vs Pred 플롯
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM,
    Dropout, Flatten, Dense, LayerNormalization
)
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt

import keras_tuner as kt


# ============================================================
# 1) 데이터 로드 & 전처리
# ============================================================

def load_oxford_ic_soh_dataset(
    base_input_dir: str = "predictions_by_window",
    windows_to_process=range(1, 11),
    cells_to_process=range(1, 9),
    n_points: int = 600,
):
    """
    Window별 · Cell별 IC 재구성 결과(predictions_by_window)를 모두 모아
    (X: 600pt IC curve, y: SOH) 데이터셋을 만든다.

    - 각 파일은 long-format:
        ['battery', 'cycle', 'voltage', 'ic_pred', 'SOH', ...]
    - 여기서 SOH와 ic_pred만 pivot해서 wide-format (1 row = 1 curve)으로 변환
    """

    # --- 2) 모든 파일 경로 생성 ---
    file_map = {}
    print("80개의 예측 파일 경로를 생성합니다...")
    for window_num in windows_to_process:
        for cell_id in cells_to_process:
            unique_id = f"W{window_num}_Cell{cell_id}"
            file_path = os.path.join(
                base_input_dir,
                f"Window_{window_num}",
                f"Cell_{cell_id}_pred_from_W{window_num}.csv",
            )
            file_map[unique_id] = file_path

    wide_list = []
    print("모든 예측 결과 파일들을 로드하여 통합합니다...")

    for unique_id, path in file_map.items():
        try:
            df = pd.read_csv(path)

            # 필요한 컬럼만 사용
            df_long = df[["battery", "cycle", "voltage", "ic_pred", "SOH"]].copy()

            # long -> wide
            df_wide = df_long.pivot_table(
                index=["battery", "cycle", "SOH"],
                columns="voltage",
                values="ic_pred",
            ).reset_index()

            wide_list.append(df_wide)
        except FileNotFoundError:
            print(f"[경고] 파일을 찾을 수 없습니다: {path}. 건너뜁니다.")
        except Exception as e:
            print(f"[경고] {unique_id} 파일 처리 실패: {e}")

    if not wide_list:
        raise RuntimeError("사용할 데이터가 없습니다. 파일 경로/형식을 확인하세요.")

    all_df = pd.concat(wide_list, ignore_index=True)
    print("모든 데이터 통합 및 Wide-Format 변환 완료.")

    # --- 3) X / y 구성 ---
    # 컬럼 중에서 float/int 타입(voltage 값)만 골라서 정렬
    curve_cols = sorted(
        [col for col in all_df.columns if isinstance(col, (float, int))],
        key=float,
    )

    if len(curve_cols) != n_points:
        print(f"[경고] 컬럼 개수({len(curve_cols)})가 예상({n_points})과 다릅니다.")

    # NaN 제거
    all_df = all_df.dropna(subset=["SOH"] + curve_cols).reset_index(drop=True)

    X = all_df[curve_cols].values.astype(np.float32)  # (N, 600)
    y = all_df["SOH"].values.astype(np.float32)       # (N,)
    meta_df = all_df[["battery", "cycle"]].copy()

    print(f"\n[데이터] 최종 X={X.shape}, y={y.shape} 형태로 구성 완료.")

    # --- 4) 분할 (train/valid/test = 60/20/20) ---
    (X_temp, X_test, y_temp, y_test, meta_temp, meta_test) = train_test_split(
        X, y, meta_df, test_size=0.20, random_state=42, shuffle=True
    )
    (X_train, X_valid, y_train, y_valid, meta_train, meta_valid) = train_test_split(
        X_temp, y_temp, meta_temp, test_size=0.25, random_state=42, shuffle=True
    )

    print("\n--- 데이터 분할 후 Shape ---")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Valid: X={X_valid.shape}, y={y_valid.shape}")
    print(f"Test : X={X_test.shape},  y={y_test.shape}")

    # --- 5) 스케일링 ---
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_valid_scaled = scaler_x.transform(X_valid)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_valid_scaled = scaler_y.transform(y_valid.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    print("\n[스케일] StandardScaler로 Train/Valid/Test 변환 완료.")

    # --- 6) 모델 입력용 3D 변환 ---
    X_train_scaled = X_train_scaled[..., np.newaxis]
    X_valid_scaled = X_valid_scaled[..., np.newaxis]
    X_test_scaled = X_test_scaled[..., np.newaxis]

    print("\n--- 최종 데이터 Shape (모델 입력용) ---")
    print(f"Train: X={X_train_scaled.shape}, Y={y_train_scaled.shape}")
    print(f"Valid: X={X_valid_scaled.shape}, Y={y_valid_scaled.shape}")
    print(f"Test : X={X_test_scaled.shape},  Y={y_test_scaled.shape}")

    return (
        X_train_scaled,
        X_valid_scaled,
        X_test_scaled,
        y_train_scaled,
        y_valid_scaled,
        y_test_scaled,
        scaler_x,
        scaler_y,
        curve_cols,
        meta_train,
        meta_valid,
        meta_test,
    )


# ============================================================
# 2) CNN+LSTM SOH 모델 정의 & 튜닝 + 최종 학습
# ============================================================

def build_tunable_cnn_lstm(hp):
    seq_len = 600
    input_seq = Input(shape=(seq_len, 1))
    x = input_seq

    # CNN block 1
    filters1 = hp.Choice("filters1", [32, 64])
    kernel_size1 = hp.Choice("kernel_size1", [3, 5])
    x = Conv1D(filters1, kernel_size1, padding="same", activation="relu")(x)
    x = LayerNormalization()(x)
    if hp.Boolean("maxpool1", default=True):
        x = MaxPooling1D(pool_size=2)(x)

    # CNN block 2
    filters2 = hp.Choice("filters2", [64, 96])
    kernel_size2 = hp.Choice("kernel_size2", [3, 5, 7])
    x = Conv1D(filters2, kernel_size2, padding="same", activation="relu")(x)
    x = LayerNormalization()(x)
    if hp.Boolean("maxpool2", default=True):
        x = MaxPooling1D(pool_size=2)(x)

    # CNN block 3 (optional)
    if hp.Boolean("add_cnn3", default=False):
        filters3 = hp.Choice("filters3", [64, 96, 128])
        kernel_size3 = hp.Choice("kernel_size3", [3, 5, 7])
        x = Conv1D(filters3, kernel_size3, padding="same", activation="relu")(x)
        x = LayerNormalization()(x)
        if hp.Boolean("maxpool3", default=True):
            x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(hp.Float("dropout_cnn", 0.2, 0.5, step=0.1))(x)

    # LSTM block 1
    lstm_units1 = hp.Choice("lstm_units1", [16, 24, 32, 48, 64])
    return_seq_1 = hp.Boolean("return_seq_1", default=True)
    x = LSTM(lstm_units1, return_sequences=return_seq_1)(x)

    # LSTM block 2 (optional)
    if return_seq_1 and hp.Boolean("add_lstm2", default=False):
        lstm_units2 = hp.Choice("lstm_units2", [8, 12, 16, 24, 32])
        x = LSTM(lstm_units2, return_sequences=False)(x)
    elif not return_seq_1:
        x = Flatten()(x)
    else:
        x = Flatten()(x)

    # Dense layers
    dense_units1 = hp.Choice("dense_units1", [16, 32, 48, 64])
    x = Dense(dense_units1, activation="relu")(x)
    x = Dropout(hp.Float("dropout_dense1", 0.2, 0.3, step=0.1))(x)

    output = Dense(1)(x)

    model = Model(inputs=input_seq, outputs=output)
    lr = hp.Choice("learning_rate", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def tune_and_train_soh_model(
    X_train_scaled,
    y_train_scaled,
    X_valid_scaled,
    y_valid_scaled,
    X_test_scaled,
    y_test_scaled,
    scaler_y,
):
    """
    KerasTuner로 하이퍼파라미터 탐색 후,
    최적 모델을 더 길게 학습하고 Test set에서 최종 성능 평가.
    """

    early_stop_tune = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    )
    reduce_lr_tune = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=6,
        factor=0.5,
    )

    tuner = kt.RandomSearch(
        build_tunable_cnn_lstm,
        objective="val_loss",
        max_trials=20,
        executions_per_trial=1,
        directory="tune_cnn_lstm_final4",
        project_name="soh_pred",
    )

    print("\n========================================================")
    print("--- 하이퍼파라미터 튜닝 시작 ---")
    print("========================================================")

    for batch_size in [16, 32, 48]:
        tuner.search(
            X_train_scaled,
            y_train_scaled,
            epochs=100,
            batch_size=batch_size,
            validation_data=(X_valid_scaled, y_valid_scaled),
            callbacks=[early_stop_tune, reduce_lr_tune],
            verbose=1,
        )

    best_model_soh = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(1)[0]

    print("Best hyperparameters:")
    print(best_hp.values)

    print("\n========================================================")
    print("--- 최적 모델로 최종 학습을 시작합니다 ---")
    print("========================================================")

    early_stop_final = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True
    )
    reduce_lr_final = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=20, factor=0.5
    )

    history_final = best_model_soh.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=1000,
        batch_size=32,
        validation_data=(X_valid_scaled, y_valid_scaled),
        callbacks=[early_stop_final, reduce_lr_final],
        verbose=1,
    )

    print("\n[성공] 최종 학습이 완료되었습니다.")

    # --- 테스트셋 평가 ---
    print("\n========================================================")
    print("--- 최종 모델 성능 평가 (Test Set) ---")
    print("========================================================")

    loss, mae_scaled = best_model_soh.evaluate(X_test_scaled, y_test_scaled, verbose=0)

    y_pred_scaled = best_model_soh.predict(X_test_scaled, verbose=0)
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
    y_test_real = scaler_y.inverse_transform(y_test_scaled)

    final_mae_real = np.mean(np.abs(y_test_real - y_pred_real))
    final_rmse_real = np.sqrt(np.mean((y_test_real - y_pred_real) ** 2))

    print(f"Test Loss (scaled): {loss:.4f}")
    print(f"Test MAE (scaled): {mae_scaled:.4f}")
    print("---------------------------------------------")
    print(f"Test MAE (real value): {final_mae_real:.4f} %")
    print(f"Test RMSE (real value): {final_rmse_real:.4f} %")
    print("========================================================")

    return best_model_soh, best_hp, history_final


# ============================================================
# 3) Window·Cell별 SOH 예측 결과 시각화
# ============================================================

def plot_soh_predictions_by_window(
    prediction_base_dir: str,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    best_model_soh: tf.keras.Model,
    n_points: int = 600,
    windows_to_process=range(1, 11),
    cells_to_process=range(1, 9),
):
    """
    predictions_by_window/Window_k/Cell_i_pred_from_Wk.csv 에서
    각 Window별 · Cell별로 SOH True vs Pred 곡선을 그려준다.
    (1단계 IC 재구성 결과를 2단계 SOH 모델에 넣어 예측)
    """

    print("Window별 -> Cell별 SOH 예측 결과 시각화를 시작합니다...")

    for window_num in windows_to_process:
        window_dir = os.path.join(prediction_base_dir, f"Window_{window_num}")
        if not os.path.exists(window_dir):
            print(f"\n[경고] 폴더를 찾을 수 없습니다: '{window_dir}'. 건너뜁니다.")
            continue

        print("\n=============================================")
        print(f"--- Processing Predictions from Window #{window_num} ---")
        print("=============================================")

        for cell_id in cells_to_process:
            pred_file_path = os.path.join(
                window_dir, f"Cell_{cell_id}_pred_from_W{window_num}.csv"
            )

            try:
                df_long = pd.read_csv(pred_file_path)

                # long -> wide (IC curve)
                df_wide = df_long.pivot_table(
                    index=["cycle", "SOH"],
                    columns="voltage",
                    values="ic_pred",
                ).reset_index()

                curve_cols = sorted(
                    [col for col in df_wide.columns if isinstance(col, (float, int))],
                    key=float,
                )
                X_soh_input = df_wide[curve_cols].values.astype(np.float32)
                y_true = df_wide["SOH"].values
                cycles = df_wide["cycle"].values

                # 스케일링 + 3D 변환
                X_soh_scaled = scaler_x.transform(X_soh_input)
                X_soh_final = X_soh_scaled[..., np.newaxis]

                y_pred_scaled = best_model_soh.predict(X_soh_final, verbose=0)
                y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()

                # 플롯
                plt.figure(figsize=(10, 6))
                plt.plot(
                    cycles,
                    y_true,
                    "b-o",
                    markerfacecolor="white",
                    label="True SOH",
                )
                plt.plot(
                    cycles,
                    y_pred,
                    "r--x",
                    label="Predicted SOH",
                )

                plt.title(
                    f"SOH Prediction from Window #{window_num} - Cell #{cell_id}",
                    fontsize=16,
                )
                plt.xlabel("Cycle")
                plt.ylabel("SOH [%]")
                plt.grid(True, linestyle="--")
                plt.legend()
                plt.tight_layout()
                plt.show()

            except FileNotFoundError:
                # 파일 없으면 skip
                continue
            except Exception as e:
                print(f"\n[오류] 파일 처리 중 문제가 발생했습니다 ({pred_file_path}): {e}")

    print("\n\n✔ 모든 시각화 작업이 완료되었습니다.")


# ============================================================
# 4) main
# ============================================================

if __name__ == "__main__":
    # 1) 데이터 로드 & 전처리
    (
        X_train_scaled,
        X_valid_scaled,
        X_test_scaled,
        y_train_scaled,
        y_valid_scaled,
        y_test_scaled,
        scaler_x,
        scaler_y,
        curve_cols,
        meta_train,
        meta_valid,
        meta_test,
    ) = load_oxford_ic_soh_dataset(
        base_input_dir="predictions_by_window",
        windows_to_process=range(1, 11),
        cells_to_process=range(1, 9),
        n_points=600,
    )

    # 2) SOH 모델 튜닝 + 최종 학습 + 테스트 평가
    best_model_soh, best_hp, history_final = tune_and_train_soh_model(
        X_train_scaled,
        y_train_scaled,
        X_valid_scaled,
        y_valid_scaled,
        X_test_scaled,
        y_test_scaled,
        scaler_y,
    )

    # 3) Window/Cell별 SOH 예측 플롯
    plot_soh_predictions_by_window(
        prediction_base_dir="predictions_by_window",
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        best_model_soh=best_model_soh,
        n_points=600,
        windows_to_process=range(1, 11),
        cells_to_process=range(1, 9),
    )
