#NASA Battery Data set(#5,6,7), SOH Pred using charging curve
import re
import pandas as pd


path_in  = "/content/B0007_cc2_capacity_org.csv"
path_out = "/content/B0007_cc2_capacity_input05_35.csv"

df = pd.read_csv(path_in)

# 2) 앞 300포인트 / 나머지(cap_* 등) 분리
left  = df.iloc[:, :300].copy()
right = df.iloc[:, 300:].copy()

# 3) 20~40% 범위 인덱스 계산
total_len = 300
start_idx = int(total_len * 0.05)  # 60
end_idx   = int(total_len * 0.35)  # 120 (exclusive)

# 4) 컬럼 순서 재배치
x_block = left.iloc[:, start_idx:end_idx]  # v_60 ~ v_119
y_block = pd.concat([left.iloc[:, :start_idx], left.iloc[:, end_idx:]], axis=1)  # 나머지

# 5) x, y 라벨 재부여
x_block.columns = [f"x_{i}" for i in range(x_block.shape[1])]
y_block.columns = [f"v_{i}" for i in range(y_block.shape[1])]

# 6) 합치기: x 먼저, 그 다음 v들
left_new = pd.concat([x_block, y_block], axis=1)

# 7) cap_* 등 뒤쪽 붙이기
out = pd.concat([left_new, right], axis=1)

# 8) 저장
out.to_csv(path_out, index=False)
print(f"완료! 저장: {path_out}")

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "/content/B0007_cc2_capacity_input05_35.csv"  # 네가 저장한 파일 경로
row  = 5  # 보고 싶은 행

total_len = 300
x_start, x_end = int(total_len*0.05), int(total_len*0.35)  # 60, 120

def num_key(s):
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else -1

df = pd.read_csv(path)

# 1) 라벨 기반으로 x_, v_ 컬럼만 뽑기 (cap_*는 제외)
x_cols = sorted([c for c in df.columns if str(c).startswith("x_")], key=num_key)
v_cols = sorted([c for c in df.columns if str(c).startswith("v_")], key=num_key)

if len(x_cols) == (x_end - x_start) and len(v_cols) == (total_len - (x_end - x_start)):
    # (권장) x/v 라벨이 존재하는 파일: x 먼저, 나머지 v로 저장되어 있음
    x_input = df.loc[row, x_cols].to_numpy(dtype=float)          # 길이 60
    v_rest  = df.loc[row, v_cols].to_numpy(dtype=float)          # 길이 240

    # 원래 순서(0..59, 60..119, 120..299)로 복원
    v_recon = np.concatenate([v_rest[:x_start], x_input, v_rest[x_start:]])  # 길이 300
else:
    # (대안) 라벨이 없으면 앞 300컬럼을 곡선으로 가정해 직접 슬라이스
    curve   = df.iloc[row, :total_len].to_numpy(dtype=float)     # 길이 300
    v_recon = curve.copy()                                       # 길이 300

# 2) 플롯: 전체 곡선은 회색, 60~119 구간만 다른 스타일로 덮어그리기
x_axis = np.arange(total_len)

plt.figure(figsize=(9,4))
plt.plot(x_axis, v_recon, label="curve (all)", alpha=0.5)          # 기본
plt.plot(x_axis[x_start:x_end], v_recon[x_start:x_end], 'o-',      # x 구간 강조
         label=f"input ({x_start}–{x_end-1})")
plt.xlabel("Index (0–299)")
plt.ylabel("Value")
plt.title(f"Row {row} with input region highlighted")
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Add,Input, Conv1D, Flatten, Dense, Dropout,BatchNormalization, Activation,MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam,RMSprop,Nadam,AdamW
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
!pip install keras-tuner
#from keras_tuner import RandomSearch
# 1. 파일 경로 및 배터리ID 매핑
file_map = {
    'B0005': '/content/B0005_cc2_capacity_input05_35.csv',
    'B0006': '/content/B0006_cc2_capacity_input05_35.csv',
    'B0007': '/content/B0007_cc2_capacity_input05_35.csv'
}

# 2. 파일별 DataFrame 만들기
dfs = []
for bat_id, path in file_map.items():
    df = pd.read_csv(path)
    n = len(df)
    meta_df = pd.DataFrame({
        'battery': [bat_id]*n,
        'cycle': np.arange(n),
    })
    # 메타+X+Y 합치기
    full_df = pd.concat([meta_df, df], axis=1)
    dfs.append(full_df)

# 3. 전체 합치기
all_df = pd.concat(dfs, ignore_index=True)

# 4. 입력/출력 컬럼 추출
input_len = 90
output_len = 210

X = all_df[[f'x_{i}' for i in range(input_len)]].values[:, np.newaxis, :]   # (N, 1, 60)
Y = all_df[[f'v_{i}' for i in range(output_len)]].values                   # (N, 240)
meta = all_df[['battery', 'cycle']].values # (N, 4)

from sklearn.model_selection import train_test_split

# train+valid (80%), test (20%)
X_temp, X_test, Y_temp, Y_test, meta_temp, meta_test = train_test_split(
    X, Y, meta, test_size=0.2, random_state=42, shuffle=True
)
# train (70%), valid (10%)
X_train, X_valid, Y_train, Y_valid, meta_train, meta_valid = train_test_split(
    X_temp, Y_temp, meta_temp, test_size=0.25, random_state=42, shuffle=True
)
# (전체의 0.8 x 0.125 = 0.1 → valid 10%)

# transpose
X_train = X_train.transpose(0,2,1)
X_valid = X_valid.transpose(0,2,1)
X_test = X_test.transpose(0,2,1)

print(f"Train shape: X={X_train.shape}, Y={Y_train.shape}")
print(f"Valid shape: X={X_valid.shape}, Y={Y_valid.shape}")
print(f"Test shape:  X={X_test.shape}, Y={Y_test.shape}")

# StandardScaler는 2D에서만 작동 → reshape 필요
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_x.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_valid_scaled = scaler_x.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
X_test_scaled = scaler_x.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

Y_train_scaled = scaler_y.fit_transform(Y_train)
Y_valid_scaled = scaler_y.transform(Y_valid)
Y_test_scaled = scaler_y.transform(Y_test)
df_train_index = pd.DataFrame(meta_train, columns=['battery', 'cycle'])
df_valid_index = pd.DataFrame(meta_valid, columns=['battery', 'cycle'])
df_test_index  = pd.DataFrame(meta_test, columns=['battery', 'cycle'])

print(X_train_scaled.shape, X_valid_scaled.shape, X_test_scaled.shape)
print(df_train_index)

input_cols = [f'x_{i}' for i in range(input_len)]
output_cols = [f'v_{i}' for i in range(output_len)]

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Add,Input, Conv1D, Flatten, Dense, Dropout,BatchNormalization, Activation,MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam,RMSprop,Nadam,AdamW
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
!pip install keras-tuner
from keras_tuner import RandomSearch


def residual_block(x,filters,kernel_size,dilation_rate=1):
  shortcut=x
  x=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate)(x)
  x=BatchNormalization()(x)
  x=Activation('relu')(x)
  x=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate)(x)
  x=BatchNormalization()(x)

  if shortcut.shape[-1]!=filters:
    shortcut=Conv1D(filters,1,padding='same')(shortcut)
  x=Add()([x,shortcut])
  x=Activation('relu')(x)
  return x

from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Flatten, Dropout, Dense, Add, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization


def build_model(hp):
    inputs = Input(shape=(90, 1))
    x = Conv1D(
        filters=hp.Choice('filters1', [16, 24, 32, 48, 64, 96, 128]),
        kernel_size=hp.Choice('kernel_size1', [3, 5, 7, 9]),
        padding='same'
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    seq_len = x.shape[1] or 90 // 2

    n_blocks = hp.Choice('n_blocks', [2, 3, 4, 5, 6])
    for i in range(n_blocks):
        x = residual_block(
            x,
            filters=hp.Choice(f'filters_block_{i}', [16, 24, 32, 48, 64, 96, 128]),
            kernel_size=hp.Choice(f'kernel_size_block_{i}', [3, 5, 7, 9]),
            dilation_rate=hp.Choice(f'dilation_rate_block_{i}', [1, 2, 3, 4, 6, 8])
        )
        seq_len = x.shape[1] if x.shape[1] is not None else seq_len // 2
        if i != n_blocks-1 and seq_len >= 2:
            x = MaxPooling1D(pool_size=2)(x)
            seq_len //= 2

    x = Conv1D(
        filters=hp.Choice('filters2', [16, 24, 32, 48, 64, 96, 128]),
        kernel_size=hp.Choice('kernel_size2', [3, 5, 7, 9]),
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    seq_len = x.shape[1] if x.shape[1] is not None else seq_len
    if seq_len >= 2:
        x = MaxPooling1D(pool_size=2)(x)
        seq_len //= 2

    # === Transformer block 추가 ===
    num_heads = hp.Choice('num_heads', [2, 4, 6, 8])
    key_dim = hp.Choice('key_dim', [8, 12, 16, 24, 32])
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    x = Flatten()(x)
    x = Dropout(hp.Float('dropout', 0.1, 0.6, step=0.05))(x)
    outputs = Dense(210)(x)

    model = Model(inputs, outputs)
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'nadam', 'adamw'])
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr)
    elif optimizer_choice == 'nadam':
        optimizer = Nadam(learning_rate=lr)
    elif optimizer_choice == 'adamw':
        optimizer = AdamW(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

es=EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

tuner=RandomSearch(
    build_model,
    objective='val_mae',
    max_trials=20,
    executions_per_trial=1,
    directory='search_dir',
    project_name='1dcnn2_2'
)

tuner.search(X_train_scaled,Y_train_scaled,
             epochs=100,
             batch_size=32,
             validation_data=(X_valid_scaled,Y_valid_scaled),
             callbacks=[es]
)

best_model=tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

Y_pred_scaled = best_model.predict(X_valid_scaled)
Y_pred = scaler_y.inverse_transform(Y_pred_scaled)
Y_true = scaler_y.inverse_transform(Y_valid_scaled)

# 예: MAE/MSE 직접 산출
import numpy as np
print("실제 단위 MAE:", np.mean(np.abs(Y_pred - Y_true)))
print("실제 단위 MSE:", np.mean((Y_pred - Y_true) ** 2))

import matplotlib.pyplot as plt
n_samples = 10
np.random.seed(40)
idx = np.random.choice(len(Y_true), n_samples, replace=False)
total_len = 300
input_start = 15
input_end = 105

for i, sample_idx in enumerate(idx):
    full_true = np.empty(total_len)
    full_pred = np.empty(total_len)
    full_true[:] = np.nan
    full_pred[:] = np.nan

    x_input = X_valid[sample_idx][:, 0] if X_valid.ndim == 3 else X_valid[sample_idx]
    full_true[input_start:input_end] = x_input
    full_pred[input_start:input_end] = x_input

    y_true = Y_true[sample_idx]
    y_pred = Y_pred[sample_idx]

    full_true[:input_start] = y_true[:input_start]
    full_pred[:input_start] = y_pred[:input_start]

    full_true[input_end:] = y_true[input_start:]
    full_pred[input_end:] = y_pred[input_start:]

    plt.figure(figsize=(8, 3.5))
    plt.plot(full_true, label='True', color='black')
    plt.plot(full_pred, label='Pred', color='red', linestyle='dashed')
    plt.axvspan(input_start, input_end-1, color='skyblue', alpha=0.3, label='Input region')
    plt.legend()
    plt.title(f'Valid sample {sample_idx}')
    plt.xlabel('point')
    plt.ylabel('voltage')
    plt.show()


all_battery_names = sorted(all_df['battery'].unique())
def extract_features_from_cc_curve(voltage_curve):
    features = {}
    VOLTAGE_CUTOFF = 4.19
    try:
        voltage_curve = voltage_curve.astype(float)
        rising_part = np.where(voltage_curve >= VOLTAGE_CUTOFF)[0]
        if len(rising_part) > 0:
            cc_end_point = rising_part[0]
        else:
            cc_end_point = len(voltage_curve)
        features['cc_length'] = cc_end_point + 1
    except (IndexError, ValueError):
        features = {'cc_length': 0}
    return features

trained_model=best_model

# --- 4. 배터리별 정량 평가 ---
if trained_model:
    print("--- 4. 각 배터리별 전체 데이터에 대한 정량 평가 시작 ---")
    all_battery_names = sorted(all_df['battery'].unique())

    for bat_id in all_battery_names:
        print("\n" + "="*25 + f" 분석 결과: 배터리 {bat_id} " + "="*25)

        battery_df = all_df[all_df['battery'] == bat_id]
        if len(battery_df) == 0: continue

        X_battery = battery_df[input_cols].values
        Y_true_battery = battery_df[output_cols].values

        # --- [핵심 수정] 예측을 위한 데이터 변환 로직 수정 ---
        # 1. 각 배터리의 X 데이터를 스케일러가 학습한 모양(-1, 1)으로 변경
        X_battery_reshaped = X_battery.reshape(-1, 1)
        # 2. 스케일링
        X_battery_scaled_flat = scaler_x.transform(X_battery_reshaped)
        # 3. 모델 입력 모양 (N, 90, 1)으로 최종 변경
        input_for_model = X_battery_scaled_flat.reshape(len(battery_df), input_len, 1)

        # 예측 수행
        Y_pred_scaled_battery = trained_model.predict(input_for_model)
        Y_pred_battery = scaler_y.inverse_transform(Y_pred_scaled_battery)

        # 성능 지표 계산
        mae = mean_absolute_error(Y_true_battery, Y_pred_battery)
        mse = mean_squared_error(Y_true_battery, Y_pred_battery)
        r2 = r2_score(Y_true_battery, Y_pred_battery)

        print("\n--- [Level 1] 전체 곡선 유사도 평가 ---")
        print(f"평균 절대 오차 (MAE): {mae:.6f} V")
        print(f"평균 제곱 오차 (MSE): {mse:.6f} V²")
        print(f"결정 계수 (R² Score): {r2:.6f}")

        # Level 2 특징 비교
        print("\n--- [Level 2] cc_length 특징 비교 ---")
        true_features_list = [extract_features_from_cc_curve(curve) for curve in Y_true_battery]
        pred_features_list = [extract_features_from_cc_curve(curve) for curve in Y_pred_battery]
        df_true_features = pd.DataFrame(true_features_list)
        df_pred_features = pd.DataFrame(pred_features_list)
        cc_length_mae = mean_absolute_error(df_true_features['cc_length'], df_pred_features['cc_length'])
        print(f"CC 구간 길이 (cc_length)의 MAE: {cc_length_mae:.4f} (포인트 개수)")

    print("\n" + "="*70)
    print("각 배터리별 전체 데이터에 대한 분석이 완료되었습니다.")
else:
    print("모델이 로드되지 않아 평가를 진행할 수 없습니다.")

    curve_pred_model=loaded_model

import os
if curve_pred_model:
    print("--- 3. 전체 곡선 예측 및 파일 저장 시작 ---")
    output_dir = '/content/predicted_curves/'
    os.makedirs(output_dir, exist_ok=True)

    for bat_id in all_df['battery'].unique():
        print(f"  - 배터리 {bat_id} 곡선 예측 중...")
        battery_df = all_df[all_df['battery'] == bat_id]

        X_battery = battery_df[input_cols].values
        input_for_model = scaler_x.transform(X_battery.reshape(-1, 1)).reshape(len(battery_df), input_len, 1)

        predicted_output_parts = scaler_y.inverse_transform(curve_pred_model.predict(input_for_model))

        reconstructed_curves = []
        for i in range(len(battery_df)):
            true_input_part = X_battery[i]
            pred_part1 = predicted_output_parts[i, :60]
            pred_part2 = predicted_output_parts[i, 60:]
            reconstructed_curve = np.concatenate([pred_part1, true_input_part, pred_part2])
            reconstructed_curves.append(reconstructed_curve)

        df_pred = pd.DataFrame(reconstructed_curves, columns=[f'v_{i}' for i in range(300)])

        # --- [수정됨] SOH 열을 제외하고 cycle과 battery 정보만 추가 ---
        df_pred['cycle'] = battery_df['cycle'].values
        df_pred['battery'] = bat_id
        df_pred['cycle']=df_pred['cycle']+1

        save_path = os.path.join(output_dir, f'{bat_id}_predict.csv')
        df_pred.to_csv(save_path, index=False)
        print(f"  -> '{save_path}' 에 저장 완료.")

    print("\n--- 모든 예측 곡선 저장이 완료되었습니다. ---")
    print(f"저장된 파일들은 '{output_dir}' 폴더에서 확인하실 수 있습니다.")

else:
    print("모델이 로드되지 않아 예측을 진행할 수 없습니다.")


import pandas as pd

# 1. 곡선 데이터셋 불러오기 (기존 SOH 컬럼이 있을 수 있음)
curve_df = pd.read_csv('/content/predicted_curves/B0007_predict.csv')

# 2. 기존 SOH 컬럼이 있다면 제거
if 'SOH' in curve_df.columns:
    curve_df = curve_df.drop(columns=['SOH'])

# 3. SOH 데이터셋 준비 (cycle, SOH만)
soh_df = pd.read_csv('/content/B0007_with_capacity_SOH_revised.csv')
soh_map = soh_df[['Cycle', 'SOH']].rename(columns={'Cycle': 'cycle'})
soh_map_unique = soh_map.drop_duplicates(subset=['cycle'])

# 4. cycle 기준으로 SOH merge
result_df = curve_df.merge(soh_map_unique, how='left', on='cycle')
cols = list(result_df.columns)
cols.remove('SOH')
cycle_idx = cols.index('cycle')
new_order = cols[:cycle_idx+1] + ['SOH'] + cols[cycle_idx+1:]
result_df = result_df[new_order]
# 5. 저장18
result_df.to_csv('/content/predicted_curves/B0007_predict.csv', index=False)
print('완료! shape:', result_df.shape)
print(result_df[['cycle', 'SOH']].head(40))  # SOH가 cycle별로 다르게 잘 붙었는지 확인

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # train_test_split import

# --- 1. 데이터 로드 (이전과 동일) ---
battery_map = {
    'B0005': '/content/predicted_curves/B0005_predict.csv',
    'B0006': '/content/predicted_curves/B0006_predict.csv',
    'B0007': '/content/predicted_curves/B0007_predict.csv'
}
dfs = []
for bat, path in battery_map.items():
    df = pd.read_csv(path)
    # SOH 컬럼이 있는지 확인
    if 'SOH' not in df.columns:
        print(f"경고: {path}에 'SOH' 컬럼이 없습니다. 임의의 값을 생성합니다.")
        df['SOH'] = np.linspace(1.0, 0.7, len(df))
    df['battery'] = bat
    dfs.append(df)
all_df = pd.concat(dfs, ignore_index=True)

# --- 2. X, y 데이터 준비 (이전과 동일) ---
curve_cols = [col for col in all_df.columns if col.startswith('v_')]
X = all_df[curve_cols].values  # (샘플수, 300)
y = all_df['SOH'].values      # (샘플수,)

# --- 3. [수정됨] train_test_split을 사용한 데이터 분할 ---
# 먼저 전체 데이터를 학습+검증 세트(80%)와 테스트 세트(20%)로 분할합니다.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# 다음으로 학습+검증 세트를 학습 세트(기존의 75%, 전체의 60%)와 검증 세트(기존의 25%, 전체의 20%)로 분할합니다.
# test_size=0.25는 temp 데이터(80%)의 25%를 의미하므로, 전체 데이터의 20%가 됩니다.
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, shuffle=True
)

print("--- 데이터 분할 후 Shape ---")
print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Validation: X={X_valid.shape}, y={y_valid.shape}")
print(f"Test: X={X_test.shape}, y={y_test.shape}")


# --- 4. 스케일링 및 데이터 증강 (이전과 동일) ---
scaler_x = MinMaxScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_valid_scaled = scaler_x.transform(X_valid)
X_test_scaled = scaler_x.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_valid_scaled = scaler_y.transform(y_valid.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Reshape for Conv1D
X_train_scaled = X_train_scaled[..., np.newaxis]
X_valid_scaled = X_valid_scaled[..., np.newaxis]
X_test_scaled = X_test_scaled[..., np.newaxis]

# Add noise to training data
def add_gaussian_noise(data, noise_level=0.01):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=data.shape)
    noisy_data = data + noise
    return noisy_data

X_train_noisy = add_gaussian_noise(X_train_scaled, noise_level=0.01)

print("\n--- 최종 데이터 Shape (Conv1D 입력용) ---")
print(f"X_train_noisy (with channel): {X_train_noisy.shape}")
print(f"X_valid_scaled (with channel): {X_valid_scaled.shape}")
print(f"X_test_scaled (with channel): {X_test_scaled.shape}")


from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dropout, Flatten, Dense,LayerNormalization
from tensorflow.keras.models import Model
!pip install keras_tuner
import keras_tuner as kt
import tensorflow as tf

def build_tunable_cnn_lstm(hp):
    seq_len = 300
    input_seq = Input(shape=(seq_len, 1))
    x = input_seq

    # CNN block 1
    filters1 = hp.Choice('filters1', [16, 32])
    kernel_size1 = hp.Choice('kernel_size1', [3, 5])
    x = Conv1D(filters1, kernel_size1, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    if hp.Boolean('maxpool1', default=True):
        x = MaxPooling1D(pool_size=2)(x)

    # CNN block 2
    filters2 = hp.Choice('filters2', [ 32,64])
    kernel_size2 = hp.Choice('kernel_size2', [3, 5])
    x = Conv1D(filters2, kernel_size2, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    if hp.Boolean('maxpool2', default=True):
        x = MaxPooling1D(pool_size=2)(x)

    # 3rd CNN block (선택적)
    if hp.Boolean('add_cnn3', default=False):
        filters3 = hp.Choice('filters3', [64,96])
        kernel_size3 = hp.Choice('kernel_size3', [3, 5])
        x = Conv1D(filters3, kernel_size3, padding='same', activation='relu')(x)
        x = LayerNormalization()(x)
        if hp.Boolean('maxpool3', default=True):
            x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(hp.Float('dropout_cnn', 0.2,0.3,step=0.1))(x)

    # LSTM block 1
    lstm_units1 = hp.Choice('lstm_units1', [16, 24, 32, 48, 64])
    return_seq_1 = hp.Boolean('return_seq_1', default=True)
    x = LSTM(lstm_units1, return_sequences=return_seq_1)(x)

    # LSTM block 2 (선택적)
    if return_seq_1 and hp.Boolean('add_lstm2', default=False):
        lstm_units2 = hp.Choice('lstm_units2', [8, 12, 16, 24, 32])
        x = LSTM(lstm_units2, return_sequences=False)(x)
    elif not return_seq_1:
        x = Flatten()(x)
    else:
        x = Flatten()(x)

    # Dense layers
    dense_units1 = hp.Choice('dense_units1', [16, 32, 48, 64])
    x = Dense(dense_units1, activation='relu')(x)
    x = Dropout(hp.Float('dropout_dense1', 0.2, 0.3, step=0.1))(x)


    output = Dense(1)(x)

    model = Model(inputs=input_seq, outputs=output)
    lr = hp.Choice('learning_rate', [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    return model




early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=6, factor=0.5)

tuner = kt.RandomSearch(
    build_tunable_cnn_lstm,
    objective='val_loss',
    max_trials=30,
    executions_per_trial=1,
    directory='tune_cnn_lstm_final4',
    project_name='soh_pred6'
)

for batch_size in [16, 32, 48]:
    tuner.search(
        X_train_scaled, y_train_scaled,
        epochs=100,
        batch_size=batch_size,
        validation_data=(X_valid_scaled, y_valid_scaled),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

best_model_soh= tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(1)[0]

print("Best hyperparameters:")
print(best_hp.values)

import matplotlib.pyplot as plt
import numpy as np

# --- 최종 수정된 Plotting 코드 ---
model=best_model_soh
all_battery_names = ['B0005', 'B0006', 'B0007']

# 각 배터리에 대해 그래프 그리기
for bat_name in all_battery_names:
    # 1. 데이터 필터링 및 추출
    battery_df = all_df[all_df['battery'] == bat_name].copy()
    X_battery = battery_df[curve_cols].values
    y_true_battery = battery_df['SOH'].values

    # 2. 데이터 전처리 및 예측
    X_battery_scaled = scaler_x.transform(X_battery)
    X_battery_reshaped = X_battery_scaled[..., np.newaxis]
    y_pred_battery_scaled = model.predict(X_battery_reshaped)

    # 3. 원래 스케일로 복원
    y_pred_battery = scaler_y.inverse_transform(y_pred_battery_scaled)

    # 4. Plotting
    plt.figure(figsize=(12, 7))

    cycle_indices = np.arange(len(battery_df))

    # 실제 SOH와 예측 SOH 플롯
    plt.plot(cycle_indices, y_true_battery, 'b-o', label=f'True SOH', linewidth=2)
    plt.plot(cycle_indices, y_pred_battery.flatten(), 'r--x', label=f'Predicted SOH', linewidth=2)

    # 그래프 제목 및 레이블
    plt.title(f'SOH Degradation Curve: True vs. Predicted for Battery {bat_name}', fontsize=16)
    plt.xlabel('Cycle Index', fontsize=12)
    plt.ylabel('State of Health (SOH)', fontsize=12)

    # y축 범위 설정을 제거하여 자동으로 조절되도록 함

    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


   #NASA Battery Data set(#5,6,7), SOH Pred using ic curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # ← 가우시안 필터

# ====== 가우시안 유틸 ======
def _sigma_pts_from_volt(width_v: float, dV: float, min_sigma: float = 0.5) -> float:
    """
    전압 폭(Volt)을 샘플 단위 sigma로 변환.
    width_v: 원하는 스무딩 폭(표준편차에 대응, V)
    dV: 격자 간격(V/pt)
    """
    if dV <= 0 or width_v is None or width_v <= 0:
        return 1.0
    return max(width_v / dV, min_sigma)

# ====== IC 계산: 가우시안 스무딩 + 가우시안 미분 (Q(V) 방식) ======
def calculate_ic_gaussian(
    voltage, current, time,
    vmin=None, vmax=None, n_points=600,
    sigma_v=0.02,          # 주 스무딩 폭(Volt) ~ 20 mV (Q(V) 1차 미분에 적용)
    post_sigma_v=None      # (옵션) dQ/dV 후처리 약한 스무딩 폭(Volt), 예: 0.008~0.015
):
    """
    입력: voltage[V], current[A], time[s] (동일 길이)
    절차: 전류 적분 → Q(t) → (전압을 약하게 스무딩 후) Q(V) 보간 → 가우시안 1차 필터로 dQ/dV
    반환: (v_grid, ic_grid)
    """
    voltage = np.asarray(voltage, float)
    current = np.asarray(current, float)
    time    = np.asarray(time, float)

    n = len(voltage)
    if n < 5:
        # 샘플이 너무 적으면 간단 차분 근사
        dt = np.diff(time, prepend=time[0])
        dq = current * dt / 3600.0
        dv = np.diff(voltage, prepend=voltage[0])
        dv[np.abs(dv) < 1e-6] = 1e-6
        return voltage, dq / dv

    # (0) 전압을 인덱스 기준으로 아주 약하게 스무딩(단조성/노이즈 완화)
    v_smooth = gaussian_filter1d(voltage, sigma=2.0, mode='nearest')

    # (1) 전류 적분 → Q(t) [Ah]
    dt = np.diff(time, prepend=time[0])
    Q  = np.cumsum(current * dt) / 3600.0

    # (2) Q(V) 보간 준비: V 단조/유일화
    mask = np.isfinite(v_smooth) & np.isfinite(Q)
    v_m, Q_m = v_smooth[mask], Q[mask]
    order = np.argsort(v_m)
    v_sorted = v_m[order]; Q_sorted = Q_m[order]
    uniq, idx = np.unique(v_sorted, return_index=True)
    v_sorted = v_sorted[idx]; Q_sorted = Q_sorted[idx]

    # (3) 전압 격자
    if vmin is None: vmin = max(3.6, float(v_sorted.min()))
    if vmax is None: vmax = min(4.2,  float(v_sorted.max()))
    if vmax - vmin < 0.05:
        vmin, vmax = float(v_sorted.min()), float(v_sorted.max())

    v_grid = np.linspace(vmin, vmax, n_points)
    Q_grid = np.interp(v_grid, v_sorted, Q_sorted)

    # (4) 가우시안 1차 필터로 dQ/dV 계산 (order=1 은 샘플 인덱스 기준 미분)
    dV = float(np.diff(v_grid).mean())               # V/pt
    sigma_pts = _sigma_pts_from_volt(sigma_v, dV)    # Volt → pt
    dQdIdx = gaussian_filter1d(Q_grid, sigma=sigma_pts, order=1, mode='nearest')
    ic = dQdIdx / dV                                 # dQ/dV 로 단위 변환

    # (5) (옵션) dQ/dV 후처리 가우시안 스무딩
    if post_sigma_v is not None and post_sigma_v > 0:
        sigma_pts2 = _sigma_pts_from_volt(post_sigma_v, dV)
        ic = gaussian_filter1d(ic, sigma=sigma_pts2, order=0, mode='nearest')

    return v_grid, ic

# ====== 메인 ======
file_path = '/content/B0005.csv'


try:
    # 1) 데이터 로드
    df = pd.read_csv(file_path)
    print(f"원본 데이터 행 개수: {len(df)}, 사이클 개수: {df['Cycle'].nunique()}")

    # 2) 시작 전압 이상치 사이클 제거 (충전 시작 전압 > 4.0V)
    start_voltage_threshold = 3.9
    charge_start_points = df[df['Current_A'] > 1.0].groupby('Cycle').first()
    anomalous_cycles = charge_start_points[charge_start_points['Voltage_V'] > start_voltage_threshold].index.tolist()

    if anomalous_cycles:
        print(f"\n시작 전압이 {start_voltage_threshold}V 이상인 이상치 사이클 {len(anomalous_cycles)}개를 발견하여 제거합니다.")
        print(f"제거 대상 사이클: {anomalous_cycles}")  # 길면 앞부분만 출력해도 됨
        df = df[~df['Cycle'].isin(anomalous_cycles)]
        print(f"제거 후 데이터 행 개수: {len(df)}, 사이클 개수: {df['Cycle'].nunique()}\n")
    else:
        print("\n시작 전압이 비정상적인 사이클은 발견되지 않았습니다.\n")

    # 3) 분석할 사이클 선택 (예: 1부터 최대까지 10 간격)
    cycles_to_plot = np.arange(1, df['Cycle'].max() + 1, 1)

    # 4) 시각화
    plt.figure(figsize=(12, 8))
    plotted = 0

    for cycle_num in cycles_to_plot:
        cycle_df  = df[df['Cycle'] == cycle_num].copy()
        charge_df = cycle_df[cycle_df['Current_A'] > 1.0].sort_values(by='Time_s')  # 충전 구간

        if len(charge_df) > 10:
            # === 가우시안 스무딩+미분 기반 IC 계산 ===
            v_grid, ic_grid = calculate_ic_gaussian(
                charge_df['Voltage_V'].values,
                charge_df['Current_A'].values,
                charge_df['Time_s'].values,
                vmin=3.7, vmax=4.15, n_points=600,
                sigma_v=0.005,           # 주 스무딩 표준편차(Volt) ~ 20 mV
                post_sigma_v=None     # (옵션) 후처리 ~ 10 mV (덜 떨리게)
            )

            # 간단 필터링(극단값 제거)
            mask = np.isfinite(ic_grid) & (v_grid >= 3.5)
            if np.any(mask):
                plt.plot(
                    v_grid[mask], ic_grid[mask],
                    label=f'Cycle {cycle_num}' if cycle_num in [1, 51, 101, 151] else None,
                    alpha=0.5
                )
                plotted += 1

    if plotted == 0:
        print("경고: 플롯된 곡선이 없습니다. 전류 임계/스무딩 폭/범위를 조정하세요.")

    plt.xlabel('Voltage (V)')
    plt.ylabel('Incremental Capacity (dQ/dV)')
    plt.title('Filtered IC (dQ/dV) Curves for Battery B0005 — Gaussian smoothing + derivative (Q(V))')
    plt.grid(True)
    plt.ylim(0, 6)           # 필요시 주석 처리해 자동 스케일로 확인
    plt.xlim(3.5, 4.2)
    plt.legend()


except FileNotFoundError:
    print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"작업 중 오류가 발생했습니다: {e}")


#sigma_v(주 스무딩): 0.02~0.03 V에서 시작. 떨림 많으면 ↑, 피크가 뭉개지면 ↓.
#post_sigma_v(후처리): 0.008~0.015 V 정도로 아주 약하게. 너무 매끈하면 None.

# ===== 4) 시각화 + 300pt long-CSV 저장 =====
plt.figure(figsize=(12, 8))
plotted = 0

long_rows = []  # 여기다 (voltage, ic_smoothed, cycle) 로 누적

for cycle_num in cycles_to_plot:
    cycle_df  = df[df['Cycle'] == cycle_num].copy()
    charge_df = cycle_df[cycle_df['Current_A'] > 1.0].sort_values(by='Time_s')  # 충전 구간

    if len(charge_df) > 10:
        # --- 300pt 보간 + 가우시안 스무딩/미분 ---
        v_grid, ic_grid = calculate_ic_gaussian(
            charge_df['Voltage_V'].values,
            charge_df['Current_A'].values,
            charge_df['Time_s'].values,
            vmin=3.7, vmax=4.15, n_points=600,
            sigma_v=0.005,      # 스무딩 폭(Volt)
            post_sigma_v=None
        )

        # (A) 플롯은 보기 좋게만
        mask = np.isfinite(ic_grid) & (v_grid >= 3.5)
        if np.any(mask):
            plt.plot(v_grid[mask], ic_grid[mask],
                     label=f'Cycle {cycle_num}' if cycle_num in [1, 51, 101, 151] else None,
                     alpha=0.5)
            plotted += 1

        # (B) CSV 저장용: 필터 없이 300포인트 그대로 누적
        long_rows.append(pd.DataFrame({
            'voltage': v_grid,
            'ic_smoothed': ic_grid,
            'cycle': int(cycle_num)
        }))

if plotted == 0:
    print("경고: 플롯된 곡선이 없습니다. 전류 임계/스무딩 폭/범위를 조정하세요.")

plt.xlabel('Voltage (V)')
plt.ylabel('Incremental Capacity (dQ/dV)')
plt.title('Filtered IC (dQ/dV) — Gaussian smoothing + derivative (Q(V))')
plt.grid(True)
plt.ylim(0, 10)
plt.xlim(3.7, 4.15)
plt.legend()
plt.tight_layout()


# ===== 5) 하나의 CSV로 저장 (long format) =====
if long_rows:
    long_df = pd.concat(long_rows, ignore_index=True)
    # 정렬(선택): 사이클, 전압 오름차순
    long_df = long_df.sort_values(['cycle', 'voltage'])
    long_df.to_csv('B0005_ic_gaussian_600pt_long.csv', index=False)
    print("[저장] 'B0005_ic_gaussian_300pt_long.csv' (columns: voltage, ic_smoothed, cycle)")
else:
    print("저장할 데이터가 없습니다.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# --- 1) 설정 ---
file_map = {
    'B0005': '/content/B0005_ic_gaussian_600pt_long.csv',
    'B0006': '/content/B0006_ic_gaussian_600pt_long.csv',
    'B0007': '/content/B0007_ic_gaussian_600pt_long.csv',
}

VOLTAGE_MIN = 3.7
VOLTAGE_MAX = 4.15
N_POINTS    = 600

INPUT_V_MIN = 3.90
INPUT_V_MAX = 4.00

# --- 2) 데이터 로드(롱 → 사이클별 벡터화) ---
all_records = []  # {'battery':.., 'cycle':.., 'ic': np.array(N_POINTS)} 리스트
n_rows_total = 0
n_cycles_total = 0

v_ref = np.linspace(VOLTAGE_MIN, VOLTAGE_MAX, N_POINTS)

for bat_id, path in file_map.items():
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[경고] 파일 없음: {path}")
        continue

    # 기대 컬럼 확인
    required_cols = {'voltage', 'ic_smoothed', 'cycle'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"[에러] {path} 컬럼 부족. 필요: {required_cols}, 현재: {df.columns.tolist()}")

    n_rows_total += len(df)
    cycles = sorted(df['cycle'].unique())
    n_cycles_total += len(cycles)

    for cyc in cycles:
        sub = df[df['cycle'] == cyc].sort_values('voltage')
        if len(sub) < 5:
            continue  # 데이터 너무 적으면 스킵

        # 전압축이 조금 다를 수 있으므로 보간으로 통일
        ic_vec = np.interp(v_ref, sub['voltage'].values, sub['ic_smoothed'].values)
        all_records.append({
            'battery': bat_id,
            'cycle': int(cyc),
            'ic': ic_vec.astype(np.float32)
        })

if not all_records:
    raise RuntimeError("[에러] 사용할 사이클 데이터가 없습니다.")

print(f"[로드] 총 행수(전부): {n_rows_total:,}, 총 사이클수(전부): {n_cycles_total:,}")
print(f"[정리] 유효 사이클 샘플: {len(all_records):,}")

# --- 3) Matrix 구성 + X/Y 분할 ---
IC = np.vstack([r['ic'] for r in all_records])  # (N, N_POINTS)
meta_battery = np.array([r['battery'] for r in all_records])
meta_cycle   = np.array([r['cycle']   for r in all_records])

# 인덱스 선택
input_mask  = (v_ref >= INPUT_V_MIN) & (v_ref <= INPUT_V_MAX)
output_mask = ~input_mask
input_indices  = np.where(input_mask)[0]
output_indices = np.where(output_mask)[0]

if input_indices.size == 0:
    raise ValueError("[에러] 입력 구간에 해당하는 포인트가 없습니다. INPUT_V_MIN/MAX를 확인하세요.")

X = IC[:, input_indices]      # (N, n_in)
Y = IC[:, output_indices]     # (N, n_out)

print(f"[축] v_ref[{VOLTAGE_MIN}~{VOLTAGE_MAX}] N={N_POINTS}, "
      f"입력포인트={len(input_indices)}, 출력포인트={len(output_indices)}")
print(f"[데이터] X={X.shape}, Y={Y.shape}")

# --- 4) 랜덤 분할 (train/valid/test = 60/20/20) ---
X_temp, X_test, Y_temp, Y_test, bat_temp, bat_test, cyc_temp, cyc_test = train_test_split(
    X, Y, meta_battery, meta_cycle, test_size=0.20, random_state=42, shuffle=True
)
X_train, X_valid, Y_train, Y_valid, bat_train, bat_valid, cyc_train, cyc_valid = train_test_split(
    X_temp, Y_temp, bat_temp, cyc_temp, test_size=0.25, random_state=42, shuffle=True
)
# (전체 0.8 중 0.25 → 0.20) => 최종 60/20/20

# --- 5) 스케일링 (MinMax) ---
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_x.fit_transform(X_train)
Y_train_scaled = scaler_y.fit_transform(Y_train)

X_valid_scaled = scaler_x.transform(X_valid)
Y_valid_scaled = scaler_y.transform(Y_valid)

X_test_scaled  = scaler_x.transform(X_test)
Y_test_scaled  = scaler_y.transform(Y_test)

print("\n[스케일] MinMaxScaler로 Train/Valid/Test 변환 완료.")

# --- 6) 최종 요약 ---
print("\n--- 데이터 분할 및 전처리 결과 ---")
print(f"Train: X={X_train_scaled.shape}, Y={Y_train_scaled.shape}")
print(f"Valid: X={X_valid_scaled.shape}, Y={Y_valid_scaled.shape}")
print(f"Test : X={X_test_scaled.shape},  Y={Y_test_scaled.shape}")

# (선택) 메타정보 DataFrame
df_train_index = pd.DataFrame({'battery': bat_train, 'cycle': cyc_train})
df_valid_index = pd.DataFrame({'battery': bat_valid, 'cycle': cyc_valid})
df_test_index  = pd.DataFrame({'battery': bat_test,  'cycle': cyc_test})

# (선택) 확인용 출력
print("\n예시 Train 메타 5개:")
print(df_train_index.head())




import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
import numpy as np

# -------- [CLS] 토큰 레이어 --------
class CLSToken(L.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls_token", shape=(1, 1, self.d_model),
            initializer="zeros", trainable=True
        )
    def call(self, x):
        b = tf.shape(x)[0]
        cls = tf.tile(self.cls, [b, 1, 1])  # (B,1,d)
        return tf.concat([cls, x], axis=1)  # (B, 1+T, d)

# -------- 포지셔널 임베딩 --------
class PositionalEmbedding(L.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.emb = L.Embedding(input_dim=max_len, output_dim=d_model)
    def call(self, x):
        T = tf.shape(x)[1]
        pos = tf.range(0, T)[tf.newaxis, :]   # (1,T)
        return x + self.emb(pos)               # (1,T,d) broadcast

# -------- Transformer Encoder 블록 --------
class TransformerBlock(L.Layer):
    def __init__(self, d_model, num_heads, mlp_ratio=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = L.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.drop1 = L.Dropout(dropout)
        self.norm1 = L.LayerNormalization(epsilon=1e-5)

        self.ffn = tf.keras.Sequential([
            L.Dense(d_model * mlp_ratio, activation="gelu"),
            L.Dropout(dropout),
            L.Dense(d_model)
        ])
        self.drop2 = L.Dropout(dropout)
        self.norm2 = L.LayerNormalization(epsilon=1e-5)

    def call(self, x, training=False):
        attn_out = self.mha(x, x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x, training=training)
        ffn_out = self.drop2(ffn_out, training=training)
        return self.norm2(x + ffn_out)

# -------- 순수 Transformer 모델 --------
def build_pure_transformer(
    input_len=133, output_len=467,
    d_model=128, num_layers=3, num_heads=4, mlp_ratio=4,
    dropout=0.1, dense_units=256, lr=1e-3
):
    inp = L.Input(shape=(input_len, 1), name="ic_input")   # (B,67,1)

    # 토큰 임베딩
    x = L.Dense(d_model)(inp)                              # (B,67,d)

    # [CLS] + Positional
    x = CLSToken(d_model)(x)                               # (B,68,d)
    x = PositionalEmbedding(max_len=input_len + 1, d_model=d_model)(x)

    # Encoder Stack
    for i in range(num_layers):
        x = TransformerBlock(d_model, num_heads, mlp_ratio, dropout, name=f"enc_{i}")(x)

    # [CLS]만 사용
    cls = L.Lambda(lambda t: t[:, 0, :], name="cls_take")(x)  # (B,d)

    # 회귀 헤드
    h = L.Dense(dense_units, activation="relu")(cls)
    h = L.Dropout(dropout)(h)
    out = L.Dense(output_len, activation="linear", name="ic_out")(h)

    model = Model(inp, out, name="PureTransformer_IC")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="mse", metrics=["mae"])
    return model



model = build_pure_transformer(
    input_len=133, output_len=467,
    d_model=128, num_layers=3, num_heads=4, mlp_ratio=4,
    dropout=0.1, dense_units=256, lr=1e-3
)
model.summary()

early   = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                           patience=15, restore_best_weights=True)
plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min",
                                               patience=6, factor=0.5, min_lr=1e-5)

history = model.fit(
    X_train_scaled, Y_train_scaled,
    validation_data=(X_valid_scaled, Y_valid_scaled),
    epochs=300, batch_size=32,
    callbacks=[early, plateau],
    verbose=1
)

# 검증셋 역스케일 평가
Y_pred_scaled = model.predict(X_valid_scaled)
Y_pred = scaler_y.inverse_transform(Y_pred_scaled)
Y_true = scaler_y.inverse_transform(Y_valid_scaled)
print("Valid MAE(real):", np.mean(np.abs(Y_pred - Y_true)))
print("Valid MSE(real):", np.mean((Y_pred - Y_true) ** 2))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pred_curves_by_battery_every5(
    model,
    X_valid_scaled, Y_valid_scaled,
    scaler_x, scaler_y,
    df_valid_index,
    v_min=3.7, v_max=4.15, n_points=600,
    input_v_min=3.90, input_v_max=4.00,
    show_true=True
):
    # ---- 0) 축/구간 인덱스 ----
    v_ref = np.linspace(v_min, v_max, n_points)
    in_mask  = (v_ref >= input_v_min) & (v_ref <= input_v_max)
    out_mask = ~in_mask
    in_idx   = np.where(in_mask)[0]
    out_idx  = np.where(out_mask)[0]

    # ---- 1) 예측 및 역스케일 ----
    # 모델 입력 모양 맞추기 (N, n_in, 1) or (N, n_in)
    if X_valid_scaled.ndim == 3:
        Xv_for_model = X_valid_scaled
        Xv_2d_for_inv = X_valid_scaled.reshape(X_valid_scaled.shape[0], -1)  # scaler_x는 2D 필요
        n_in = X_valid_scaled.shape[1]
    else:
        Xv_for_model = X_valid_scaled
        Xv_2d_for_inv = X_valid_scaled
        n_in = X_valid_scaled.shape[1]


    assert len(in_idx) == n_in, f"입력 구간 길이({len(in_idx)})와 모델 입력 길이({n_in})가 다릅니다. input_v_min/max를 확인하세요."

    # 예측
    Y_pred_scaled = model.predict(Xv_for_model, verbose=0)           # (N, n_out)
    # 역스케일
    X_in_true  = scaler_x.inverse_transform(Xv_2d_for_inv)           # (N, n_in)
    Y_true     = scaler_y.inverse_transform(Y_valid_scaled)          # (N, n_out)
    Y_pred     = scaler_y.inverse_transform(Y_pred_scaled)           # (N, n_out)

    # ---- 2) 배터리별 5 간격 사이클 선택 및 플롯 ----
    for bat in sorted(df_valid_index['battery'].unique()):
        dfb = df_valid_index[df_valid_index['battery'] == bat].copy()
        if dfb.empty:
            print(f"[{bat}] 검증셋에 해당 배터리 샘플이 없습니다.")
            continue

        cycles = sorted(dfb['cycle'].astype(int).unique())
        start_cyc = cycles[0]
        sel_cycles = [c for c in cycles if (c - start_cyc) % 3 == 0]
        if len(sel_cycles) == 0:
            print(f"[{bat}] 5 간격으로 선택된 사이클이 없습니다.")
            continue

        # 그림
        plt.figure(figsize=(10, 5))
        plt.title(f"{bat} — Pred vs True (cycles every 5 from {start_cyc})")
        plt.xlabel("Voltage (V)")
        plt.ylabel("IC (dQ/dV)")
        plt.grid(True, alpha=0.3)

        # 색상 팔레트
        colors = plt.cm.tab20(np.linspace(0, 1, len(sel_cycles)))

        plotted = 0
        for k, cyc in enumerate(sel_cycles):
            # 해당 사이클의 첫 번째 인덱스 선택 (동일 cycle이 여러 개면 첫 샘플 사용)
            idx_candidates = dfb.index[dfb['cycle'].astype(int) == int(cyc)].tolist()
            if len(idx_candidates) == 0:
                continue
            idx = idx_candidates[0]

            # 전체 300pt 재구성
            full_true = np.empty(n_points, dtype=float)
            full_pred = np.empty(n_points, dtype=float)

            # 입력 구간: 항상 실제 입력(정답) 사용
            full_true[in_idx] = X_in_true[idx]
            full_pred[in_idx] = X_in_true[idx]

            # 출력 구간: 정답/예측 각자 채우기
            full_true[out_idx] = Y_true[idx]
            full_pred[out_idx] = Y_pred[idx]

            # 플롯
            plt.plot(v_ref, full_pred, color=colors[k], lw=1.8, label=f"pred cycle {int(cyc)}")
            if show_true:
                plt.plot(v_ref, full_true, color=colors[k], lw=1.0, ls='--', alpha=0.7)

            plotted += 1

        # 입력 구간 음영
        plt.axvspan(input_v_min, input_v_max, color='gray', alpha=0.15, label='Input window')

        if plotted > 0 and plotted <= 14:
            plt.legend(ncol=2, fontsize=9)
        plt.xlim(v_min, v_max)
        plt.tight_layout()
        plt.show()

plot_pred_curves_by_battery_every5(
    model=model,
    X_valid_scaled=X_test_scaled,
    Y_valid_scaled=Y_test_scaled,
    scaler_x=scaler_x,
    scaler_y=scaler_y,
    df_valid_index=df_valid_index,
    v_min=3.7, v_max=4.15, n_points=600,
    input_v_min=3.90, input_v_max=4.00,
    show_true=True
)


import os
import numpy as np
import pandas as pd

def save_ic_predictions_per_battery_safe(
    file_map,
    model, scaler_x, scaler_y,
    v_min=3.7, v_max=4.15, n_points=300,
    input_v_min=3.90, input_v_max=4.00,
    out_dir="ic_pred_curves",
    per_battery=True,
    batch_size=256,
    verbose=True,
):
    os.makedirs(out_dir, exist_ok=True)

    # 0) 공통 전압축과 구간 인덱스
    v_ref = np.linspace(v_min, v_max, n_points)
    in_mask  = (v_ref >= input_v_min) & (v_ref <= input_v_max)
    out_mask = ~in_mask
    in_idx   = np.where(in_mask)[0]
    out_idx  = np.where(out_mask)[0]

    # 1) 모델 입출력 형태를 "가능하면" 추론 (안되면 런타임에 예측 결과로 확인)
    def _infer_io_shapes(m):
        exp_3d = None   # True: (None, n_in, 1) 기대, False: (None, n_in)
        squeeze_out = None  # True: (None, n_out, 1) 출력
        n_in_model = None
        n_out_model = None
        try:
            in_tensor  = getattr(m, "inputs", [None])[0]
            out_tensor = getattr(m, "outputs", [None])[0]
            if in_tensor is not None:
                in_rank = in_tensor.shape.rank
                if in_rank == 3:
                    exp_3d = True
                    n_in_model = int(in_tensor.shape[1])
                elif in_rank == 2:
                    exp_3d = False
                    n_in_model = int(in_tensor.shape[-1])
            if out_tensor is not None:
                out_rank = out_tensor.shape.rank
                if out_rank == 3:
                    squeeze_out = (int(out_tensor.shape[-1]) == 1)
                    n_out_model = int(out_tensor.shape[1])
                elif out_rank == 2:
                    squeeze_out = False
                    n_out_model = int(out_tensor.shape[-1])
        except Exception:
            pass
        return exp_3d, squeeze_out, n_in_model, n_out_model

    exp_3d_hint, squeeze_hint, n_in_hint, n_out_hint = _infer_io_shapes(model)

    # 2) 배터리 루프
    all_rows = []
    for bat, path in file_map.items():
        if verbose: print(f"\n[{bat}] loading {path} ...")
        df = pd.read_csv(path)
        for col in ["voltage", "ic_smoothed", "cycle"]:
            if col not in df.columns:
                raise ValueError(f"{path} 에 '{col}' 컬럼이 없습니다.")

        cycles = sorted(df["cycle"].unique())
        if not cycles:
            if verbose: print(f"[{bat}] 사이클이 없어 스킵")
            continue

        # 사이클별 입력 배치 만들기
        X_list, true_full_list, meta_cycles = [], [], []
        for cyc in cycles:
            sub = df[df["cycle"] == cyc].sort_values("voltage")
            if len(sub) < 5:
                continue
            ic_full = np.interp(v_ref, sub["voltage"].values,
                                sub["ic_smoothed"].values).astype(np.float32)
            X_list.append(ic_full[in_idx])       # 입력 구간 벡터
            true_full_list.append(ic_full)       # 전체 300pt (재구성용)
            meta_cycles.append(int(cyc))

        if not X_list:
            if verbose: print(f"[{bat}] 유효 사이클 없음 (스킵)")
            continue

        X_batch = np.vstack(X_list)                        # (M, n_in)
        X_batch_scaled = scaler_x.transform(X_batch)       # (M, n_in)

        # 3) 모델 입력 차원 맞춰 예측 (3D→실패시 2D, 또는 힌트대로)
        def _predict_with_fallback(X_scaled):
            # 후보들: [(expects_3d_bool, ndarray_to_feed), ...]
            candidates = []
            if exp_3d_hint is True:
                candidates = [(True, X_scaled[..., np.newaxis]), (False, X_scaled)]
            elif exp_3d_hint is False:
                candidates = [(False, X_scaled), (True, X_scaled[..., np.newaxis])]
            else:
                # 힌트 없음 → 3D 먼저 시도 후 2D
                candidates = [(True, X_scaled[..., np.newaxis]), (False, X_scaled)]

            last_err = None
            for exp3d, X_in in candidates:
                try:
                    Y_scaled = model.predict(X_in, batch_size=batch_size, verbose=0)
                    Y_scaled = np.asarray(Y_scaled)
                    return exp3d, Y_scaled
                except Exception as e:
                    last_err = e
                    continue
            raise RuntimeError(f"model.predict 실패: {type(last_err).__name__}: {last_err}")

        exp_3d_used, Y_pred_scaled = _predict_with_fallback(X_batch_scaled)

        # 출력 모양 정리 (None, n_out) 로
        if Y_pred_scaled.ndim == 3 and Y_pred_scaled.shape[-1] == 1:
            Y_pred_scaled = Y_pred_scaled[..., 0]  # squeeze
        if Y_pred_scaled.ndim != 2:
            raise ValueError(f"예상치 못한 예측 랭크: {Y_pred_scaled.shape}")

        # 4) 출력 길이 검증: out_idx 길이와 동일해야 함
        if Y_pred_scaled.shape[1] != len(out_idx):
            raise ValueError(
                f"출력 길이 불일치: 예측 n_out={Y_pred_scaled.shape[1]} vs "
                f"out_idx={len(out_idx)}. (n_points, input_v_min/max 확인)"
            )

        # 역스케일
        Y_pred = scaler_y.inverse_transform(Y_pred_scaled)  # (M, n_out)

        # 5) full curve 재구성 & 롱포맷 누적
        bat_rows = []
        for i, cyc in enumerate(meta_cycles):
            ic_true_full = true_full_list[i]
            ic_pred_full = np.empty_like(ic_true_full, dtype=np.float32)
            ic_pred_full[in_idx]  = ic_true_full[in_idx]   # 입력구간: 실제
            ic_pred_full[out_idx] = Y_pred[i]              # 출력구간: 예측

            tmp = pd.DataFrame({
                "battery": bat,
                "cycle": int(cyc),
                "voltage": v_ref,
                "ic_true": ic_true_full,
                "ic_pred": ic_pred_full
            })
            if per_battery:
                bat_rows.append(tmp)
            else:
                all_rows.append(tmp)

        if per_battery:
            out_path = os.path.join(out_dir, f"{bat}_ic_pred_{n_points}pt.csv")
            pd.concat(bat_rows, ignore_index=True).to_csv(out_path, index=False)
            if verbose: print(f"[{bat}] saved → {out_path}")

    if not per_battery and all_rows:
        out_path = os.path.join(out_dir, f"ALL_ic_pred_{n_points}pt.csv")
        pd.concat(all_rows, ignore_index=True).to_csv(out_path, index=False)
        if verbose: print(f"[ALL] saved → {out_path}")


file_map = {
    'B0005': '/content/B0005_ic_gaussian_600pt_long.csv',
    'B0006': '/content/B0006_ic_gaussian_600pt_long.csv',
    'B0007': '/content/B0007_ic_gaussian_600pt_long.csv',
}

save_ic_predictions_per_battery_safe(
    file_map=file_map,
    model=model,                 # or best_model
    scaler_x=scaler_x, scaler_y=scaler_y,
    v_min=3.7, v_max=4.15, n_points=600,
    input_v_min=3.90, input_v_max=4.00,
    out_dir="ic_preds",
    per_battery=True,
    batch_size=32,
    verbose=True
)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
import glob
import seaborn as sns


# 1. CSV 불러오기
df = pd.read_csv('/content/B0007.csv')
print("✅ 데이터 로드 완료")

# 2. Cycle별 Capacity 계산 함수
def compute_capacity_scalar(group):
    t = group['Time_s'].values
    i = group['Current_A'].values
    dt = np.diff(t, prepend=t[0])  # Δt [s]
    cap = np.sum(i * dt) / 3600    # Ah (스칼라)
    group['capacity'] = cap
    return group

# 3. 그룹별 처리
df = df.groupby('Cycle', group_keys=False).apply(compute_capacity_scalar)

# 4. 결과 확인
print(df.head(10))

# 5. 저장
output_path = '/content/B0007_with_cycle_capacity.csv'
df.to_csv(output_path, index=False)

remove_cycles = [1, 33, 170]
df2=pd.read_csv('/content/B0007_with_cycle_capacity.csv')
# Cycle이 1, 33, 170이 아닌 행만 남김
df_filtered = df2[~df2['Cycle'].isin(remove_cycles)]

# 결과 확인
print(f"원래 데이터 수: {len(df2)}")
print(f"제거 후 데이터 수: {len(df_filtered)}")

# 저장 (선택)
df_filtered.to_csv('/content/B0007_removed_cycles.csv', index=False)
print("✅ Cycle 1, 33, 170 제거 완료 → B0005_removed_cycles.csv 저장됨")

df2=pd.read_csv('/content/B0007_removed_cycles.csv')

for n in np.unique(df2['Cycle']):
    cap = df2[df2['Cycle'] == n]['capacity'].iloc[0]
    print(f'Cycle {n}: Capacity = {cap:.4f} Ah')


import pandas as pd
import numpy as np

# 1. 데이터 로드 (이미 df_filtered에 있다면 생략 가능)
df = pd.read_csv('/content/B0007_removed_cycles.csv')

# 2. 각 Cycle의 고유한 capacity 값 추출
cycle_caps = df.groupby('Cycle')['capacity'].first()  # Series: index=Cycle, value=capacity

# 3. 최대 capacity 찾기 (MAX SOH 기준)
max_cap = cycle_caps.max()

# 4. SOH 계산 (%)
soh_dict = (cycle_caps / max_cap * 100).to_dict()  # {cycle_num: SOH%}

# 5. df에 SOH 열 추가
df['SOH'] = df['Cycle'].map(soh_dict)

# 6. 확인
print(df[['Cycle', 'capacity', 'SOH']].drop_duplicates().head(168))

# 7. 저장
df.to_csv('/content/B0007_with_SOH.csv', index=False)
print("✅ SOH 계산 및 추가 완료 → B0005_with_SOH.csv 저장됨")
print(df['SOH'].unique())

import pandas as pd

# --- ★★★ 사용자 설정 부분 ★★★ ---

# 1. 파일 경로를 지정하세요.
# SOH 열을 추가하고 싶은 대상 파일
FILE_A_PATH = '/content/ic_preds/B0007_ic_pred_600pt.csv'
# SOH 정보가 들어있는 원본 파일
FILE_B_PATH = '/content/B0007_with_SOH.csv'
# 결과가 저장될 새로운 파일 이름
OUTPUT_FILE_PATH = '/content/ic_preds/B0007_ic_pred_600pt_SOH.csv'

# 2. 각 파일의 컬럼(열) 이름을 확인하고 필요시 수정하세요.
# (대소문자를 구분합니다)
CYCLE_COL_A = 'cycle'  # A 파일의 사이클 컬럼 이름
CYCLE_COL_B = 'Cycle'  # B 파일의 사이클 컬럼 이름
SOH_COL_B = 'SOH'      # B 파일의 SOH 컬럼 이름

# --------------------------------

try:
    # 1. 두 개의 CSV 파일 불러오기
    print(f"대상 파일 '{FILE_A_PATH}'을(를) 불러옵니다...")
    df_A = pd.read_csv(FILE_A_PATH)

    print(f"SOH 원본 파일 '{FILE_B_PATH}'을(를) 불러옵니다...")
    df_B = pd.read_csv(FILE_B_PATH)

    # 2. B 파일에서 SOH 정보 추출 (중복 제거)
    # B 파일의 각 사이클별 SOH 값을 고유하게 만듭니다. (성능 향상 및 오류 방지)
    soh_lookup = df_B[[CYCLE_COL_B, SOH_COL_B]].drop_duplicates().reset_index(drop=True)

    # 3. A 파일에 SOH 정보 병합(merge)
    print(f"'{CYCLE_COL_A}' 컬럼을 기준으로 SOH 값을 병합합니다...")

    # 컬럼 이름이 다를 경우를 대비하여 B의 사이클 컬럼 이름을 A에 맞춤
    soh_lookup.rename(columns={CYCLE_COL_B: CYCLE_COL_A}, inplace=True)

    # pd.merge를 사용하여 A파일 기준으로 B파일의 SOH 정보를 붙임
    # how='left'는 A파일의 모든 행을 유지하라는 의미입니다.
    df_A_with_soh = pd.merge(df_A, soh_lookup, on=CYCLE_COL_A, how='left')

    # 4. 결과 확인 및 저장
    # 병합 후 SOH가 비어있는(NaN) 행이 있는지 확인
    nan_count = df_A_with_soh['SOH'].isnull().sum()
    if nan_count > 0:
        print(f"\n[경고] A 파일의 사이클 중 {nan_count}개 행에 해당하는 SOH 값을 B 파일에서 찾을 수 없습니다.")
    else:
        print("\n모든 행에 SOH 값이 성공적으로 복사되었습니다.")

    # 새로운 CSV 파일로 저장
    df_A_with_soh.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\n결과가 '{OUTPUT_FILE_PATH}' 파일로 저장되었습니다.")

    # 결과 미리보기
    print("\n--- 결과 데이터 샘플 (상위 5줄) ---")
    print(df_A_with_soh.head())


except FileNotFoundError as e:
    print(f"[오류] 파일을 찾을 수 없습니다: {e.filename}")
    print("파일 경로와 이름이 올바른지 확인해주세요.")
except KeyError as e:
    print(f"[오류] 지정된 컬럼을 찾을 수 없습니다: {e}")
    print("사용자 설정 부분의 컬럼 이름이 실제 파일과 일치하는지 확인해주세요.")
except Exception as e:
    print(f"작업 중 오류가 발생했습니다: {e}")


    import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler  # 필요시 StandardScaler로 교체 가능

# ---- 설정 ----
# 파일이 하나만 있어도 아래 dict 형식으로 넣어줘. (여러 배터리 확장 가능)
battery_map = {
    'B0005': '/content/ic_preds/B0005_ic_pred_600pt_SOH.csv',
    'B0006': '/content/ic_preds/B0006_ic_pred_600pt_SOH.csv',
    'B0007': '/content/ic_preds/B0007_ic_pred_600pt_SOH.csv',
    #'B0018': '/content/ic_preds/B0018_ic_pred_600pt_with_SOH.csv'
    #,
}

# long -> 공통 격자 보간이 필요할 때 사용
V_MIN, V_MAX = 3.7, 4.15
DEFAULT_N_POINTS = 600  # 파일에 v_0..v_<N-1> 있으면 그걸 따름

def _detect_wide_columns(cols):
    """wide 형식이면 v_0, v_1 ... 형태 컬럼 리스트 반환, 아니면 빈 리스트"""
    vcols = [c for c in cols if c.startswith('v_')]
    vcols_sorted = sorted(vcols, key=lambda x: int(x.split('_')[1])) if vcols else []
    return vcols_sorted

def _load_one_file(battery_id, path, v_min=V_MIN, v_max=V_MAX, fallback_n=DEFAULT_N_POINTS):
    """
    파일 1개 로드 → (DataFrame: 한 행=한 cycle, 컬럼: v_0..v_(N-1), cycle, SOH, battery)
    wide/long 모두 처리
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # 공통 컬럼명 추출
    def pick(cols, cands):
        for c in cands:
            if c in cols: return c
        return None

    cyc_col  = pick(df.columns, ['cycle', 'Cycle'])
    soh_col  = pick(df.columns, ['SOH','soh','SOH(%)','SOH_percent',
                                 'capacity_retention','State_of_Health'])
    # wide 형식인지 확인
    vcols = _detect_wide_columns(df.columns)

    if vcols:
        # ---- WIDE 형식 ----
        wide = df.copy()
        if cyc_col is None:
            raise KeyError(f"[{battery_id}] cycle 컬럼을 찾을 수 없습니다. (wide 형식)")
        if soh_col is None:
            raise KeyError(f"[{battery_id}] SOH 컬럼이 없습니다. (wide 형식)")

        # 필요한 컬럼만
        keep = [cyc_col, soh_col] + vcols
        wide = wide[keep].copy()
        # cycle/SOH 정리
        wide.rename(columns={cyc_col:'cycle', soh_col:'SOH'}, inplace=True)
        # SOH 없는 cycle 제거
        wide = wide.dropna(subset=['SOH'])
        # 배터리 태그
        wide['battery'] = battery_id

        # 숫자형 강제
        wide['cycle'] = pd.to_numeric(wide['cycle'], errors='coerce').astype('Int64')
        wide['SOH']   = pd.to_numeric(wide['SOH'], errors='coerce')
        wide = wide.dropna(subset=['cycle','SOH'])

        # 정렬
        wide = wide.sort_values(['cycle']).reset_index(drop=True)
        return wide

    else:
        # ---- LONG 형식 (cycle, voltage, ic_pred|ic_smoothed, SOH) ----
        ic_col = pick(df.columns, ['ic_pred','ic_smoothed','ic','IC','dQdV'])
        volt_col = pick(df.columns, ['voltage','Voltage','V'])

        if cyc_col is None or soh_col is None or ic_col is None or volt_col is None:
            raise KeyError(
                f"[{battery_id}] long 형식으로 추정되지만 필요한 컬럼이 없습니다.\n"
                f"   columns={list(df.columns)}\n"
                f"   need cycle/cycle, SOH/soh, {volt_col or 'voltage'}, {ic_col or 'ic_pred/ic_smoothed'}"
            )

        # SOH 없는 row 제거 → cycle 단위 집계 시 NaN 사이클 제거됨
        df = df.dropna(subset=[soh_col])

        # 사이클별 포인트 수 확인 (가장 흔한 개수 사용)
        counts = df.groupby(cyc_col).size()
        n_mode = int(counts.mode().iloc[0]) if not counts.empty else fallback_n
        n_points = max(n_mode, fallback_n)  # 보통 600

        # 공통 전압 격자
        v_ref = np.linspace(v_min, v_max, n_points)

        rows = []
        for cyc, g in df.groupby(cyc_col):
            g = g.sort_values(volt_col)
            v = g[volt_col].values
            ic = g[ic_col].values
            # 보간 (전압 범위 좁으면 그대로 interpolate; 외삽은 단순 nearest로 처리)
            ic_interp = np.interp(v_ref, v, ic, left=ic[0], right=ic[-1])
            soh_val = float(pd.to_numeric(g[soh_col], errors='coerce').dropna().iloc[0])
            row = {'battery': battery_id, 'cycle': int(cyc), 'SOH': soh_val}
            # v_0.. 채우기
            for i in range(n_points):
                row[f'v_{i}'] = ic_interp[i]
            rows.append(row)

        wide = pd.DataFrame(rows)
        wide = wide.sort_values(['cycle']).reset_index(drop=True)
        return wide

# ---- 모든 파일 로드 & 합치기 ----
wide_list = []
for bat, p in battery_map.items():
    try:
        wide_list.append(_load_one_file(bat, p))
    except Exception as e:
        print(f"[경고] {bat} 로드 실패: {e}")

if not wide_list:
    raise RuntimeError("사용할 데이터가 없습니다. 파일 경로/형식을 확인하세요.")

all_df = pd.concat(wide_list, ignore_index=True)

# ---- X / y 구성 ----
curve_cols = _detect_wide_columns(all_df.columns)
if not curve_cols:
    # 안전장치: v_0.. 형태가 아니면 숫자형 v_*만 다시 탐색
    curve_cols = sorted([c for c in all_df.columns if c.startswith('v_')],
                        key=lambda x: int(x.split('_')[1]))

if not curve_cols:
    raise RuntimeError("곡선 컬럼(v_0.. 형태)을 찾을 수 없습니다.")

# SOH 없는 사이클 전체 제거
all_df = all_df.dropna(subset=['SOH']).reset_index(drop=True)

X = all_df[curve_cols].to_numpy(dtype=np.float32)     # (N, T)
y = all_df['SOH'].to_numpy(dtype=np.float32)          # (N,)

# ---- 분할 (60/20/20) ----
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, shuffle=True
)

print("--- 데이터 분할 후 Shape ---")
print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Valid: X={X_valid.shape}, y={y_valid.shape}")
print(f"Test : X={X_test.shape},  y={y_test.shape}")

# ---- 스케일링 ----
# *X는 곡선 전체의 스케일을 맞추는 게 핵심 → MinMaxScaler(0~1) 또는 StandardScaler 중 선택
# 여기선 네 코드와 맞춰 MinMax 사용 (필요시 Standard로 바꿔도 됨)
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_valid_scaled = scaler_x.transform(X_valid)
X_test_scaled  = scaler_x.transform(X_test)

# y(SOH)는 이미 [0~1] 근처일 가능성 큼. 그래도 스케일러 준비(후에 역변환 편의)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_valid_scaled = scaler_y.transform(y_valid.reshape(-1, 1))
y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1))

# ---- Conv1D 입력용 3D로 변환 ----
X_train_scaled = X_train_scaled[..., np.newaxis]   # (N, T, 1)
X_valid_scaled = X_valid_scaled[..., np.newaxis]
X_test_scaled  = X_test_scaled[..., np.newaxis]

# ---- (선택) 학습 안정화를 위한 약한 노이즈 증강 ----
def add_gaussian_noise(data, noise_level=0.01):
    return data + np.random.normal(0.0, noise_level, size=data.shape).astype(data.dtype)

X_train_noisy = add_gaussian_noise(X_train_scaled, noise_level=0.01)

print("\n--- 최종 데이터 Shape (Conv1D 입력용) ---")
print(f"X_train_noisy: {X_train_noisy.shape}")
print(f"X_valid_scaled: {X_valid_scaled.shape}")

print(f"X_test_scaled : {X_test_scaled.shape}")

print(f"y_train_scaled : {y_train_scaled.shape}")
print(f"y_valid_scaled : {y_valid_scaled.shape}")

# (참고) 메타 보관: 나중에 배터리/사이클별 플롯/분석용
df_train_index = all_df.iloc[:len(X_train)].copy()  # 필요시 실제 인덱스로 맞게 저장 로직 변경

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dropout, Flatten, Dense,LayerNormalization
from tensorflow.keras.models import Model
!pip install keras_tuner
import keras_tuner as kt
import tensorflow as tf

def build_tunable_cnn_lstm(hp):
    seq_len = 600
    input_seq = Input(shape=(seq_len, 1))
    x = input_seq

    # CNN block 1
    filters1 = hp.Choice('filters1', [16, 32])
    kernel_size1 = hp.Choice('kernel_size1', [3, 5])
    x = Conv1D(filters1, kernel_size1, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    if hp.Boolean('maxpool1', default=True):
        x = MaxPooling1D(pool_size=2)(x)

    # CNN block 2
    filters2 = hp.Choice('filters2', [ 32,64,128])
    kernel_size2 = hp.Choice('kernel_size2', [3, 5])
    x = Conv1D(filters2, kernel_size2, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    if hp.Boolean('maxpool2', default=True):
        x = MaxPooling1D(pool_size=2)(x)

    # 3rd CNN block (선택적)
    if hp.Boolean('add_cnn3', default=False):
        filters3 = hp.Choice('filters3', [64,128,256])
        kernel_size3 = hp.Choice('kernel_size3', [3, 5])
        x = Conv1D(filters3, kernel_size3, padding='same', activation='relu')(x)
        x = LayerNormalization()(x)
        if hp.Boolean('maxpool3', default=True):
            x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(hp.Float('dropout_cnn', 0.2,0.5,step=0.05))(x)

    # LSTM block 1
    lstm_units1 = hp.Choice('lstm_units1', [16, 24, 32, 48, 64])
    return_seq_1 = hp.Boolean('return_seq_1', default=True)
    x = LSTM(lstm_units1, return_sequences=return_seq_1)(x)

    # LSTM block 2 (선택적)
    if return_seq_1 and hp.Boolean('add_lstm2', default=False):
        lstm_units2 = hp.Choice('lstm_units2', [8, 12, 16, 24, 32])
        x = LSTM(lstm_units2, return_sequences=False)(x)
    elif not return_seq_1:
        x = Flatten()(x)
    else:
        x = Flatten()(x)

    # Dense layers
    dense_units1 = hp.Choice('dense_units1', [16, 32, 48, 64])
    x = Dense(dense_units1, activation='relu')(x)
    x = Dropout(hp.Float('dropout_dense1', 0.2, 0.5, step=0.1))(x)


    output = Dense(1)(x)

    model = Model(inputs=input_seq, outputs=output)
    lr = hp.Choice('learning_rate', [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    return model




early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=6, factor=0.5)

tuner = kt.RandomSearch(
    build_tunable_cnn_lstm,
    objective='val_loss',
    max_trials=100,
    executions_per_trial=1,
    directory='tune_cnn_lstm_final4',
    project_name='soh_pred'
)

for batch_size in [16, 32, 48]:
    tuner.search(
        X_train_scaled, y_train_scaled,
        epochs=200,
        batch_size=batch_size,
        validation_data=(X_valid_scaled, y_valid_scaled),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

best_model_soh= tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(1)[0]

print("Best hyperparameters:")
print(best_hp.values)



# =========================
# 5) 배터리별 SOH 곡선 (True vs Pred)
# =========================
model = best_model_soh

# 예측 편의를 위해 전체 데이터에 대해 예측(배터리별 슬라이스)
X_all_scaled = scaler_x.transform(all_df[curve_cols].values)[..., np.newaxis]
y_all_pred_scaled = model.predict(X_all_scaled, verbose=0)
y_all_pred = scaler_y.inverse_transform(y_all_pred_scaled).ravel()

# all_df에 예측값 붙이기
all_df_plot = all_df.copy()
all_df_plot['SOH_pred'] = y_all_pred

# 배터리 목록
all_battery_names = sorted(all_df_plot['battery'].unique().tolist())

for bat_name in all_battery_names:
    dfb = all_df_plot[all_df_plot['battery'] == bat_name].copy()
    if dfb.empty:
        continue
    # 사이클 기준 정렬
    dfb = dfb.sort_values('cycle')
    cycles = dfb['cycle'].astype(int).values
    y_true_battery = dfb['SOH'].values
    y_pred_battery = dfb['SOH_pred'].values

    plt.figure(figsize=(12, 7))
    plt.plot(cycles, y_true_battery, 'b-o', label='True SOH', linewidth=2)
    plt.plot(cycles, y_pred_battery, 'r--x', label='Predicted SOH', linewidth=2)

    plt.title(f'SOH Degradation: True vs Predicted — {bat_name}', fontsize=16)
    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel('SOH', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()