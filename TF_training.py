import pandas as pd
import numpy as np
import tensorflow as tf
import os
from glob import glob
import matplotlib.pyplot as plt
import os

# --- 設定値 ---
N_TIME_STEPS = 100
STRIDE = 40
N_CHANNELS = 2
BASE_DIR = '0326'

def process_file(file_path):
    """CSVを読み込み、(時間, チャンネル)に転置してウィンドウ切り出し"""
    try:
        df = pd.read_csv(file_path, header=None)
        data = df.values  # shape: (時間ステップ数, チャンネル数)
        
        # 正規化（必要に応じて。ここでは-128~127を-1.0~1.0にする例）
        data = data.astype(np.float32) / 128.0

        windows = []
        if len(data) < N_TIME_STEPS:
            return np.array([])

        for i in range(0, len(data) - N_TIME_STEPS + 1, STRIDE):
            window = data[i : i + N_TIME_STEPS]
            if window.shape == (N_TIME_STEPS, N_CHANNELS):
                windows.append(window)
        return np.array(windows)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.array([])

def load_data_split_by_file(base_path):
    x_train, y_train = [], []
    x_test_info = [] # 評価用に詳細情報を保持

    for label in ['0', '1']:
        folder_path = os.path.join(base_path, label)
        csv_files = sorted(glob(os.path.join(folder_path, "*.csv")))
        
        if not csv_files:
            continue
            
        # 最初の1ファイルを評価用、残りを学習用
        test_file = csv_files[0]
        train_files = csv_files[1:]

        # 評価用データの読み込み
        tw = process_file(test_file)
        if len(tw) > 0:
            x_test_info.append({
                'filename': os.path.basename(test_file),
                'label': int(label),
                'data': tw
            })

        # 学習用データの読み込み
        for f in train_files:
            trw = process_file(f)
            if len(trw) > 0:
                x_train.extend(trw)
                y_train.extend([int(label)] * len(trw))

    return (np.array(x_train, dtype=np.float32), 
            np.array(y_train, dtype=np.float32).reshape(-1, 1),
            x_test_info)

def save_tflite_as_cpp(tflite_content, model_name="sensor_model"):
    """
    TFLiteのバイナリデータを、提示された hello_world 形式の 
    .h と .cpp ファイルとして保存する
    """
    header_file = f"{model_name}_data.h"
    source_file = f"{model_name}_data.cpp"
    
    # --- 1. ヘッダファイル (.h) の作成 ---
    with open(header_file, "w") as f:
        f.write("#include <cstdint>\n\n")
        f.write(f"extern const unsigned int g_{model_name}_data_size;\n")
        f.write(f"extern const unsigned char g_{model_name}_data[];\n")

    # --- 2. ソースファイル (.cpp) の作成 ---
    with open(source_file, "w") as f:
        f.write("#include <cstdint>\n")
        f.write(f'#include "{header_file}"\n\n')
        f.write(f"const unsigned int g_{model_name}_data_size = {len(tflite_content)};\n")
        f.write(f"alignas(16) const unsigned char g_{model_name}_data[] = {{\n    ")
        
        # バイナリデータを 0xXX 形式の文字列に変換して12個ずつ書き込む
        hex_array = [f"0x{b:02x}" for b in tflite_content]
        for i, val in enumerate(hex_array):
            f.write(val)
            if i < len(hex_array) - 1:
                f.write(",")
            if (i + 1) % 12 == 0:
                f.write("\n    ")
                
        f.write("\n};\n")

    print(f"Generated: {header_file} and {source_file}")

# --- 1. データ準備 ---
X_train, y_train, test_info_list = load_data_split_by_file(BASE_DIR)

if len(X_train) == 0:
    raise RuntimeError(f"学習データが見つかりません。'{BASE_DIR}/0/' と '{BASE_DIR}/1/' にCSVファイルが2件以上あるか確認してください。")

if len(test_info_list) == 0:
    raise RuntimeError(f"評価データが見つかりません。各ラベルフォルダのCSVファイルが1件以上あり、{N_TIME_STEPS}行以上のデータが含まれているか確認してください。")

# 評価データ全体の配列を作成（学習中の検証用）
x_test_all = np.concatenate([info['data'] for info in test_info_list])
y_test_all = np.concatenate([[info['label']] * len(info['data']) for info in test_info_list]).reshape(-1, 1)

print(f"Train samples: {len(X_train)}")
print(f"Evaluation samples: {len(x_test_all)}")

# --- 2. モデル構築 ---
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(N_TIME_STEPS, N_CHANNELS)),
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 3. 学習 ---
history = model.fit(X_train, y_train, epochs=30, batch_size=32, 
                    validation_data=(x_test_all, y_test_all), verbose=1)

# --- 4. 詳細評価 ---
print("\n" + "="*30)
print("DETAILED EVALUATION")
print("="*30)

for info in test_info_list:
    predictions = model.predict(info['data'], verbose=0)
    # 0.5を閾値として判定
    pred_labels = (predictions > 0.5).astype(int)
    accuracy = np.mean(pred_labels == info['label'])
    
    print(f"File: {info['filename']}")
    print(f"  True Label: {info['label']}")
    print(f"  Accuracy:   {accuracy:.2%}")
    print(f"  Raw Output Average: {np.mean(predictions):.4f}")
    
    # 判定ミスがあった場合の表示
    if accuracy < 1.0:
        error_indices = np.where(pred_labels.flatten() != info['label'])[0]
        print(f"  Mistakes at window indices: {error_indices[:10]}...") 

# --- 5. TFLite 保存 ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
save_tflite_as_cpp(tflite_model, model_name="sensor_model")
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print("\nmodel.tflite saved.")