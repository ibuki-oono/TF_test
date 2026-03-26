#include <stdio.h>
#include "pico/stdlib.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 先ほど生成したヘッダファイルをインクルード
#include "sensor_model_data.h"

// ----------------------------------------------------------------
// 設定・定数
// ----------------------------------------------------------------

// テンソルアリーナのサイズ
// Conv1Dを使用するため、sin波モデルより多めに確保（10KB〜20KB程度）
constexpr int kTensorArenaSize = 1024 * 20; 
static uint8_t tensor_arena[kTensorArenaSize];

constexpr int N_TIME_STEPS = 100;
constexpr int N_CHANNELS = 2;

// ----------------------------------------------------------------
// メイン
// ----------------------------------------------------------------

int main() {
    stdio_init_all();
    tflite::InitializeTarget();

    sleep_ms(2000);
    printf("=== Sensor Inference on RP2040 ===\n");

    // 1. モデルのロード
    const tflite::Model* model = tflite::GetModel(g_sensor_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model version mismatch!\n");
        return 1;
    }

    // 2. オペレータの登録
    // TFLite は Conv1D を Conv2D + ExpandDims/Squeeze に変換するため 2D 系を登録する
    static tflite::MicroMutableOpResolver<9> resolver;
    resolver.AddConv2D();            // Conv1D → Conv2D に変換される
    resolver.AddMaxPool2D();         // MaxPooling1D → MaxPool2D に変換される
    resolver.AddMean();              // GlobalAveragePooling1D 用
    resolver.AddFullyConnected();    // Dense 用
    resolver.AddRelu();              // 活性化関数用
    resolver.AddLogistic();          // 出力の Sigmoid 用
    resolver.AddExpandDims();        // 1D→2D 変換用
    resolver.AddSqueeze();           // 2D→1D 変換用
    resolver.AddReshape();           // 形状変換用

    // 3. インタープリタの構築
    static tflite::MicroInterpreter interpreter(
            model, resolver, tensor_arena, kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors() failed\n");
        return 1;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("Inference ready. Arena used: %d bytes\n", (int)interpreter.arena_used_bytes());

    // 4. 推論ループ（例としてダミーデータで推論）
    while (true) {
        // 本来はここでセンサーから100サンプル溜まったバッファをコピーする
        // 入力形状は [1, 100, 2] なので、data.f[0] 〜 data.f[199] まで埋める
        for (int i = 0; i < N_TIME_STEPS * N_CHANNELS; ++i) {
            // 学習時の正規化（/128.0）を忘れずに適用すること
            input->data.f[i] = 0.0f; 
        }

        // 推論実行
        uint32_t start_time = to_ms_since_boot(get_absolute_time());
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("Invoke() failed\n");
        }
        uint32_t end_time = to_ms_since_boot(get_absolute_time());

        // 結果の取得（Sigmoid出力なので 0.0 〜 1.0）
        float prediction = output->data.f[0];
        int label = (prediction > 0.5f) ? 1 : 0;

        printf("Pred: %.4f -> Label: %d (Time: %u ms)\n",
                (double)prediction, label, (unsigned)(end_time - start_time));

        sleep_ms(500);
    }

    return 0;
}