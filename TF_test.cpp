/*
 * TFLite Micro Hello World on Raspberry Pi Pico
 *
 * sin(x) を近似するモデルを使って推論を繰り返す。
 * - USB Serial に推論結果 (x, y_pred, y_true, 誤差) を出力
 * - 内蔵 LED の明るさを推論結果 (sin 波) に応じて PWM で制御
 */

#include <math.h>
#include <stdio.h>

#include "hardware/irq.h"
#include "hardware/pwm.h"
#include "pico/stdlib.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "hello_world_float_model_data.h"

// テンソルアリーナのサイズ (バイト)
// 実際の使用量は RecordingMicroInterpreter で計測できる
constexpr int kTensorArenaSize = 3000;
static uint8_t tensor_arena[kTensorArenaSize];

// 推論する x の範囲: 0 〜 2π
constexpr float kXrange = 2.f * 3.14159265358979f;

// 1サイクルあたりの推論ステップ数 (多いほど滑らか・遅い)
constexpr int kInferencesPerCycle = 50;

// 各ステップの待機時間 (ms)
constexpr int kStepDelayMs = 100;

// ----------------------------------------------------------------
// LED (PWM) ユーティリティ
// ----------------------------------------------------------------

// PWM を初期化して LED ピンに接続する
static void led_pwm_init() {
  gpio_set_function(PICO_DEFAULT_LED_PIN, GPIO_FUNC_PWM);
  uint slice = pwm_gpio_to_slice_num(PICO_DEFAULT_LED_PIN);
  pwm_config cfg = pwm_get_default_config();
  // クロック分周: sysclk / 4 → カウンタが遅くなり PWM 周波数が下がる
  pwm_config_set_clkdiv(&cfg, 4.f);
  pwm_init(slice, &cfg, true);
}

// y_value (-1.0 〜 1.0) を LED の明るさ (0 〜 255) にマッピングして設定する
// PWM レベルは brightness^2 にして視覚的にリニアに見せる
static void led_set_brightness(float y_value) {
  int brightness = (int)(127.5f * (y_value + 1.0f));
  if (brightness < 0) brightness = 0;
  if (brightness > 255) brightness = 255;
  pwm_set_gpio_level(PICO_DEFAULT_LED_PIN, (uint16_t)(brightness * brightness));
}

// ----------------------------------------------------------------
// main
// ----------------------------------------------------------------

int main() {
  // Pico 標準初期化 (USB/UART stdio を有効化)
  stdio_init_all();

  // TFLite Micro ターゲット初期化
  tflite::InitializeTarget();

  // USB シリアルが接続されるまで少し待つ
  sleep_ms(2000);
  printf("=== TFLite Micro Hello World on Raspberry Pi Pico ===\n");

  // ----------------------------------------------------------------
  // 1. モデルのロード
  // ----------------------------------------------------------------
  const tflite::Model* model = tflite::GetModel(g_hello_world_float_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("ERROR: Model schema version %d != supported %d\n",
           model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }
  printf("Model loaded. Schema version: %d\n", model->version());

  // ----------------------------------------------------------------
  // 2. オペレータの登録
  //    hello_world モデルは FullyConnected のみ使用
  // ----------------------------------------------------------------
  static tflite::MicroMutableOpResolver<1> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    printf("ERROR: Failed to add FullyConnected op\n");
    return 1;
  }

  // ----------------------------------------------------------------
  // 3. インタープリタの構築とテンソルのアロケート
  // ----------------------------------------------------------------
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                              kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\n");
    return 1;
  }

  // 入出力テンソルへのポインタを取得
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  printf("Arena used: %d bytes\n", (int)interpreter.arena_used_bytes());
  printf("Input  tensor: type=%d shape=[%d]\n", input->type,
         input->dims->data[0]);
  printf("Output tensor: type=%d shape=[%d]\n", output->type,
         output->dims->data[0]);
  printf("Starting inference loop (step=%dms x %d steps per cycle)...\n\n",
         kStepDelayMs, kInferencesPerCycle);

  // LED 初期化
  led_pwm_init();

  // ----------------------------------------------------------------
  // 4. 推論ループ
  // ----------------------------------------------------------------
  int step = 0;
  while (true) {
    // 現在のステップから x を計算 (0 〜 2π)
    float position =
        static_cast<float>(step) / static_cast<float>(kInferencesPerCycle);
    float x = position * kXrange;

    // 入力テンソルに x を書き込む
    input->data.f[0] = x;

    // 推論実行
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("ERROR: Invoke() failed at x=%.3f\n", (double)x);
      break;
    }

    // 推論結果の取得
    float y_pred = output->data.f[0];
    float y_true = sinf(x);
    float err = fabsf(y_true - y_pred);

    // 結果を USB Serial に出力
    printf("step=%3d  x=%6.3f  y_pred=%7.4f  y_true=%7.4f  err=%.4f\n", step,
           (double)x, (double)y_pred, (double)y_true, (double)err);

    // LED の明るさを推論結果に応じて設定
    led_set_brightness(y_pred);

    // 次のステップへ (サイクル末尾で折り返す)
    step = (step + 1) % kInferencesPerCycle;
    sleep_ms(kStepDelayMs);
  }

  return 0;
}
