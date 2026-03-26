#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "hardware/clocks.h"
#include "uart_rx.pio.h"   // pico_generate_pio_header で自動生成
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "sensor_model_data.h"

// ----------------------------------------------------------------
// ハードウェア設定
// ----------------------------------------------------------------
#define PIO_UART     pio0
#define BAUD_RATE    115200
#define UART_RX_PIN  10   // GPIO10（PIO ソフトシリアル RX）
#define LED_PIN      25   // Pico 2 オンボード LED

// ----------------------------------------------------------------
// TFLite 設定
// ----------------------------------------------------------------
constexpr int kTensorArenaSize = 1024 * 60;
static uint8_t tensor_arena[kTensorArenaSize];

constexpr int N_TIME_STEPS = 100;
constexpr int N_CHANNELS   = 2;

// ----------------------------------------------------------------
// 受信バッファ
// ----------------------------------------------------------------
static char  line_buf[64];
static int   line_pos    = 0;

static float sample_buf[N_TIME_STEPS][N_CHANNELS];
static int   sample_count = 0;

// "xx,yy" を 1 行パースして sample_buf に追加、100 個で true を返す
static bool push_line(const char* line) {
    float x, y;
    if (sscanf(line, "%f,%f", &x, &y) != 2) {
        return false;
    }
    // 学習時と同じ正規化（-128~127 → -1.0~1.0）
    x /= 128.0f;
    y /= 128.0f;
    sample_buf[sample_count][0] = x;
    sample_buf[sample_count][1] = y;
    ++sample_count;
    return sample_count >= N_TIME_STEPS;
}

// ----------------------------------------------------------------
// メイン
// ----------------------------------------------------------------
int main() {
    stdio_init_all();
    tflite::InitializeTarget();

    // LED 初期化（消灯）
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    gpio_put(LED_PIN, 0);

    // PIO UART 初期化（GPIO10 = RX、ソフトシリアル）
    uint offset = pio_add_program(PIO_UART, &uart_rx_program);
    uint sm     = pio_claim_unused_sm(PIO_UART, true);
    uart_rx_program_init(PIO_UART, sm, offset, UART_RX_PIN, BAUD_RATE);

    sleep_ms(2000);
    printf("=== Sensor Inference on RP2350 ===\n");
    printf("PIO UART RX: GPIO%d, %d baud\n", UART_RX_PIN, BAUD_RATE);

    // 1. モデルのロード
    const tflite::Model* model = tflite::GetModel(g_sensor_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model version mismatch!\n");
        return 1;
    }

    // 2. オペレータの登録
    // TFLite は Conv1D を Conv2D + ExpandDims/Squeeze に変換するため 2D 系を登録する
    static tflite::MicroMutableOpResolver<9> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();              // GlobalAveragePooling1D 用
    resolver.AddFullyConnected();    // Dense 用
    resolver.AddRelu();
    resolver.AddLogistic();          // Sigmoid 用
    resolver.AddExpandDims();        // 1D→2D 変換用
    resolver.AddSqueeze();           // 2D→1D 変換用
    resolver.AddReshape();

    // 3. インタープリタの構築
    static tflite::MicroInterpreter interpreter(
            model, resolver, tensor_arena, kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors() failed\n");
        return 1;
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("Arena used: %d bytes\n", (int)interpreter.arena_used_bytes());
    // 入力テンソルの型と形状を確認
    printf("Input type: %d (1=float32), dims: %d\n",
           input->type, input->dims->size);
    for (int i = 0; i < input->dims->size; ++i) {
        printf("  dim[%d] = %d\n", i, input->dims->data[i]);
    }
    printf("Collecting %d samples (format: xx,yy)...\n", N_TIME_STEPS);

    // 4. メインループ：PIO UART から 1 文字ずつ受信してラインを組み立てる
    while (true) {
        if (!uart_rx_program_readable(PIO_UART, sm)) {
            continue;
        }

        char c = uart_rx_program_getc(PIO_UART, sm);

        if (c == '\n' || c == '\r') {
            if (line_pos == 0) continue; // 空行は無視

            line_buf[line_pos] = '\0';
            line_pos = 0;

            bool full = push_line(line_buf);
            printf("[%d/%d] %s\n", sample_count, N_TIME_STEPS, line_buf);

            if (full) {
                // 5. 入力テンソルに格納（形状 [1, 100, 2]）
                for (int t = 0; t < N_TIME_STEPS; ++t) {
                    for (int ch = 0; ch < N_CHANNELS; ++ch) {
                        input->data.f[t * N_CHANNELS + ch] = sample_buf[t][ch];
                    }
                }
                sample_count = 0; // バッファをリセットして次の 100 サンプルへ

                // デバッグ：最初の3サンプルの入力値を確認
                printf("Input[0]: %.3f, %.3f\n",
                       (double)input->data.f[0], (double)input->data.f[1]);
                printf("Input[1]: %.3f, %.3f\n",
                       (double)input->data.f[2], (double)input->data.f[3]);
                printf("Input[2]: %.3f, %.3f\n",
                       (double)input->data.f[4], (double)input->data.f[5]);

                // 6. 推論実行
                uint32_t t0 = to_ms_since_boot(get_absolute_time());
                if (interpreter.Invoke() != kTfLiteOk) {
                    printf("Invoke() failed\n");
                    continue;
                }
                uint32_t t1 = to_ms_since_boot(get_absolute_time());

                // 7. 結果取得（Sigmoid 出力: 0.0〜1.0）
                float prediction = output->data.f[0];
                int   label      = (prediction > 0.5f) ? 1 : 0;

                // 8. LED 制御（label=1 で 1 秒点灯、0 で消灯）
                if (label == 1) {
                    gpio_put(LED_PIN, 1);
                    sleep_ms(1000);
                    gpio_put(LED_PIN, 0);
                }

                printf("Pred: %.4f -> Label: %d (%u ms)\n",
                       (double)prediction, label, (unsigned)(t1 - t0));
                printf("Collecting next %d samples...\n", N_TIME_STEPS);
            }
        } else if (line_pos < (int)sizeof(line_buf) - 1) {
            line_buf[line_pos++] = c;
        }
    }

    return 0;
}
