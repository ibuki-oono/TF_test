#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/pio.h"
#include "hardware/clocks.h"
#include "hardware/uart.h"
#include "uart_rx.pio.h"
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
#define UART_RX_PIN  10
#define LED_PIN      25

#define RESULT_UART      uart0
#define RESULT_UART_TX   0    // GPIO0 = UART0 TX
#define RESULT_UART_RX   1    // GPIO1 = UART0 RX
#define RESULT_BAUD_RATE 115200

// ----------------------------------------------------------------
// TFLite 設定
// ----------------------------------------------------------------
constexpr int kTensorArenaSize = 1024 * 60;
static uint8_t tensor_arena[kTensorArenaSize];

constexpr int N_TIME_STEPS = 200;
constexpr int N_CHANNELS   = 2;

// ----------------------------------------------------------------
// ダブルバッファ（Core0 が書き込み、Core1 が読み出し）
// Core0 が buf[write_idx] に溜め、満杯になったら Core1 に送る。
// Core0 は次の buf[1-write_idx] に書き始め、Core1 の処理中も継続できる。
// ----------------------------------------------------------------
static float sample_buf[2][N_TIME_STEPS][N_CHANNELS];
static int   write_buf_idx  = 0;  // Core0 が現在書き込んでいるバッファ

// Core0 ローカル状態
static char line_buf[64];
static int  line_pos     = 0;
static int  sample_count = 0;

// "xx,yy" をパースして正規化し sample_buf に追加。100 個で true を返す。
static bool push_line(const char* line) {
    float x, y;
    if (sscanf(line, "%f,%f", &x, &y) != 2) return false;
    x /= 128.0f;
    y /= 128.0f;
    sample_buf[write_buf_idx][sample_count][0] = x;
    sample_buf[write_buf_idx][sample_count][1] = y;
    ++sample_count;
    return sample_count >= N_TIME_STEPS;
}

// ----------------------------------------------------------------
// Core1: TFLite 推論
// ----------------------------------------------------------------
void core1_entry() {
    tflite::InitializeTarget();

    const tflite::Model* model = tflite::GetModel(g_sensor_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("[Core1] Model version mismatch!\n");
        return;
    }

    // Conv1D は TFLite 内部で Conv2D + ExpandDims/Squeeze に変換される
    static tflite::MicroMutableOpResolver<9> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();           // GlobalAveragePooling1D 用
    resolver.AddFullyConnected(); // Dense 用
    resolver.AddRelu();
    resolver.AddLogistic();       // Sigmoid 用
    resolver.AddExpandDims();
    resolver.AddSqueeze();
    resolver.AddReshape();

    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("[Core1] AllocateTensors() failed\n");
        return;
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("[Core1] Ready. Arena: %d bytes\n", (int)interpreter.arena_used_bytes());

    // Core0 に初期化完了を通知
    multicore_fifo_push_blocking(1);

    while (true) {
        // Core0 からバッファインデックスが届くまで待機
        uint32_t buf_idx = multicore_fifo_pop_blocking();

        // 受け取ったバッファを入力テンソルにコピー
        for (int t = 0; t < N_TIME_STEPS; ++t) {
            for (int ch = 0; ch < N_CHANNELS; ++ch) {
                input->data.f[t * N_CHANNELS + ch] = sample_buf[buf_idx][t][ch];
            }
        }

        // 推論実行
        uint32_t t0 = to_ms_since_boot(get_absolute_time());
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("[Core1] Invoke() failed\n");
            continue;
        }
        uint32_t t1 = to_ms_since_boot(get_absolute_time());

        float prediction = output->data.f[0];
        int   label      = (prediction > 0.5f) ? 1 : 0;

        printf("[Core1] Pred: %.4f -> Label: %d (%u ms)\n",
               (double)prediction, label, (unsigned)(t1 - t0));

        // 推論結果を UART0 (GPIO0/1) に送信（小数点以下2桁）
        char result_buf[16];
        snprintf(result_buf, sizeof(result_buf), "%.2f\n", (double)prediction);
        uart_puts(RESULT_UART, result_buf);

        if (label == 1) {
            gpio_put(LED_PIN, 1);
            sleep_ms(1000);
            gpio_put(LED_PIN, 0);
        }
    }
}

// ----------------------------------------------------------------
// Core0: メイン（シリアル受信 + バッファ管理）
// ----------------------------------------------------------------
int main() {
    stdio_init_all();

    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    gpio_put(LED_PIN, 0);

    // UART0 初期化（GPIO0=TX, GPIO1=RX）推論結果送信用
    uart_init(RESULT_UART, RESULT_BAUD_RATE);
    gpio_set_function(RESULT_UART_TX, GPIO_FUNC_UART);
    gpio_set_function(RESULT_UART_RX, GPIO_FUNC_UART);

    // PIO UART 初期化（GPIO10 = RX、ソフトシリアル）
    uint offset = pio_add_program(PIO_UART, &uart_rx_program);
    uint sm     = pio_claim_unused_sm(PIO_UART, true);
    uart_rx_program_init(PIO_UART, sm, offset, UART_RX_PIN, BAUD_RATE);

    sleep_ms(2000);
    printf("=== Sensor Inference on RP2350 (dual-core) ===\n");
    printf("[Core0] Launching Core1...\n");

    // Core1 を起動し、初期化完了を待つ
    multicore_launch_core1(core1_entry);
    multicore_fifo_pop_blocking();

    printf("[Core0] Core1 ready. Collecting into buf[%d]...\n", write_buf_idx);

    // メインループ：PIO UART から 1 文字ずつ受信
    while (true) {
        if (!uart_rx_program_readable(PIO_UART, sm)) continue;

        char c = uart_rx_program_getc(PIO_UART, sm);

        if (c == '\n' || c == '\r') {
            if (line_pos == 0) continue;

            line_buf[line_pos] = '\0';
            line_pos = 0;

            bool full = push_line(line_buf);
            printf("[Core0] [%d/%d] %s\n", sample_count, N_TIME_STEPS, line_buf);

            if (full) {
                int sent_idx = write_buf_idx;
                write_buf_idx = 1 - write_buf_idx;  // バッファを切り替え
                sample_count  = 0;

                // Core1 に処理を依頼（ブロッキング: SIO FIFO が満杯の場合のみ待機）
                multicore_fifo_push_blocking((uint32_t)sent_idx);
                printf("[Core0] -> buf[%d] sent to Core1, collecting into buf[%d]\n",
                       sent_idx, write_buf_idx);
            }
        } else if (line_pos < (int)sizeof(line_buf) - 1) {
            line_buf[line_pos++] = c;
        }
    }

    return 0;
}
