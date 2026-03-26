# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2 CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(TF_test C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Add executable. Default name is the project name, version 0.1

# 3. pico-tflmicro ライブラリの追加
add_subdirectory(pico-tflmicro)

add_executable(TF_test
    TF_test.cpp
    pico-tflmicro/examples/hello_world/sensor_model_data.cpp
    pico-tflmicro/examples/hello_world/hello_world_int8_model_data.cpp
)

# uart_rx.pio から uart_rx.pio.h を自動生成する
pico_generate_pio_header(TF_test ${CMAKE_CURRENT_LIST_DIR}/uart_rx.pio)

target_include_directories(TF_test PRIVATE
    pico-tflmicro/src  # クローンしたリポジトリのルート
    pico-tflmicro/examples/hello_world  # モデルデータかつヘッダーファイルのディレクトリ
)

pico_set_program_name(TF_test "TF_test")
pico_set_program_version(TF_test "0.1")

# USB Serial で推論結果を出力する
pico_enable_stdio_uart(TF_test 0)
pico_enable_stdio_usb(TF_test 1)

# Add the standard library to the build
target_link_libraries(TF_test
        pico_stdlib
        pico-tflmicro
        hardware_pio
        hardware_clocks
)

# Add the standard include files to the build
target_include_directories(TF_test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
)

pico_add_extra_outputs(TF_test)

