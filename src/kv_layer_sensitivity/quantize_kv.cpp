
#include "ggml.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static bool process(
    int layer,
    const std::string & name,
    const std::string & type,
    const std::string & dir
) {
    const std::string prefix =
        dir + "/layer" + std::to_string(layer) + "_";

    const std::string input =
        prefix + name + "_f16.bin";

    const std::string output =
        prefix + name + "_" + type + "_reconstructed_f32.bin";

    FILE * file = std::fopen(input.c_str(), "rb");

    if (!file) {
        std::fprintf(stderr, "Cannot open %s\n", input.c_str());
        return false;
    }

    // Determine number of F16 values from input file size
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        return false;
    }

    const long file_size = std::ftell(file);

    if (file_size <= 0 ||
        file_size % sizeof(ggml_fp16_t) != 0) {
        std::fprintf(stderr, "Invalid input size: %s\n", input.c_str());
        std::fclose(file);
        return false;
    }

    std::rewind(file);

    const size_t count =
        static_cast<size_t>(file_size) / sizeof(ggml_fp16_t);

    // Q8_0 and Q4_0 require blocks of 32 values
    if (count % QK8_0 != 0 || count % QK4_0 != 0) {
        std::fprintf(stderr, "Invalid quantization block size\n");
        std::fclose(file);
        return false;
    }

    std::vector<ggml_fp16_t> fp16(count);

    const size_t read_count = std::fread(
        fp16.data(),
        sizeof(ggml_fp16_t),
        count,
        file
    );

    std::fclose(file);

    if (read_count != count) {
        std::fprintf(stderr, "Unexpected input size\n");
        return false;
    }

    std::vector<float> original(count);
    std::vector<float> reconstructed(count);

    for (size_t i = 0; i < count; ++i) {
        original[i] = ggml_fp16_to_fp32(fp16[i]);

        if (!std::isfinite(original[i])) {
            std::fprintf(stderr, "Non-finite input value\n");
            return false;
        }
    }

    // Quantize and reconstruct using GGML routines
    if (type == "q8_0") {
        std::vector<block_q8_0> blocks(count / QK8_0);

        quantize_row_q8_0_ref(
            original.data(),
            blocks.data(),
            count
        );

        dequantize_row_q8_0(
            blocks.data(),
            reconstructed.data(),
            count
        );

    } else if (type == "q4_0") {
        std::vector<block_q4_0> blocks(count / QK4_0);

        quantize_row_q4_0_ref(
            original.data(),
            blocks.data(),
            count
        );

        dequantize_row_q4_0(
            blocks.data(),
            reconstructed.data(),
            count
        );

    } else {
        std::fprintf(stderr, "Unsupported quantization type\n");
        return false;
    }

    // Save reconstructed values as F32
    file = std::fopen(output.c_str(), "wb");

    if (!file) {
        std::fprintf(stderr, "Cannot open %s\n", output.c_str());
        return false;
    }

    const size_t written = std::fwrite(
        reconstructed.data(),
        sizeof(float),
        count,
        file
    );

    const int close_result = std::fclose(file);

    if (written != count || close_result != 0) {
        std::fprintf(stderr, "Failed to write %s\n", output.c_str());
        return false;
    }

    std::printf(
        "Saved %s (%zu values)\n",
        output.c_str(),
        count
    );

    return true;
}

int main(int argc, char ** argv) {
    const std::string output_dir =
        argc > 1
            ? argv[1]
            : "results/task31_kv_layer_sensitivity/pilot";

    const int n_layers = 16;

    for (int layer = 0; layer < n_layers; ++layer) {
        for (const std::string & name : {"k", "v"}) {
            for (const std::string & type : {"q8_0", "q4_0"}) {
                if (!process(layer, name, type, output_dir)) {
                    return 1;
                }
            }
        }
    }

    return 0;
}