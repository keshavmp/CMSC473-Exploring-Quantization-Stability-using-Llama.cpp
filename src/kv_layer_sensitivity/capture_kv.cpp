
#include "llama.h"
#include "llama-kv-cache.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const char * model_path =
        "models/Llama-3.2-1B-Instruct-Q2_K.gguf";

    const char * prompt =
        argc > 1 ? argv[1] : "Explain how transformer attention works.";

    const std::string output_dir =
        argc > 2 ? argv[2] : "results/task31_kv_layer_sensitivity/pilot";

    // Initialize llama.cpp backends
    llama_backend_init();

    // Load model with GPU offloading
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;

    llama_model * model =
        llama_model_load_from_file(model_path, model_params);

    if (!model) {
        std::fprintf(stderr, "Failed to load model\n");
        llama_backend_free();
        return 1;
    }

    // Configure context and F16 KV cache
    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 512;
    ctx_params.type_k = GGML_TYPE_F16;
    ctx_params.type_v = GGML_TYPE_F16;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (!ctx) {
        std::fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Tokenize prompt
    const llama_vocab * vocab = llama_model_get_vocab(model);

    int32_t n_tokens = llama_tokenize(
        vocab,
        prompt,
        (int32_t) std::strlen(prompt),
        nullptr,
        0,
        true,
        false
    );

    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }

    if (n_tokens <= 0) {
        std::fprintf(stderr, "Tokenization failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::vector<llama_token> tokens(n_tokens);

    int32_t actual_tokens = llama_tokenize(
        vocab,
        prompt,
        (int32_t) std::strlen(prompt),
        tokens.data(),
        (int32_t) tokens.size(),
        true,
        false
    );

    if (actual_tokens <= 0) {
        std::fprintf(stderr, "Tokenization failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::printf("Prompt tokens: %d\n", actual_tokens);

    // Ensure prompt fits within the context window
    if (actual_tokens > static_cast<int32_t>(llama_n_ctx(ctx))) {
        std::fprintf(stderr, "Prompt exceeds context capacity\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Process prompt in batches
    const int32_t batch_size =
        static_cast<int32_t>(llama_n_batch(ctx));

    for (int32_t offset = 0;
        offset < actual_tokens;
        offset += batch_size) {

        const int32_t chunk_size =
            std::min(batch_size, actual_tokens - offset);

        llama_batch batch = llama_batch_get_one(
            tokens.data() + offset,
            chunk_size
        );

        const int32_t decode_result = llama_decode(ctx, batch);

        if (decode_result != 0) {
            std::fprintf(
                stderr,
                "llama_decode failed at offset %d: %d\n",
                offset,
                decode_result
            );

            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        std::printf(
            "Processed tokens: %d/%d\n",
            offset + chunk_size,
            actual_tokens
        );
    }

    std::printf("Prompt decoded successfully.\n");

    // Wait for Metal operations to finish before reading tensors
    llama_synchronize(ctx);

    // Retrieve internal KV cache
    llama_memory_t memory = llama_get_memory(ctx);

    auto * kv = dynamic_cast<llama_kv_cache *>(memory);

    if (!kv) {
        std::fprintf(stderr, "KV cache cast failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Check occupied cache cells
    const auto & cells = kv->get_cells(0);

    const size_t occupied_tokens = cells.get_used();

    std::printf("Occupied KV cells: %zu\n", occupied_tokens);
    std::printf("First occupied cell: %u\n", cells.used_min());
    std::printf("Last occupied cell + 1: %u\n", cells.used_max_p1());

    // Our pilot assumes contiguous cells beginning at index 0
   if (occupied_tokens != static_cast<size_t>(actual_tokens) ||
        cells.used_min() != 0 ||
        cells.used_max_p1() != occupied_tokens) { {
        std::fprintf(stderr, "Unexpected occupied KV-cell layout\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Save binary tensor data
    auto save_binary = [](
        const std::string & path,
        const std::vector<uint8_t> & data
    ) {
        FILE * file = std::fopen(path.c_str(), "wb");

        if (!file) {
            std::fprintf(stderr, "Failed to open %s\n", path.c_str());
            return false;
        }

        const size_t written = std::fwrite(
            data.data(), 1, data.size(), file
        );

        const int close_result = std::fclose(file);

        if (written != data.size() || close_result != 0) {
            std::fprintf(stderr, "Failed to write %s\n", path.c_str());
            return false;
        }

        return true;
    };

    // Extract K and V from every transformer layer
    const int n_layers = llama_model_n_layer(model);

    std::printf("Transformer layers: %d\n", n_layers);

    for (int layer = 0; layer < n_layers; ++layer) {

        ggml_tensor * k = kv->get_k_storage(layer);
        ggml_tensor * v = kv->get_v_storage(layer);

        if (!k || !v) {
            std::fprintf(
                stderr,
                "K/V tensor retrieval failed at layer %d\n",
                layer
            );
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        // Verify expected F16, non-transposed cache layout
        if (k->type != GGML_TYPE_F16 ||
            v->type != GGML_TYPE_F16 ||
            k->ne[0] != v->ne[0]) {
            std::fprintf(
                stderr,
                "Unexpected KV tensor layout at layer %d\n",
                layer
            );
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        const size_t values_per_token =
            static_cast<size_t>(k->ne[0]);

        const size_t populated_bytes =
            occupied_tokens *
            values_per_token *
            sizeof(uint16_t);

        if (populated_bytes > ggml_nbytes(k) ||
            populated_bytes > ggml_nbytes(v)) {
            std::fprintf(
                stderr,
                "Populated data exceeds tensor size at layer %d\n",
                layer
            );
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        // Read only the populated portion of each tensor
        auto read_populated = [&](ggml_tensor * tensor) {
            std::vector<uint8_t> data(populated_bytes);

            ggml_backend_tensor_get(
                tensor,
                data.data(),
                0,
                populated_bytes
            );

            return data;
        };

        auto k_data = read_populated(k);
        auto v_data = read_populated(v);

        const std::string prefix =
            output_dir + "/layer" + std::to_string(layer);

        const bool saved_k = save_binary(
            prefix + "_k_f16.bin",
            k_data
        );

        const bool saved_v = save_binary(
            prefix + "_v_f16.bin",
            v_data
        );

        if (!saved_k || !saved_v) {
            std::fprintf(
                stderr,
                "Failed to save tensors at layer %d\n",
                layer
            );
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        std::printf(
            "Saved layer %d K/V (%zu bytes each)\n",
            layer,
            populated_bytes
        );
    }

    // Cleanup
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
