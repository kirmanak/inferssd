// Drives the llama_decode path llama-completion uses: one multi-token prompt
// batch, then one-token steps, via llama_batch_get_one. Does not call
// llama_ssd_prefill_phase and does not call kimi_cpu_sink_cache_begin.
// A repeated expert on the second one-token step must be a host-cache hit.
#include "llama.h"
#include "llama-ssd.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

static void require(bool ok, const char * why) {
    if (!ok) {
        throw std::runtime_error(why);
    }
}

static uint64_t field_after(const std::string & line, const char * key) {
    const auto at = line.find(key);
    if (at == std::string::npos) {
        throw std::runtime_error(std::string("missing ") + key);
    }
    char * end = nullptr;
    const uint64_t value = std::strtoull(line.c_str() + at + std::strlen(key), &end, 10);
    if (end == line.c_str() + at + std::strlen(key)) {
        throw std::runtime_error(std::string("bad ") + key);
    }
    return value;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: test-completion-cache-phase MODEL EXPERTS_TSV LOG\n");
        return 2;
    }
    const std::string log_path = argv[3];
    try {
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.use_mmap = true;
        mp.use_mlock = false;
        mp.defer_experts = true;
        mp.use_extra_bufts = false;
        using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
        model_ptr model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "model load failed");

        const std::string source = std::string("[\"") + argv[2] + "\"]";
        const llama_ssd_override values[] = {
            {"expert.source", source.c_str()},
            {"vram.budget", "0"},
            {"prefetch.depth", "0"},
        };
        char error[512] = {};
        using option_ptr = std::unique_ptr<llama_ssd_options, decltype(&llama_ssd_options_free)>;
        option_ptr options(llama_ssd_options_create(nullptr, values, 3, nullptr, error, sizeof(error)),
            llama_ssd_options_free);
        require(bool(options), error[0] ? error : "ssd options failed");

        auto cp = llama_context_default_params();
        cp.ssd_options = options.get();
        cp.n_ctx = 64;
        cp.n_batch = 8;
        cp.n_ubatch = 8;
        cp.n_seq_max = 1;
        cp.n_threads = 2;
        cp.n_threads_batch = 2;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = false;
        cp.op_offload = false;
        using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
        context_ptr ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "context create failed");

        struct stderr_capture {
            std::FILE * log = nullptr;
            int saved = -1;
            explicit stderr_capture(const std::string & path) {
                log = std::fopen(path.c_str(), "w");
                if (!log) {
                    throw std::runtime_error("log open failed");
                }
                std::fflush(stderr);
                saved = dup(fileno(stderr));
                if (saved < 0 || dup2(fileno(log), fileno(stderr)) < 0) {
                    throw std::runtime_error("redirect stderr failed");
                }
            }
            ~stderr_capture() {
                std::fflush(stderr);
                if (saved >= 0) {
                    dup2(saved, fileno(stderr));
                    close(saved);
                }
                if (log) {
                    std::fclose(log);
                }
            }
        };

        {
            stderr_capture capture(log_path);
            auto decode_tokens = [&](const std::vector<llama_token> & tokens) {
                llama_token buf[8];
                require(tokens.size() <= 8, "batch too large");
                for (size_t i = 0; i < tokens.size(); ++i) {
                    buf[i] = tokens[i];
                }
                // Same batch llama-completion builds in common_prompt_batch_decode.
                const llama_batch batch = llama_batch_get_one(buf, int32_t(tokens.size()));
                const int rc = llama_decode(ctx.get(), batch);
                if (rc != 0) {
                    throw std::runtime_error("llama_decode failed: " + std::to_string(rc));
                }
            };

            // Multi-token prompt, then two one-token steps. Top-10 of 12 experts
            // cannot be disjoint across two steps, so the second step repeats one.
            decode_tokens({1, 2, 3, 4});
            decode_tokens({5});
            decode_tokens({6});
            ctx.reset();
        }

        std::ifstream in(log_path);
        require(bool(in), "log read failed");
        std::string line, destroy;
        while (std::getline(in, line)) {
            if (line.find("reason=before_destroy") != std::string::npos) {
                destroy = line;
            }
        }
        require(!destroy.empty(), "cache before_destroy line missing");
        const uint64_t hits = field_after(destroy, "hits=");
        const uint64_t admitted = field_after(destroy, "admitted_bytes=");
        std::printf("COMPLETION_CACHE_PHASE hits=%llu admitted_bytes=%llu\n",
            (unsigned long long) hits, (unsigned long long) admitted);
        std::fflush(stdout);
        require(hits > 0, "repeated expert was not a cache hit");
        llama_model_free(model.release());
        llama_backend_free();
        return 0;
    } catch (const std::exception & failure) {
        std::fprintf(stderr, "completion cache phase: %s\n", failure.what());
        return 1;
    }
}
