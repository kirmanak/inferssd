// Multi-token prompt experts whose hidden size is 640 (not a multiple of 256).
// Fitting experts go through Vulkan's existing quantized mul_mat. An expert past
// the device budget stays on the CPU sink. The reference is a budget-0 prefill
// of the same weights, not a second matmul implementation.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-ssd/gpu/vulkan-expert.h"
#include "ggml-ssd/host/arbiter.h"
#include "ggml-ssd/settings/settings.h"
#include "ssd/cpu-prefill.h"
#include "ssd/runtime.h"
#include "test-resources.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

static void require(bool ok, const std::string & what) {
    if (!ok) { throw std::runtime_error(what); }
}

static size_t aligned(size_t n) { return (n + 4095u) & ~size_t(4095); }

static std::string tensor_name(int role) {
    static const char * roles[] = { "up", "gate", "down" };
    return std::string("blk.0.ffn_") + roles[role] + "_exps.weight";
}

struct entry {
    uint64_t offset = 0;
    size_t raw_bytes = 0;
    size_t read_bytes = 0;
    std::vector<uint8_t> data;
};

struct fixture {
    int latent = 256;
    int hidden = 640;
    int experts = 2;
    int fd = -1;
    std::string path;
    ggml_type up_type = GGML_TYPE_Q4_K;
    ggml_type down_type = GGML_TYPE_Q8_0;
    std::map<std::pair<std::string, int32_t>, entry> entries;

    ~fixture() {
        if (fd >= 0) { ssd::platform::close_fd(fd); }
        if (!path.empty()) { std::remove(path.c_str()); }
    }

    static bool resolve(void * user, const char * name, int32_t id, kimi_cpu_sink_source * out) {
        auto & self = *static_cast<fixture *>(user);
        auto it = self.entries.find({ name, id });
        if (it == self.entries.end()) { return false; }
        out->fd_direct = self.fd;
        out->offset = it->second.offset;
        out->nbytes = it->second.read_bytes;
        return true;
    }
};

static void write_role(fixture & f, int role, ggml_type type, int cols, int rows) {
    const size_t bytes = ggml_row_size(type, cols) * size_t(rows);
    for (int expert = 0; expert < f.experts; ++expert) {
        std::vector<float> values(size_t(cols) * size_t(rows));
        for (size_t i = 0; i < values.size(); ++i) {
            // Nonzero, deterministic, small enough that quantized matmul stays finite.
            values[i] = std::sin(float(i + 1) * 0.017f + float(expert + 1) * 0.4f + float(role) * 1.7f) * 0.08f;
        }
        entry item;
        item.raw_bytes = bytes;
        item.read_bytes = aligned(bytes);
        item.data.resize(bytes);
        const size_t wrote = ggml_quantize_chunk(type, values.data(), item.data.data(), 0, rows, cols, nullptr);
        require(wrote == bytes, "quantized expert byte count mismatch");
        f.entries[{ tensor_name(role), expert }] = std::move(item);
    }
}

static void make_fixture(fixture & f) {
    require(f.hidden == 640 && f.hidden % 256 != 0, "expert hidden size must be 640");
    require(f.latent % 256 == 0, "Q4_K up/gate K must be a multiple of 256");
    require(f.hidden % 32 == 0 && f.latent % 32 == 0, "Q8_0 rows must cover whole blocks");
    const std::string dir = "/var/home/kirmanak/models/inferssd/vulkan-expert-test";
    std::filesystem::create_directories(dir);
    f.path = dir + "/fixture-" + std::to_string(getpid()) + ".bin";
    write_role(f, 0, f.up_type, f.latent, f.hidden);
    write_role(f, 1, f.up_type, f.latent, f.hidden);
    write_role(f, 2, f.down_type, f.hidden, f.latent);
    uint64_t off = 4096;
    for (auto & kv : f.entries) {
        kv.second.offset = off;
        off += kv.second.read_bytes;
    }
    std::ofstream file(f.path, std::ios::binary | std::ios::trunc);
    require(bool(file), "fixture create failed");
    std::vector<char> zero(4096, 0);
    auto pad_to = [&](uint64_t target) {
        uint64_t here = static_cast<uint64_t>(file.tellp());
        while (here < target) {
            const size_t count = std::min<uint64_t>(zero.size(), target - here);
            file.write(zero.data(), count);
            require(bool(file), "fixture pad failed");
            here += count;
        }
    };
    pad_to(4096);
    for (const auto & kv : f.entries) {
        pad_to(kv.second.offset);
        file.write(reinterpret_cast<const char *>(kv.second.data.data()), kv.second.raw_bytes);
        require(bool(file), "fixture write failed");
        pad_to(kv.second.offset + kv.second.read_bytes);
    }
    file.close();
    f.fd = test_resources::open_direct(f.path);
}

struct tensors {
    ggml_context * ctx = nullptr;
    ggml_tensor * role[3] = {};
    explicit tensors(const fixture & f) {
        ctx = ggml_init({ ggml_tensor_overhead() * 4, nullptr, true });
        require(ctx, "metadata context failed");
        const ggml_type types[3] = { f.up_type, f.up_type, f.down_type };
        for (int r = 0; r < 3; ++r) {
            role[r] = ggml_new_tensor_3d(ctx, types[r], r == 2 ? f.hidden : f.latent, r == 2 ? f.latent : f.hidden, f.experts);
            ggml_set_name(role[r], tensor_name(r).c_str());
            require(role[r] && role[r]->data == nullptr, "expert metadata must not carry a payload");
        }
    }
    ~tensors() { ggml_free(ctx); }
};

struct held_sink {
    std::shared_ptr<ssd::host::process_arbiter> owner;
    std::unique_ptr<ssd::host::context_client> client;
    ssd::settings::snapshot settings;
    kimi_cpu_sink * sink = nullptr;
    held_sink() = default;
    held_sink(held_sink && other) noexcept
        : owner(std::move(other.owner)), client(std::move(other.client)),
          settings(std::move(other.settings)), sink(other.sink) { other.sink = nullptr; }
    held_sink & operator=(held_sink && other) noexcept {
        if (this != &other) {
            if (sink) { kimi_cpu_sink_destroy(sink); }
            owner = std::move(other.owner);
            client = std::move(other.client);
            settings = std::move(other.settings);
            sink = other.sink;
            other.sink = nullptr;
        }
        return *this;
    }
    held_sink(const held_sink &) = delete;
    held_sink & operator=(const held_sink &) = delete;
    ~held_sink() { if (sink) { kimi_cpu_sink_destroy(sink); } }
};

static held_sink make_sink(fixture & f, uint64_t budget, const ggml_vk_expert_api_v1 * api, ggml_backend_t backend, uint64_t free_bytes) {
    held_sink held;
    std::vector<ssd::settings::override_value> values = { { "vram.budget", std::to_string(budget) }, { "prefetch.depth", "0" } };
    auto parsed = ssd::settings::parse("", "", values);
    require(bool(parsed), parsed.error);
    held.settings = std::move(parsed.value);
    held.owner = std::make_shared<ssd::host::process_arbiter>();
    auto made = held.owner->make_client();
    require(bool(made), made.error);
    held.client = std::move(made.value);
    ssd::runtime_services services{};
    services.host = held.client.get();
    services.vulkan_experts = api;
    services.vulkan_backend = backend;
    services.vulkan_free_bytes = free_bytes;
    char error[512] = {};
    held.sink = ssd::cpu::create_configured(&held.settings, &services, fixture::resolve, &f, error, sizeof(error));
    require(held.sink, std::string("create_configured: ") + error);
    return held;
}

static std::string field(const std::string & line, const char * key) {
    const std::string token = std::string(key) + "=";
    const auto pos = line.find(token);
    if (pos == std::string::npos) { return {}; }
    const auto start = pos + token.size();
    const auto end = line.find(' ', start);
    return line.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

static double rel_l2(const std::vector<float> & got, const std::vector<float> & ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        require(std::isfinite(got[i]) && std::isfinite(ref[i]), "nonfinite expert output");
        const double d = double(got[i]) - double(ref[i]);
        num += d * d;
        den += double(ref[i]) * double(ref[i]);
    }
    require(den > 0, "CPU reference output is zero; a stub would pass");
    return std::sqrt(num / den);
}

int main() {
    try {
        fixture weights;
        make_fixture(weights);
        tensors meta(weights);
        const uint64_t triplet = uint64_t(meta.role[0]->nb[2]) + uint64_t(meta.role[1]->nb[2]) + uint64_t(meta.role[2]->nb[2]);
        require(triplet > 0, "expert triplet is empty");

        constexpr int ntokens = 4;
        constexpr int nslots = 2;
        std::vector<float> input(size_t(ntokens) * weights.latent);
        std::vector<int32_t> ids(size_t(ntokens) * nslots);
        std::vector<float> route_weights(ids.size());
        for (size_t i = 0; i < input.size(); ++i) { input[i] = std::sin(float(i + 3) * 0.09f) * 0.35f; }
        for (int token = 0; token < ntokens; ++token) {
            // First touch is expert 0, then expert 1. Budget of one triplet keeps 1 on the CPU sink.
            ids[size_t(token) * nslots + 0] = 0;
            ids[size_t(token) * nslots + 1] = 1;
            route_weights[size_t(token) * nslots + 0] = 0.35f + 0.01f * float(token);
            route_weights[size_t(token) * nslots + 1] = 0.55f;
        }
        const kimi_cpu_sink_activation activation = { KIMI_CPU_SINK_ACTIVATION_SWIGLU, 0, 0, 0, nullptr, 0 };

        ggml_backend_load_all();
        ggml_backend_reg_t vulkan = nullptr;
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            if (reg && std::strcmp(ggml_backend_reg_name(reg), "Vulkan") == 0) { vulkan = reg; break; }
        }
        require(vulkan && ggml_backend_reg_dev_count(vulkan) > 0, "Vulkan device was not loaded");
        ggml_backend_dev_t device = ggml_backend_reg_dev_get(vulkan, 0);
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        require(backend, "Vulkan backend init failed");
        struct backend_free { ggml_backend_t backend; ~backend_free() { ggml_backend_free(backend); } } guard{ backend };
        ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(device, &props);
        const ggml_vk_expert_api_getter getter = reinterpret_cast<ggml_vk_expert_api_getter>(
                ggml_backend_reg_get_proc_address(vulkan, "ggml_backend_vk_expert_api_v1"));
        const ggml_vk_expert_api_v1 * api = getter ? getter() : nullptr;
        if (!api || api->abi != 1 || api->size != sizeof(*api) || !api->supports || !api->mul_mat || !api->session_create) {
            char error[512] = {};
            ssd::runtime_services services{};
            auto owner = std::make_shared<ssd::host::process_arbiter>();
            auto made = owner->make_client();
            require(bool(made), made.error);
            auto parsed = ssd::settings::parse("", "", { { "vram.budget", std::to_string(triplet) } });
            require(bool(parsed), parsed.error);
            services.host = made.value.get();
            kimi_cpu_sink * rejected = ssd::cpu::create_configured(&parsed.value, &services, fixture::resolve, &weights, error, sizeof(error));
            if (rejected) { kimi_cpu_sink_destroy(rejected); }
            std::printf("VULKAN_EXPERT_FAIL: Vulkan exports no expert provider device=%s create=%s\n",
                    props.description ? props.description : "?",
                    rejected ? "accepted a nonzero budget with no provider" : error);
            return 1;
        }
        require(api->supports(weights.up_type) && api->supports(weights.down_type), "Vulkan mul_mat does not cover the fixture types");

        auto cpu = make_sink(weights, 0, nullptr, nullptr, 0);
        char error[512] = {};
        auto cpu_prefill = std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)>(
                kimi_cpu_prefill_create(cpu.sink, error, sizeof(error)), kimi_cpu_prefill_destroy);
        require(bool(cpu_prefill), std::string("cpu prefill: ") + error);
        std::vector<float> reference(size_t(ntokens) * weights.latent);
        kimi_cpu_prefill_stats cpu_stats = {};
        require(kimi_cpu_prefill_run_activation(cpu_prefill.get(), meta.role[0], meta.role[1], meta.role[2],
                input.data(), ids.data(), route_weights.data(), nslots, ntokens, &activation, reference.data(),
                &cpu_stats, error, sizeof(error)), std::string("cpu prefill run: ") + error);
        require(cpu_stats.gpu_experts == 0, "budget 0 prefill used a GPU expert");

        auto gpu = make_sink(weights, triplet, api, backend, props.memory_free);
        auto gpu_prefill = std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)>(
                kimi_cpu_prefill_create(gpu.sink, error, sizeof(error)), kimi_cpu_prefill_destroy);
        require(bool(gpu_prefill), std::string("vulkan prefill: ") + error);
        const std::string log_path = "/var/home/kirmanak/models/inferssd/vulkan-expert-test/console-" + std::to_string(getpid()) + ".log";
        fflush(stderr);
        const int saved = dup(STDERR_FILENO);
        require(saved >= 0, "dup stderr failed");
        const int log_fd = open(log_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        require(log_fd >= 0, "console log open failed");
        require(dup2(log_fd, STDERR_FILENO) >= 0, "redirect stderr failed");
        std::vector<float> got(reference.size());
        kimi_cpu_prefill_stats gpu_stats = {};
        const bool ran = kimi_cpu_prefill_run_activation(gpu_prefill.get(), meta.role[0], meta.role[1], meta.role[2],
                input.data(), ids.data(), route_weights.data(), nslots, ntokens, &activation, got.data(),
                &gpu_stats, error, sizeof(error));
        fflush(stderr);
        dup2(saved, STDERR_FILENO);
        close(saved);
        close(log_fd);
        require(ran, std::string("vulkan prefill run: ") + error);

        std::ifstream console(log_path);
        std::string line, vulkan_line, source_line;
        while (std::getline(console, line)) {
            if (line.find("[kimi_vulkan_expert]") != std::string::npos) { vulkan_line = line; }
            if (line.find("[kimi_cpu_prefill_sources]") != std::string::npos) { source_line = line; }
        }
        std::remove(log_path.c_str());
        require(!vulkan_line.empty(), "prefill did not report [kimi_vulkan_expert]");
        require(!source_line.empty(), "prefill did not report [kimi_cpu_prefill_sources]");
        require(field(vulkan_line, "gpu_ids") == "0", "fitting expert was not Vulkan matmul id 0: " + vulkan_line);
        require(field(vulkan_line, "cpu_ids") == "1", "over-budget expert was not left on the CPU sink: " + vulkan_line);
        require(field(vulkan_line, "gpu_experts") == "1" && field(vulkan_line, "cpu_experts") == "1", vulkan_line);
        require(std::stod(field(vulkan_line, "gpu_expert_us")) > 0, "gpu_expert_us was not greater than zero");
        require(std::stoull(field(source_line, "vram_experts")) > 0, "source log GPU expert count was zero");
        require(std::stod(field(source_line, "gpu_expert_us")) > 0, "source log gpu_expert_us was zero");
        const uint64_t assigned = std::stoull(field(vulkan_line, "assigned_bytes"));
        const uint64_t device_bytes = std::stoull(field(vulkan_line, "device_bytes"));
        const uint64_t logged_budget = std::stoull(field(vulkan_line, "budget"));
        require(assigned == triplet && assigned <= logged_budget && logged_budget == triplet, "budget assignment mismatch: " + vulkan_line);
        require(device_bytes > 0 && device_bytes <= (64ull << 20), "device staging is outside one reused buffer: " + vulkan_line);
        const double error_l2 = rel_l2(got, reference);
        require(error_l2 <= 2e-2, "Vulkan expert output missed CPU parity, rel L2 " + std::to_string(error_l2));
        std::printf("VULKAN_EXPERT_MATCH hidden=640 up=%s down=%s rel_l2=%.6g gpu_ids=0 cpu_ids=1 device_bytes=%llu budget=%llu\n",
                ggml_type_name(weights.up_type), ggml_type_name(weights.down_type), error_l2,
                (unsigned long long) device_bytes, (unsigned long long) triplet);
        return 0;
    } catch (const std::exception & e) {
        std::printf("VULKAN_EXPERT_FAIL: %s\n", e.what());
        return 1;
    }
}
