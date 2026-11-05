#include "server-common.h"
#include "common.h"
#include "llama.h"
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char ** argv) {
    if (argc != 5) return 2;
    auto params = llama_model_default_params();
    params.vocab_only = true;
    ggml_backend_dev_t no_devices[] = {nullptr};
    params.devices = no_devices;
    params.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], params);
    if (!model) return 3;
    const auto * vocab = llama_model_get_vocab(model);
    std::ifstream template_file(argv[2]);
    std::stringstream template_text; template_text << template_file.rdbuf();
    server_chat_params options{};
    options.use_jinja = true;
    options.prefill_assistant = false;
    options.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    options.reasoning_budget = 128;
    options.tmpls = common_chat_templates_init(model, template_text.str());
    std::ifstream input(argv[3]);
    std::ofstream output(argv[4]);
    std::string line;
    size_t n = 0;
    while (std::getline(input, line)) {
        auto row = json::parse(line);
        auto request = row.at("request");
        std::vector<raw_buffer> files;
        auto parsed = oaicompat_chat_params_parse(request, options, files);
        const auto text = parsed.at("prompt").get<std::string>();
        const auto tokens = tokenize_chat_prompt(vocab, parsed);
        const auto old_tokens = common_tokenize(vocab, text, true, true);
        json result = {{"key", row.at("key")}, {"group", row.at("group")},
            {"tokens", tokens}, {"old_tokens", old_tokens},
            {"input_spans", parsed.at("chat_input_spans")},
            {"template_bytes_equal", text == row.at("expected_prompt").get<std::string>()},
            {"old_ids_equal", old_tokens == row.at("expected_tokens").get<std::vector<llama_token>>()},
            {"new_ids_equal", tokens == row.at("expected_tokens").get<std::vector<llama_token>>()},
            {"source_sha256", row.at("source_sha256")}};
        output << result.dump() << '\n';
        ++n;
    }
    std::cout << "Processed " << n << " token-only records; model weights not loaded\n";
    llama_model_free(model);
}
