#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "modeling.hpp"
#include "tokenizer.hpp"
#include "weights.hpp"

namespace {

struct Args {
    std::string command;
    std::string prompt;
    int max_tokens = 50;
    float temperature = 0.8F;
    int top_k = 40;
    uint64_t seed = 42;
    bool greedy = false;
    std::string weights_dir = "weights";
};

const char* option_value(int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", argv[index]);
        std::exit(2);
    }
    return argv[++index];
}

void usage() {
    std::printf(
        "usage:\n"
        "  gpt2 generate \"<prompt>\" [--max-tokens N] [--temp T]\n"
        "                [--top-k K] [--seed N] [--greedy]\n"
        "                [--weights-dir DIR]\n"
        "  gpt2 logits \"<prompt>\" [--weights-dir DIR]\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }
    Args args;
    args.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--max-tokens") args.max_tokens = std::atoi(option_value(argc, argv, i));
        else if (arg == "--temp") args.temperature = std::atof(option_value(argc, argv, i));
        else if (arg == "--top-k") args.top_k = std::atoi(option_value(argc, argv, i));
        else if (arg == "--seed") args.seed = std::strtoull(option_value(argc, argv, i), nullptr, 10);
        else if (arg == "--greedy") args.greedy = true;
        else if (arg == "--weights-dir") args.weights_dir = option_value(argc, argv, i);
        else if (!arg.empty() && arg[0] != '-') args.prompt = arg;
        else { std::fprintf(stderr, "unknown option: %s\n", arg.c_str()); return 2; }
    }

    try {
        const std::string model_path = args.weights_dir + "/model.safetensors";
        std::fprintf(stderr, "loading tokenizer...\n");
        tk::Tokenizer tokenizer(args.weights_dir + "/encoder.json",
                                args.weights_dir + "/vocab.bpe");
        std::fprintf(stderr, "loading weights (%s)...\n", model_path.c_str());
        gpt2::GPT2 model(gpt2::build_gpt2_weights(gpt2::load_safetensors(model_path), {}));

        if (args.command == "generate") {
            const std::string output = model.generate(
                tokenizer, args.prompt, args.max_tokens,
                args.greedy ? 0.0F : args.temperature, args.top_k, args.seed);
            std::printf("%s\n", output.c_str());
            return 0;
        }
        if (args.command == "logits") {
            const std::vector<int> ids = tokenizer.encode(args.prompt);
            if (ids.empty()) {
                std::fprintf(stderr, "logits requires a non-empty prompt\n");
                return 2;
            }
            std::fprintf(stderr, "prompt tokens (%zu): ", ids.size());
            for (int id : ids) std::fprintf(stderr, "%d ", id);
            std::fprintf(stderr, "\n");
            model.reset_cache();
            const std::vector<float> logits = model.forward(ids, 0);
            std::vector<int> indices(logits.size());
            std::iota(indices.begin(), indices.end(), 0);
            const size_t count = std::min<size_t>(5, indices.size());
            std::partial_sort(indices.begin(), indices.begin() + count, indices.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            std::printf("Top 5 token IDs: [");
            for (size_t i = 0; i < count; ++i) std::printf("%d%s", indices[i], i + 1 < count ? ", " : "");
            std::printf("]\nTop 5 values: [");
            for (size_t i = 0; i < count; ++i) std::printf("%.6f%s", logits[indices[i]], i + 1 < count ? ", " : "");
            std::printf("]\nTop 5 tokens: [");
            for (size_t i = 0; i < count; ++i) {
                const std::string token = tokenizer.decode({indices[i]});
                std::printf("\"%s\"%s", token.c_str(), i + 1 < count ? ", " : "");
            }
            std::printf("]\n");
            return 0;
        }
        usage();
        return 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
