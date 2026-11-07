#pragma once
// Explicit immutable parsed index shared by typed source_index and parser tests.
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "parser.h"

namespace c0 {

struct source_identity {
    uint64_t device = 0;
    uint64_t inode = 0;
    uint64_t inode_hi = 0; // the upper half of a 128-bit file id (Windows)
    uint64_t size = 0;
    bool operator==(const source_identity & other) const {
        return device == other.device && inode == other.inode && inode_hi == other.inode_hi && size == other.size;
    }
};

inline source_identity identity(const expert_pack_source & source) {
    ssd::platform::file_identity file;
    if (!source.file || !ssd::platform::identity(ssd::platform::stream_fd(source.file), file)) { return {}; }
    return {file.device, file.id_lo, file.id_hi, file.size};
}

struct extent {
    int source_id = -1;
    uint64_t offset = 0;
    uint64_t logical_bytes = 0;
    bool alias = false;
    source_identity file;
};

class expert_index {
public:
    static std::unique_ptr<expert_index> open(const description & desc);
    // False means an absent key or a logical range beyond the opened file.
    // There is no transport padding here. No tensor type is inferred: pack v1
    // does not store it; the caller's tensor context supplies that information.
    bool locate(const char * name, int expert, size_t bytes, extent & result, bool include_identity = true) const {
        if (!state_->enabled || !name || !name[0] || expert < 0 || !bytes) { return false; }
        const auto & entries = state_->entries;
        size_t lo = 0;
        size_t hi = entries.size();
        // A tensor whose experts are one entry each in order holds expert e at its first entry plus e. That entry is
        // the binary search's answer when it has the key and the entry before it does not (the list is sorted).
        const auto first = first_entry_.find(std::string_view(name));
        if (first != first_entry_.end() && size_t(expert) < entries.size() - first->second) {
            const size_t i = first->second + size_t(expert);
            const auto & item = entries[i];
            const auto & before = entries[i - (i > first->second)];
            if (item.expert_idx == expert && item.nbytes == bytes && !std::strcmp(item.tensor, name) &&
                    (i == first->second || before.expert_idx != expert || before.nbytes != bytes)) {
                lo = hi = i;
            }
        }
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            const auto & item = state_->entries[mid];
            const int name_cmp = std::strcmp(item.tensor, name);
            const int cmp = name_cmp ? name_cmp : item.expert_idx != expert ?
                (item.expert_idx < expert ? -1 : 1) : item.nbytes != bytes ? (item.nbytes < bytes ? -1 : 1) : 0;
            if (cmp < 0) { lo = mid + 1; } else { hi = mid; }
        }
        if (lo == state_->entries.size()) { return false; }
        const auto & item = state_->entries[lo];
        if (std::strcmp(item.tensor, name) || item.expert_idx != expert || item.nbytes != bytes ||
            item.source_idx < 0 || size_t(item.source_idx) >= state_->sources.size()) { return false; }
        const auto & source = state_->sources[size_t(item.source_idx)];
        if (item.offset > source.file_size || bytes > source.file_size - item.offset) { return false; }
        result = {item.source_idx, item.offset, item.nbytes, item.alias_source, include_identity ? identity(source) : source_identity{}};
        return true;
    }
    // Borrowed descriptor, valid until this index is destroyed; never close it.
    // Buffered handles support file-identity tests. Typed source_index separately
    // requires a direct descriptor before publishing a source view.
    int borrowed_fd(int source_id) const {
        if (source_id < 0 || size_t(source_id) >= state_->sources.size()) { return -1; }
        return ssd::platform::stream_fd(state_->sources[size_t(source_id)].file);
    }
    int borrowed_direct_fd(int source_id) const {
        if (source_id < 0 || size_t(source_id) >= state_->sources.size()) { return -1; }
        return state_->sources[size_t(source_id)].fd_direct;
    }
    size_t source_count() const { return state_->sources.size(); }
    const std::vector<expert_pack_entry> & entries_for_test() const { return state_->entries; }
    source_identity source_identity_for_test(size_t i) const { return identity(state_->sources.at(i)); }
private:
    explicit expert_index(std::unique_ptr<parser_state> state) : state_(std::move(state)) {
        const auto & entries = state_->entries;
        for (size_t i = 0; i < entries.size(); ++i) {
            if (!i || std::strcmp(entries[i - 1].tensor, entries[i].tensor)) { first_entry_.emplace(entries[i].tensor, i); }
        }
    }
    std::unique_ptr<const parser_state> state_;
    // Each tensor's first entry. The views point into the entries, which do not change after open.
    std::unordered_map<std::string_view, size_t> first_entry_;
};

namespace detail {
class builder {
public:
    builder() : owned_(std::make_unique<parser_state>()), g_expert_pack(*owned_) {}
    std::unique_ptr<parser_state> build(const description & desc) {
        g_expert_pack.io_backend = desc.io_backend;
        std::vector<expert_pack_entry> entries;
        if (!append_sources(g_expert_pack, desc, entries)) { return {}; }
        if (entries.empty()) { return {}; }
        if (!finalize_entries(entries, desc.replace_duplicates || !desc.page_cache_packs.empty())) { return {}; }
        g_expert_pack.enabled = true;
        g_expert_pack.inited = true;
        g_expert_pack.entries = std::move(entries);
        return std::move(owned_);
    }
private:
    std::unique_ptr<parser_state> owned_;
    parser_state & g_expert_pack;

};
} // namespace detail

inline std::unique_ptr<expert_index> expert_index::open(const description & desc) {
    detail::builder build;
    auto state = build.build(desc);
    return state ? std::unique_ptr<expert_index>(new expert_index(std::move(state))) : nullptr;
}
} // namespace c0
