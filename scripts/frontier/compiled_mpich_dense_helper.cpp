#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int REQUIRED_THREAD_LEVEL = MPI_THREAD_SERIALIZED;

struct BucketDescriptor {
    int index = 0;
    std::int64_t nbytes = 0;
    std::string checksum;
    std::string path;
};

struct Request {
    std::string run_id;
    int rank = -1;
    int world_size = -1;
    int generation = -1;
    int base_generation = -1;
    int quorum = -1;
    double timeout_s = 900.0;
    std::int64_t payload_bytes = 0;
    std::string header_path;
    std::vector<BucketDescriptor> buckets;
};

struct TensorEntry {
    std::string name;
    std::string dtype;
    std::int64_t offset = 0;
    std::int64_t nbytes = 0;
    std::int64_t numel = 0;
};

struct LossValues {
    bool has_loss = false;
    bool has_loss_100 = false;
    double loss = 0.0;
    double loss_100 = 0.0;
};

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void write_text_atomic(const fs::path& path, const std::string& text);

void trace_event(const fs::path& request_path, const std::string& event, int rank) {
    const char* trace_dir = std::getenv("ASYNC_COMPILED_MPICH_TRACE_DIR");
    if (trace_dir == nullptr || std::string(trace_dir).empty()) {
        return;
    }
    try {
        fs::create_directories(trace_dir);
        const char* procid = std::getenv("SLURM_PROCID");
        std::ostringstream name;
        name << "rank_" << (procid == nullptr ? std::to_string(rank) : std::string(procid))
             << "." << event << ".txt";
        std::ostringstream body;
        body << "event=" << event << "\n";
        body << "rank=" << rank << "\n";
        body << "request_path=" << request_path.string() << "\n";
        write_text_atomic(fs::path(trace_dir) / name.str(), body.str());
    } catch (...) {
    }
}

std::vector<char> read_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return std::vector<char>(std::istreambuf_iterator<char>(in), {});
}

void write_bytes_atomic(const fs::path& path, const std::vector<char>& data) {
    fs::create_directories(path.parent_path());
    fs::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("cannot write " + tmp.string());
        }
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
    }
    fs::rename(tmp, path);
}

void write_text_atomic(const fs::path& path, const std::string& text) {
    std::vector<char> data(text.begin(), text.end());
    write_bytes_atomic(path, data);
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

std::string find_string(const std::string& text, const std::string& key, bool required = true) {
    std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch m;
    if (std::regex_search(text, m, re)) {
        return m[1].str();
    }
    if (required) {
        throw std::runtime_error("missing string field " + key);
    }
    return "";
}

long long find_int(const std::string& text, const std::string& key, bool required = true, long long fallback = 0) {
    std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch m;
    if (std::regex_search(text, m, re)) {
        return std::stoll(m[1].str());
    }
    if (required) {
        throw std::runtime_error("missing int field " + key);
    }
    return fallback;
}

double find_double(const std::string& text, const std::string& key, bool required = true, double fallback = 0.0) {
    std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
    std::smatch m;
    if (std::regex_search(text, m, re)) {
        return std::stod(m[1].str());
    }
    if (required) {
        throw std::runtime_error("missing double field " + key);
    }
    return fallback;
}

bool find_bool(const std::string& text, const std::string& key, bool fallback = false) {
    std::regex re("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch m;
    if (std::regex_search(text, m, re)) {
        return m[1].str() == "true";
    }
    return fallback;
}

Request parse_request(const fs::path& request_path) {
    std::string text = read_text(request_path);
    Request req;
    req.run_id = find_string(text, "run_id");
    req.rank = static_cast<int>(find_int(text, "rank"));
    req.world_size = static_cast<int>(find_int(text, "world_size"));
    req.generation = static_cast<int>(find_int(text, "generation"));
    req.base_generation = static_cast<int>(find_int(text, "base_generation"));
    req.quorum = static_cast<int>(find_int(text, "quorum"));
    req.timeout_s = find_double(text, "timeout_s", false, 900.0);
    req.payload_bytes = find_int(text, "payload_bytes", false, 0);
    req.header_path = find_string(text, "header_path");

    std::regex bucket_re(
        "\\{\"checksum_sha256\"\\s*:\\s*\"([^\"]*)\"\\s*,\\s*"
        "\"index\"\\s*:\\s*([0-9]+)\\s*,\\s*"
        "\"ipc\"\\s*:\\s*\\{\\s*\"kind\"\\s*:\\s*\"file\"\\s*,\\s*"
        "\"offset\"\\s*:\\s*0\\s*,\\s*\"path\"\\s*:\\s*\"([^\"]*)\"\\s*\\}\\s*,\\s*"
        "\"nbytes\"\\s*:\\s*([0-9]+)\\s*\\}",
        std::regex::ECMAScript);
    for (auto it = std::sregex_iterator(text.begin(), text.end(), bucket_re);
         it != std::sregex_iterator(); ++it) {
        BucketDescriptor bucket;
        bucket.checksum = (*it)[1].str();
        bucket.index = std::stoi((*it)[2].str());
        bucket.path = (*it)[3].str();
        bucket.nbytes = std::stoll((*it)[4].str());
        req.buckets.push_back(bucket);
    }
    std::sort(req.buckets.begin(), req.buckets.end(), [](const auto& a, const auto& b) {
        return a.index < b.index;
    });
    return req;
}

std::vector<TensorEntry> parse_tensor_entries(const std::string& header) {
    std::vector<TensorEntry> entries;
    std::size_t tensors_pos = header.rfind("\"tensors\"");
    if (tensors_pos == std::string::npos) {
        throw std::runtime_error("request header has no top-level tensor metadata");
    }
    std::string tensor_section = header.substr(tensors_pos);
    std::regex tensor_re(
        "\\{\"checksum_sha256\"\\s*:\\s*\"[^\"]*\"\\s*,\\s*"
        "\"dtype\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*"
        "\"name\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*"
        "\"nbytes\"\\s*:\\s*([0-9]+)\\s*,\\s*"
        "\"numel\"\\s*:\\s*([0-9]+)\\s*,\\s*"
        "\"offset\"\\s*:\\s*([0-9]+)\\s*,\\s*"
        "\"shape\"\\s*:\\s*\\[[^\\]]*\\]\\s*\\}",
        std::regex::ECMAScript);
    for (auto it = std::sregex_iterator(tensor_section.begin(), tensor_section.end(), tensor_re);
         it != std::sregex_iterator(); ++it) {
        TensorEntry entry;
        entry.dtype = (*it)[1].str();
        entry.name = (*it)[2].str();
        entry.nbytes = std::stoll((*it)[3].str());
        entry.numel = std::stoll((*it)[4].str());
        entry.offset = std::stoll((*it)[5].str());
        entries.push_back(entry);
    }
    if (entries.empty()) {
        throw std::runtime_error("request header has no tensor metadata");
    }
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return a.offset < b.offset;
    });
    return entries;
}

LossValues parse_loss_values(const std::string& header) {
    LossValues values;
    std::regex loss_re("\"loss_window\"\\s*:\\s*\\{([^}]*)\\}");
    std::smatch m;
    if (!std::regex_search(header, m, loss_re)) {
        return values;
    }
    std::string body = m[1].str();
    std::regex item_re("\"([^\"]+)\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
    for (auto it = std::sregex_iterator(body.begin(), body.end(), item_re);
         it != std::sregex_iterator(); ++it) {
        std::string key = (*it)[1].str();
        double value = std::stod((*it)[2].str());
        if (key == "loss") {
            values.has_loss = true;
            values.loss = value;
        } else if (key == "loss_100") {
            values.has_loss_100 = true;
            values.loss_100 = value;
        }
    }
    return values;
}

std::string thread_level_name(int level) {
    if (level == MPI_THREAD_SINGLE) return "MPI_THREAD_SINGLE";
    if (level == MPI_THREAD_FUNNELED) return "MPI_THREAD_FUNNELED";
    if (level == MPI_THREAD_SERIALIZED) return "MPI_THREAD_SERIALIZED";
    if (level == MPI_THREAD_MULTIPLE) return "MPI_THREAD_MULTIPLE";
    return "unknown";
}

std::string rel_aggregate_bucket(int rank, int generation, int bucket) {
    char buf[160];
    std::snprintf(
        buf,
        sizeof(buf),
        "rank_%05d/gen%06d/aggregate.bucket%05d.bin",
        rank,
        generation,
        bucket);
    return std::string(buf);
}

std::string rel_rank_header(int rank, int generation) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "rank_%05d/gen%06d/header.json", rank, generation);
    return std::string(buf);
}

std::size_t element_size_for_dtype(const std::string& dtype) {
    if (dtype == "float32" || dtype == "float") return 4;
    if (dtype == "float64" || dtype == "double") return 8;
    if (dtype == "bfloat16" || dtype == "float16" || dtype == "half") return 2;
    throw std::runtime_error("unsupported aggregate reduce dtype: " + dtype);
}

float read_float32(const char* ptr) {
    float value = 0.0f;
    std::memcpy(&value, ptr, sizeof(float));
    return value;
}

double read_float64(const char* ptr) {
    double value = 0.0;
    std::memcpy(&value, ptr, sizeof(double));
    return value;
}

float read_bfloat16(const char* ptr) {
    std::uint16_t bf = 0;
    std::memcpy(&bf, ptr, sizeof(std::uint16_t));
    std::uint32_t bits = static_cast<std::uint32_t>(bf) << 16;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(float));
    return value;
}

std::uint16_t float_to_bfloat16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    return static_cast<std::uint16_t>(bits >> 16);
}

void append_encoded_value(std::vector<char>& out, const std::string& dtype, double value) {
    if (dtype == "float32" || dtype == "float") {
        float v = static_cast<float>(value);
        const char* p = reinterpret_cast<const char*>(&v);
        out.insert(out.end(), p, p + sizeof(float));
    } else if (dtype == "float64" || dtype == "double") {
        double v = value;
        const char* p = reinterpret_cast<const char*>(&v);
        out.insert(out.end(), p, p + sizeof(double));
    } else if (dtype == "bfloat16") {
        std::uint16_t v = float_to_bfloat16(static_cast<float>(value));
        const char* p = reinterpret_cast<const char*>(&v);
        out.insert(out.end(), p, p + sizeof(std::uint16_t));
    } else if (dtype == "float16" || dtype == "half") {
        throw std::runtime_error("float16 aggregate reduce is not implemented; use float32 or bfloat16 dense deltas");
    } else {
        throw std::runtime_error("unsupported aggregate reduce dtype: " + dtype);
    }
}

void append_bucket_weighted_values(
    std::vector<double>& values,
    const std::vector<char>& bucket,
    std::int64_t bucket_stream_offset,
    const std::vector<TensorEntry>& tensors,
    double weight) {
    for (const auto& tensor : tensors) {
        if (tensor.offset < bucket_stream_offset || tensor.offset >= bucket_stream_offset + static_cast<std::int64_t>(bucket.size())) {
            continue;
        }
        std::int64_t local_offset = tensor.offset - bucket_stream_offset;
        if (local_offset < 0 || local_offset + tensor.nbytes > static_cast<std::int64_t>(bucket.size())) {
            throw std::runtime_error("tensor metadata crosses aggregate bucket boundary");
        }
        std::size_t elem_size = element_size_for_dtype(tensor.dtype);
        if (tensor.nbytes % static_cast<std::int64_t>(elem_size) != 0) {
            throw std::runtime_error("tensor byte length is not divisible by dtype element size");
        }
        const char* base = bucket.data() + local_offset;
        for (std::int64_t i = 0; i < tensor.nbytes; i += static_cast<std::int64_t>(elem_size)) {
            double value = 0.0;
            if (tensor.dtype == "float32" || tensor.dtype == "float") {
                value = static_cast<double>(read_float32(base + i));
            } else if (tensor.dtype == "float64" || tensor.dtype == "double") {
                value = read_float64(base + i);
            } else if (tensor.dtype == "bfloat16") {
                value = static_cast<double>(read_bfloat16(base + i));
            } else {
                throw std::runtime_error("unsupported aggregate reduce dtype: " + tensor.dtype);
            }
            values.push_back(value * weight);
        }
    }
}

std::vector<char> encode_mean_bucket(
    const std::vector<double>& sums,
    std::int64_t bucket_stream_offset,
    std::int64_t bucket_nbytes,
    const std::vector<TensorEntry>& tensors,
    double total_weight) {
    double denominator = total_weight > 0.0 ? total_weight : 1.0;
    std::vector<char> out;
    std::size_t cursor = 0;
    for (const auto& tensor : tensors) {
        if (tensor.offset < bucket_stream_offset ||
            tensor.offset >= bucket_stream_offset + bucket_nbytes) {
            continue;
        }
        std::int64_t local_offset = tensor.offset - bucket_stream_offset;
        if (local_offset < 0) {
            continue;
        }
        if (static_cast<std::size_t>(local_offset) != out.size()) {
            throw std::runtime_error("aggregate tensor metadata is not contiguous within bucket");
        }
        std::size_t elem_size = element_size_for_dtype(tensor.dtype);
        std::size_t elems = static_cast<std::size_t>(tensor.nbytes) / elem_size;
        if (cursor + elems > sums.size()) {
            throw std::runtime_error("aggregate reduce sum buffer shorter than tensor metadata");
        }
        for (std::size_t i = 0; i < elems; ++i) {
            append_encoded_value(out, tensor.dtype, sums[cursor + i] / denominator);
        }
        cursor += elems;
    }
    if (cursor != sums.size()) {
        throw std::runtime_error("aggregate reduce sum buffer longer than tensor metadata");
    }
    return out;
}

std::string build_aggregate_result_json(
    const Request& req,
    int world_rank,
    int world_size,
    int provided,
    int accepted_count,
    int stale_count,
    int failed_count,
    int timed_out_count,
    int invalid_count,
    long long accepted_tokens,
    long long accepted_local_steps,
    const std::vector<int>& accepted_ranks,
    const std::vector<int>& stale_ranks,
    const std::vector<int>& failed_ranks,
    const std::vector<int>& timed_out_ranks,
    const std::vector<int>& invalid_ranks,
    const std::vector<std::string>& aggregate_bucket_paths,
    const std::vector<long long>& per_bucket_bytes,
    const std::vector<double>& per_bucket_reduce_s,
    double reduce_total_s,
    double loss_sum,
    double loss_100_sum,
    int loss_count,
    int loss_100_count,
    std::int64_t bytes_sent,
    std::int64_t aggregate_bytes,
    int aggregate_owner_rank,
    const std::string& aggregate_source_header) {
    std::ostringstream out;
    out << "{";
    out << "\"schema_version\":1,";
    out << "\"transport\":\"compiled-cray-mpich-helper-collective-reduce\",";
    out << "\"reducer\":\"mpi_reduce_bucketed_weighted_sum\",";
    out << "\"strict_collective_all_launched_ranks\":true,";
    out << "\"status\":\"" << (accepted_count >= req.quorum ? "advanced" : "deferred") << "\",";
    out << "\"rank\":" << world_rank << ",";
    out << "\"generation\":" << req.generation << ",";
    out << "\"base_generation\":" << req.base_generation << ",";
    out << "\"accepted_count\":" << accepted_count << ",";
    out << "\"stale_count\":" << stale_count << ",";
    out << "\"failed_count\":" << failed_count << ",";
    out << "\"timed_out_count\":" << timed_out_count << ",";
    out << "\"invalid_count\":" << invalid_count << ",";
    out << "\"accepted_tokens\":" << accepted_tokens << ",";
    out << "\"accepted_local_steps\":" << accepted_local_steps << ",";
    auto emit_ints = [&out](const char* key, const std::vector<int>& values) {
        out << "\"" << key << "\":[";
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i) out << ",";
            out << values[i];
        }
        out << "]";
    };
    emit_ints("accepted_ranks", accepted_ranks);
    out << ",";
    emit_ints("stale_ranks", stale_ranks);
    out << ",";
    emit_ints("failed_ranks", failed_ranks);
    out << ",";
    emit_ints("timed_out_ranks", timed_out_ranks);
    out << ",";
    emit_ints("invalid_ranks", invalid_ranks);
    out << ",\"bytes_sent\":" << bytes_sent << ",";
    out << "\"bytes_received\":" << aggregate_bytes << ",";
    out << "\"aggregate_update_bytes\":" << aggregate_bytes << ",";
    out << "\"helper_exit_code\":0,";
    out << "\"mpi\":{\"provided_thread_level\":\"" << thread_level_name(provided)
        << "\",\"world_size\":" << world_size << ",\"root_rank\":0"
        << ",\"collective\":\"MPI_Reduce\"},";
    out << "\"aggregate_payload\":{\"rank\":" << aggregate_owner_rank
        << ",\"source_header_path\":\"" << json_escape(aggregate_source_header)
        << "\",\"bucket_paths\":[";
    for (std::size_t i = 0; i < aggregate_bucket_paths.size(); ++i) {
        if (i) out << ",";
        out << "\"" << json_escape(aggregate_bucket_paths[i]) << "\"";
    }
    out << "]},";
    out << "\"received_payloads\":[],";
    out << "\"reduce_metrics\":{\"bucket_count\":" << aggregate_bucket_paths.size()
        << ",\"aggregate_bucket_count\":" << aggregate_bucket_paths.size()
        << ",\"aggregate_update_bytes\":" << aggregate_bytes
        << ",\"reduce_duration_s\":" << reduce_total_s
        << ",\"per_bucket\":[";
    for (std::size_t i = 0; i < aggregate_bucket_paths.size(); ++i) {
        if (i) out << ",";
        out << "{\"index\":" << i
            << ",\"bytes\":" << per_bucket_bytes[i]
            << ",\"reduce_latency_s\":" << per_bucket_reduce_s[i] << "}";
    }
    out << "]},";
    out << "\"aggregate_loss_window\":{";
    bool wrote_loss = false;
    if (loss_count > 0 && accepted_tokens > 0) {
        out << "\"loss\":" << (loss_sum / static_cast<double>(accepted_tokens));
        wrote_loss = true;
    }
    if (loss_100_count > 0 && accepted_tokens > 0) {
        if (wrote_loss) out << ",";
        out << "\"loss_100\":" << (loss_100_sum / static_cast<double>(accepted_tokens));
    }
    out << "}}";
    return out.str();
}

void run_diagnostic(int rank, int size, int provided) {
    int send_value = rank;
    int recv_value = -1;
    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    MPI_Sendrecv(&send_value, 1, MPI_INT, right, 17,
                 &recv_value, 1, MPI_INT, left, 17,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank == 0) {
        std::cout << "{\"diagnostic\":\"compiled_mpich_dense_helper\","
                  << "\"transport\":\"compiled-cray-mpich-helper-collective-reduce\","
                  << "\"world_size\":" << size << ","
                  << "\"provided_thread_level\":\"" << thread_level_name(provided) << "\","
                  << "\"rank0_received_from\":" << recv_value << "}" << std::endl;
    }
}

int ensure_mpi_initialized(int argc, char** argv, int* provided, bool* initialized_here) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        *initialized_here = false;
        int queried = -1;
        MPI_Query_thread(&queried);
        *provided = queried;
        return MPI_SUCCESS;
    }
    *initialized_here = true;
    return MPI_Init_thread(&argc, &argv, REQUIRED_THREAD_LEVEL, provided);
}

int run_once_impl(const fs::path& ipc_dir, const fs::path& request_path, int rank, int size, int provided) {
    trace_event(request_path, "run_once_enter", rank);
    if (ipc_dir.empty() || request_path.empty()) {
        throw std::runtime_error("--ipc-dir and --request are required");
    }
    Request req = parse_request(request_path);
    trace_event(request_path, "request_parsed", rank);
    if (req.world_size != size) {
        throw std::runtime_error("request world_size does not match MPI world size");
    }
    if (req.rank != rank) {
        throw std::runtime_error("request rank does not match MPI rank");
    }
    if (req.quorum <= 0 || req.quorum > req.world_size) {
        throw std::runtime_error("invalid quorum");
    }

    std::string result_json;
    std::string header = read_text(ipc_dir / req.header_path);
    std::vector<TensorEntry> tensors = parse_tensor_entries(header);
    LossValues losses = parse_loss_values(header);

    int root_generation = req.generation;
    int root_base_generation = req.base_generation;
    MPI_Bcast(&root_generation, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&root_base_generation, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bool stale = req.generation != root_generation || req.base_generation != root_base_generation ||
                 find_int(header, "staleness", false, 0) > 0;
    bool failed = find_bool(header, "failed", false);
    bool timed_out_flag = find_bool(header, "timed_out", false);
    bool invalid = find_bool(header, "invalid", false);
    long long tokens = find_int(header, "tokens", false, 0);
    long long local_steps = find_int(header, "local_steps", false, 0);
    bool accepted = !stale && !failed && !timed_out_flag && !invalid && tokens > 0;
    double weight = accepted ? static_cast<double>(tokens) : 0.0;

    int local_counts[5] = {
        accepted ? 1 : 0,
        stale ? 1 : 0,
        failed ? 1 : 0,
        timed_out_flag ? 1 : 0,
        invalid ? 1 : 0,
    };
    int global_counts[5] = {0, 0, 0, 0, 0};
    MPI_Allreduce(local_counts, global_counts, 5, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    long long local_sums[3] = {
        accepted ? tokens : 0,
        accepted ? local_steps : 0,
        req.payload_bytes,
    };
    long long global_sums[3] = {0, 0, 0};
    MPI_Allreduce(local_sums, global_sums, 3, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    double local_loss[4] = {
        (accepted && losses.has_loss) ? losses.loss * static_cast<double>(tokens) : 0.0,
        (accepted && losses.has_loss_100) ? losses.loss_100 * static_cast<double>(tokens) : 0.0,
        (accepted && losses.has_loss) ? 1.0 : 0.0,
        (accepted && losses.has_loss_100) ? 1.0 : 0.0,
    };
    double global_loss[4] = {0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(local_loss, global_loss, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int rank_lists_local[5] = {
        accepted ? rank : -1,
        stale ? rank : -1,
        failed ? rank : -1,
        timed_out_flag ? rank : -1,
        invalid ? rank : -1,
    };
    std::vector<int> gathered(static_cast<std::size_t>(size * 5), -1);
    MPI_Allgather(rank_lists_local, 5, MPI_INT, gathered.data(), 5, MPI_INT, MPI_COMM_WORLD);

    // Build the established MPICH node topology once per generation.  Only
    // local rank zero owns aggregate memory/files; those node leaders exchange
    // the node sums.  The other seven ranks retain only their active bucket.
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm leader_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(
        MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank = -1;
    MPI_Comm_rank(node_comm, &local_rank);
    int node_leader_rank = rank;
    MPI_Allreduce(&rank, &node_leader_rank, 1, MPI_INT, MPI_MIN, node_comm);
    MPI_Comm_split(
        MPI_COMM_WORLD, local_rank == 0 ? 0 : MPI_UNDEFINED, rank, &leader_comm);

    // Establish a common, bucket-bounded collective schedule before entering
    // the loop. A mismatched descriptor count must fail collectively instead
    // of letting one rank execute fewer Reduce/Bcast calls and deadlock peers.
    long long local_bucket_count = static_cast<long long>(req.buckets.size());
    long long min_bucket_count = 0;
    long long max_bucket_count = 0;
    MPI_Allreduce(&local_bucket_count, &min_bucket_count, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_bucket_count, &max_bucket_count, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (min_bucket_count != max_bucket_count) {
        throw std::runtime_error("MPI ranks have different aggregate bucket counts");
    }

    std::vector<std::string> aggregate_bucket_paths;
    std::vector<long long> per_bucket_bytes;
    std::vector<double> per_bucket_reduce_s;
    std::int64_t aggregate_bytes = 0;
    std::int64_t stream_offset = 0;
    auto reduce_start = std::chrono::steady_clock::now();
    for (const auto& bucket_desc : req.buckets) {
        auto bucket_start = std::chrono::steady_clock::now();
        long long local_bucket_shape[2] = {
            static_cast<long long>(bucket_desc.index),
            static_cast<long long>(bucket_desc.nbytes),
        };
        long long min_bucket_shape[2] = {0, 0};
        long long max_bucket_shape[2] = {0, 0};
        MPI_Allreduce(local_bucket_shape, min_bucket_shape, 2, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(local_bucket_shape, max_bucket_shape, 2, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if (min_bucket_shape[0] != max_bucket_shape[0] || min_bucket_shape[1] != max_bucket_shape[1]) {
            throw std::runtime_error("MPI ranks have different aggregate bucket layouts");
        }
        std::vector<char> bucket = read_bytes(ipc_dir / bucket_desc.path);
        if (static_cast<std::int64_t>(bucket.size()) != bucket_desc.nbytes) {
            throw std::runtime_error("bucket length does not match request descriptor");
        }
        std::vector<double> local_values;
        append_bucket_weighted_values(local_values, bucket, stream_offset, tensors, weight);
        std::vector<double> node_sums(
            local_rank == 0 ? local_values.size() : 0, 0.0);
        MPI_Reduce(
            local_values.data(),
            local_rank == 0 ? node_sums.data() : nullptr,
            static_cast<int>(local_values.size()), MPI_DOUBLE, MPI_SUM, 0, node_comm);
        std::vector<double> leader_sums(
            local_rank == 0 ? local_values.size() : 0, 0.0);
        if (local_rank == 0) {
            MPI_Allreduce(
                node_sums.data(), leader_sums.data(), static_cast<int>(node_sums.size()),
                MPI_DOUBLE, MPI_SUM, leader_comm);
        }
        std::vector<char> aggregate;
        if (local_rank == 0) {
            aggregate = encode_mean_bucket(
                leader_sums, stream_offset, bucket_desc.nbytes, tensors,
                static_cast<double>(global_sums[0]));
        }
        long long aggregate_size = local_rank == 0 ? static_cast<long long>(aggregate.size()) : 0;
        MPI_Bcast(&aggregate_size, 1, MPI_LONG_LONG, 0, node_comm);
        if (aggregate_size < 0 || aggregate_size > static_cast<long long>(INT32_MAX)) {
            throw std::runtime_error("aggregate bucket exceeds MPI byte-count limit");
        }

        // The node-local IPC root is shared by the eight ranks on this node.
        // Materialize exactly one aggregate copy and synchronize visibility;
        // Python consumers stream this leader-owned workspace bucket-by-bucket.
        std::string rel = rel_aggregate_bucket(
            node_leader_rank, req.generation, bucket_desc.index);
        if (local_rank == 0) {
            write_bytes_atomic(ipc_dir / rel, aggregate);
        }
        MPI_Barrier(node_comm);
        aggregate_bucket_paths.push_back(rel);
        per_bucket_bytes.push_back(aggregate_size);
        aggregate_bytes += aggregate_size;
        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - bucket_start).count();
        per_bucket_reduce_s.push_back(elapsed);
        stream_offset += bucket_desc.nbytes;
    }
    if (leader_comm != MPI_COMM_NULL) MPI_Comm_free(&leader_comm);
    MPI_Comm_free(&node_comm);
    double reduce_total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - reduce_start).count();
    auto collect_ranks = [&gathered, size](int column) {
        std::vector<int> ranks;
        for (int i = 0; i < size; ++i) {
            int value = gathered[static_cast<std::size_t>(i * 5 + column)];
            if (value >= 0) ranks.push_back(value);
        }
        return ranks;
    };
    result_json = build_aggregate_result_json(
        req, rank, size, provided,
        global_counts[0], global_counts[1], global_counts[2], global_counts[3], global_counts[4],
        global_sums[0], global_sums[1],
        collect_ranks(0), collect_ranks(1), collect_ranks(2), collect_ranks(3), collect_ranks(4),
        aggregate_bucket_paths, per_bucket_bytes, per_bucket_reduce_s, reduce_total_s,
        global_loss[0], global_loss[1], static_cast<int>(global_loss[2]), static_cast<int>(global_loss[3]),
        global_sums[2], aggregate_bytes, node_leader_rank,
        rel_rank_header(node_leader_rank, req.generation));
    trace_event(request_path, "collective_reduce_complete", rank);

    fs::path result_path = request_path.parent_path();
    char name[64];
    std::snprintf(name, sizeof(name), "result.gen%06d.json", req.generation);
    result_path /= name;
    write_text_atomic(result_path, result_json + "\n");
    trace_event(request_path, "result_written", rank);
    return 0;
}

}  // namespace

extern "C" int compiled_mpich_dense_helper_run_once(const char* ipc_dir_c, const char* request_path_c) {
    fs::path request_path = request_path_c == nullptr ? fs::path() : fs::path(request_path_c);
    trace_event(request_path, "bridge_enter", -1);
    int provided = -1;
    bool initialized_here = false;
    char program[] = "compiled_mpich_dense_helper_bridge";
    char* argv[] = {program, nullptr};
    int argc = 1;
    int rc = ensure_mpi_initialized(argc, argv, &provided, &initialized_here);
    if (rc != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed rc=" << rc << std::endl;
        return 10;
    }

    int rank = -1;
    int size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    trace_event(request_path, "mpi_initialized", rank);

    try {
        if (provided < REQUIRED_THREAD_LEVEL) {
            throw std::runtime_error("MPI thread level below MPI_THREAD_SERIALIZED");
        }
        if (ipc_dir_c == nullptr || request_path_c == nullptr) {
            throw std::runtime_error("ipc_dir and request_path are required");
        }
        return run_once_impl(fs::path(ipc_dir_c), request_path, rank, size, provided);
    } catch (const std::exception& exc) {
        std::cerr << "compiled_mpich_dense_helper error rank=" << rank << ": " << exc.what() << std::endl;
        if (!request_path.empty()) {
            try {
                fs::path error_path = request_path.parent_path() / "error.json";
                write_text_atomic(error_path, std::string("{\"schema_version\":1,\"status\":\"error\",\"error\":\"") +
                                              json_escape(exc.what()) + "\"}\n");
            } catch (...) {
            }
        }
        return 20;
    }
}

int main(int argc, char** argv) {
    fs::path ipc_dir;
    fs::path request_path;
    bool run_once = false;
    bool diagnostic = false;
    bool validate_request = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--ipc-dir" && i + 1 < argc) {
            ipc_dir = argv[++i];
        } else if (arg == "--request" && i + 1 < argc) {
            request_path = argv[++i];
        } else if (arg == "--run-once") {
            run_once = true;
        } else if (arg == "--diagnostic") {
            diagnostic = true;
        } else if (arg == "--validate-request") {
            validate_request = true;
        } else if (arg == "--help") {
            std::cout << "usage: compiled_mpich_dense_helper --diagnostic | --request PATH --validate-request | --ipc-dir DIR --request PATH --run-once\n";
            return 0;
        }
    }

    if (validate_request) {
        try {
            if (request_path.empty()) {
                throw std::runtime_error("--request is required");
            }
            Request req = parse_request(request_path);
            std::cout << "{\"schema_version\":1,\"bucket_count\":" << req.buckets.size() << ",\"bucket_paths\":[";
            for (std::size_t i = 0; i < req.buckets.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << "\"" << json_escape(req.buckets[i].path) << "\"";
            }
            std::cout << "]}" << std::endl;
            return 0;
        } catch (const std::exception& exc) {
            std::cerr << "compiled_mpich_dense_helper validate-request error: " << exc.what() << std::endl;
            return 20;
        }
    }

    int provided = -1;
    bool initialized_here = false;
    int rc = ensure_mpi_initialized(argc, argv, &provided, &initialized_here);
    if (rc != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed rc=" << rc << std::endl;
        return 10;
    }

    int rank = -1;
    int size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int exit_code = 0;
    try {
        if (provided < REQUIRED_THREAD_LEVEL) {
            throw std::runtime_error("MPI thread level below MPI_THREAD_SERIALIZED");
        }
        if (diagnostic) {
            run_diagnostic(rank, size, provided);
        } else if (run_once) {
            run_once_impl(ipc_dir, request_path, rank, size, provided);
        } else {
            throw std::runtime_error("no command selected");
        }
    } catch (const std::exception& exc) {
        std::cerr << "compiled_mpich_dense_helper error rank=" << rank << ": " << exc.what() << std::endl;
        if (!request_path.empty()) {
            try {
                fs::path error_path = request_path.parent_path() / "error.json";
                write_text_atomic(error_path, std::string("{\"schema_version\":1,\"status\":\"error\",\"error\":\"") +
                                              json_escape(exc.what()) + "\"}\n");
            } catch (...) {
            }
        }
        exit_code = 20;
    }

    if (initialized_here) {
        MPI_Finalize();
    }
    return exit_code;
}
