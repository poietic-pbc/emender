#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int TAG_PREAMBLE = 63100;
constexpr int TAG_HEADER = 63101;
constexpr int TAG_BUCKET_SIZE = 63102;
constexpr int TAG_BUCKET_BASE = 63200;
constexpr int TAG_RESULT_SIZE = 63300;
constexpr int TAG_RESULT = 63301;
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

struct PayloadPaths {
    int rank = -1;
    std::string header_path;
    std::vector<std::string> bucket_paths;
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
        "\"index\"\\s*:\\s*([0-9]+).*?\"nbytes\"\\s*:\\s*([0-9]+).*?"
        "\"checksum_sha256\"\\s*:\\s*\"([^\"]*)\".*?\"path\"\\s*:\\s*\"([^\"]*)\"",
        std::regex::ECMAScript);
    for (auto it = std::sregex_iterator(text.begin(), text.end(), bucket_re);
         it != std::sregex_iterator(); ++it) {
        BucketDescriptor bucket;
        bucket.index = std::stoi((*it)[1].str());
        bucket.nbytes = std::stoll((*it)[2].str());
        bucket.checksum = (*it)[3].str();
        bucket.path = (*it)[4].str();
        req.buckets.push_back(bucket);
    }
    std::sort(req.buckets.begin(), req.buckets.end(), [](const auto& a, const auto& b) {
        return a.index < b.index;
    });
    return req;
}

std::string thread_level_name(int level) {
    if (level == MPI_THREAD_SINGLE) return "MPI_THREAD_SINGLE";
    if (level == MPI_THREAD_FUNNELED) return "MPI_THREAD_FUNNELED";
    if (level == MPI_THREAD_SERIALIZED) return "MPI_THREAD_SERIALIZED";
    if (level == MPI_THREAD_MULTIPLE) return "MPI_THREAD_MULTIPLE";
    return "unknown";
}

std::string rel_received_header(int generation, int rank) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "rank_00000/gen%06d/from_rank_%05d.header.json", generation, rank);
    return std::string(buf);
}

std::string rel_received_bucket(int generation, int rank, int bucket) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "rank_00000/gen%06d/from_rank_%05d.bucket%05d.bin", generation, rank, bucket);
    return std::string(buf);
}

std::string build_result_json(
    const Request& req,
    int world_rank,
    int world_size,
    int provided,
    const std::vector<PayloadPaths>& payloads,
    const std::vector<int>& timed_out,
    std::int64_t bytes_sent,
    std::int64_t bytes_received) {
    std::ostringstream out;
    out << "{";
    out << "\"schema_version\":1,";
    out << "\"transport\":\"compiled-cray-mpich-helper-p2p\",";
    out << "\"status\":\"" << (static_cast<int>(payloads.size()) >= req.quorum ? "advanced" : "deferred") << "\",";
    out << "\"rank\":" << world_rank << ",";
    out << "\"generation\":" << req.generation << ",";
    out << "\"base_generation\":" << req.base_generation << ",";
    out << "\"accepted_ranks\":[";
    for (std::size_t i = 0; i < payloads.size(); ++i) {
        if (i) out << ",";
        out << payloads[i].rank;
    }
    out << "],\"timed_out_ranks\":[";
    for (std::size_t i = 0; i < timed_out.size(); ++i) {
        if (i) out << ",";
        out << timed_out[i];
    }
    out << "],\"failed_ranks\":[],\"stale_ranks\":[],";
    out << "\"bytes_sent\":" << bytes_sent << ",";
    out << "\"bytes_received\":" << bytes_received << ",";
    out << "\"helper_exit_code\":0,";
    out << "\"mpi\":{\"provided_thread_level\":\"" << thread_level_name(provided)
        << "\",\"world_size\":" << world_size << ",\"root_rank\":0},";
    out << "\"received_payloads\":[";
    for (std::size_t i = 0; i < payloads.size(); ++i) {
        if (i) out << ",";
        out << "{\"rank\":" << payloads[i].rank
            << ",\"header_path\":\"" << json_escape(payloads[i].header_path)
            << "\",\"bucket_paths\":[";
        for (std::size_t j = 0; j < payloads[i].bucket_paths.size(); ++j) {
            if (j) out << ",";
            out << "\"" << json_escape(payloads[i].bucket_paths[j]) << "\"";
        }
        out << "]}";
    }
    out << "]}";
    return out.str();
}

PayloadPaths copy_local_payload_to_root(const fs::path& ipc_dir, const Request& req) {
    PayloadPaths paths;
    paths.rank = req.rank;
    paths.header_path = rel_received_header(req.generation, req.rank);
    write_text_atomic(ipc_dir / paths.header_path, read_text(ipc_dir / req.header_path));
    for (const auto& bucket : req.buckets) {
        std::string rel = rel_received_bucket(req.generation, req.rank, bucket.index);
        write_bytes_atomic(ipc_dir / rel, read_bytes(ipc_dir / bucket.path));
        paths.bucket_paths.push_back(rel);
    }
    return paths;
}

void send_result_string(int dest, const std::string& result) {
    long long n = static_cast<long long>(result.size());
    MPI_Send(&n, 1, MPI_LONG_LONG, dest, TAG_RESULT_SIZE, MPI_COMM_WORLD);
    MPI_Send(result.data(), static_cast<int>(result.size()), MPI_BYTE, dest, TAG_RESULT, MPI_COMM_WORLD);
}

std::string recv_result_string(int source) {
    long long n = 0;
    MPI_Recv(&n, 1, MPI_LONG_LONG, source, TAG_RESULT_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (n < 0 || n > (1LL << 30)) {
        throw std::runtime_error("invalid result size");
    }
    std::string result(static_cast<std::size_t>(n), '\0');
    MPI_Recv(result.data(), static_cast<int>(n), MPI_BYTE, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return result;
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
                  << "\"transport\":\"compiled-cray-mpich-helper-p2p\","
                  << "\"world_size\":" << size << ","
                  << "\"provided_thread_level\":\"" << thread_level_name(provided) << "\","
                  << "\"rank0_received_from\":" << recv_value << "}" << std::endl;
    }
}

}  // namespace

int main(int argc, char** argv) {
    fs::path ipc_dir;
    fs::path request_path;
    bool run_once = false;
    bool diagnostic = false;
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
        } else if (arg == "--help") {
            std::cout << "usage: compiled_mpich_dense_helper --diagnostic | --ipc-dir DIR --request PATH --run-once\n";
            return 0;
        }
    }

    int provided = -1;
    int rc = MPI_Init_thread(&argc, &argv, REQUIRED_THREAD_LEVEL, &provided);
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
            if (ipc_dir.empty() || request_path.empty()) {
                throw std::runtime_error("--ipc-dir and --request are required");
            }
            Request req = parse_request(request_path);
            if (req.world_size != size) {
                throw std::runtime_error("request world_size does not match MPI world size");
            }
            if (req.rank != rank) {
                throw std::runtime_error("request rank does not match MPI rank");
            }
            if (req.quorum <= 0 || req.quorum > req.world_size) {
                throw std::runtime_error("invalid quorum");
            }

            std::int64_t bytes_sent = req.payload_bytes;
            std::int64_t bytes_received = 0;
            std::vector<PayloadPaths> payloads;
            std::vector<int> timed_out;
            std::string result_json;

            if (rank == 0) {
                payloads.push_back(copy_local_payload_to_root(ipc_dir, req));
                bytes_received += req.payload_bytes;
                for (int peer = 1; peer < size; ++peer) {
                    long long preamble[5] = {0, 0, 0, 0, 0};
                    MPI_Recv(preamble, 5, MPI_LONG_LONG, peer, TAG_PREAMBLE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    int generation = static_cast<int>(preamble[0]);
                    int bucket_count = static_cast<int>(preamble[1]);
                    long long header_len = preamble[2];
                    long long payload_len = preamble[3];
                    int base_generation = static_cast<int>(preamble[4]);
                    if (generation != req.generation || base_generation != req.base_generation) {
                        timed_out.push_back(peer);
                        continue;
                    }
                    std::vector<char> header(static_cast<std::size_t>(header_len));
                    MPI_Recv(header.data(), static_cast<int>(header.size()), MPI_BYTE, peer, TAG_HEADER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    PayloadPaths paths;
                    paths.rank = peer;
                    paths.header_path = rel_received_header(req.generation, peer);
                    write_bytes_atomic(ipc_dir / paths.header_path, header);
                    for (int b = 0; b < bucket_count; ++b) {
                        long long bucket_len = 0;
                        MPI_Recv(&bucket_len, 1, MPI_LONG_LONG, peer, TAG_BUCKET_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        std::vector<char> bucket(static_cast<std::size_t>(bucket_len));
                        MPI_Recv(bucket.data(), static_cast<int>(bucket.size()), MPI_BYTE, peer, TAG_BUCKET_BASE + (b % 512), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        std::string rel = rel_received_bucket(req.generation, peer, b);
                        write_bytes_atomic(ipc_dir / rel, bucket);
                        paths.bucket_paths.push_back(rel);
                    }
                    bytes_received += payload_len;
                    payloads.push_back(paths);
                }
                for (int peer = 0; peer < size; ++peer) {
                    bool seen = std::any_of(payloads.begin(), payloads.end(), [peer](const PayloadPaths& p) {
                        return p.rank == peer;
                    });
                    if (!seen) {
                        timed_out.push_back(peer);
                    }
                }
                result_json = build_result_json(req, rank, size, provided, payloads, timed_out, bytes_sent, bytes_received);
                for (int peer = 1; peer < size; ++peer) {
                    send_result_string(peer, result_json);
                }
            } else {
                std::string header = read_text(ipc_dir / req.header_path);
                long long preamble[5] = {
                    req.generation,
                    static_cast<long long>(req.buckets.size()),
                    static_cast<long long>(header.size()),
                    req.payload_bytes,
                    req.base_generation,
                };
                MPI_Send(preamble, 5, MPI_LONG_LONG, 0, TAG_PREAMBLE, MPI_COMM_WORLD);
                MPI_Send(header.data(), static_cast<int>(header.size()), MPI_BYTE, 0, TAG_HEADER, MPI_COMM_WORLD);
                for (const auto& bucket_desc : req.buckets) {
                    std::vector<char> bucket = read_bytes(ipc_dir / bucket_desc.path);
                    long long bucket_len = static_cast<long long>(bucket.size());
                    MPI_Send(&bucket_len, 1, MPI_LONG_LONG, 0, TAG_BUCKET_SIZE, MPI_COMM_WORLD);
                    MPI_Send(bucket.data(), static_cast<int>(bucket.size()), MPI_BYTE, 0,
                             TAG_BUCKET_BASE + (bucket_desc.index % 512), MPI_COMM_WORLD);
                }
                result_json = recv_result_string(0);
            }

            fs::path result_path = request_path.parent_path();
            char name[64];
            std::snprintf(name, sizeof(name), "result.gen%06d.json", req.generation);
            result_path /= name;
            write_text_atomic(result_path, result_json + "\n");
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

    MPI_Finalize();
    return exit_code;
}
