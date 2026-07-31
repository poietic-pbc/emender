// Standalone async DiLoCo delta transport benchmark for Frontier.
//
// Dense x/z delta bytes move only through MPI point-to-point buffers. Files are
// used for small JSON metrics and logs, not as the tensor update data plane.

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace {

constexpr std::uint64_t kMiB = 1024ull * 1024ull;

struct Options {
  std::string mode = "pair";
  std::string device = "cpu";
  std::uint64_t payload_mib = 16;
  std::uint64_t chunk_mib = 4;
  int iters = 20;
  int warmup = 5;
  int dtype_bytes = 2;
  std::string metrics_path;
};

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

std::string env_or_empty(const char* name) {
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string();
}

[[noreturn]] void usage_and_exit(const char* argv0, int code) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cerr
        << "usage: " << argv0
        << " [--mode pair|fanin] [--device cpu|hip] [--payload-mib N]"
        << " [--chunk-mib N] [--iters N] [--warmup N]"
        << " [--dtype-bytes N] [--metrics PATH]\n";
  }
  MPI_Finalize();
  std::exit(code);
}

Options parse_options(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return std::string(argv[++i]);
    };
    if (arg == "--help" || arg == "-h") {
      usage_and_exit(argv[0], 0);
    } else if (arg == "--mode") {
      opt.mode = require_value(arg);
    } else if (arg == "--device") {
      opt.device = require_value(arg);
    } else if (arg == "--payload-mib") {
      opt.payload_mib = std::stoull(require_value(arg));
    } else if (arg == "--chunk-mib") {
      opt.chunk_mib = std::stoull(require_value(arg));
    } else if (arg == "--iters") {
      opt.iters = std::stoi(require_value(arg));
    } else if (arg == "--warmup") {
      opt.warmup = std::stoi(require_value(arg));
    } else if (arg == "--dtype-bytes") {
      opt.dtype_bytes = std::stoi(require_value(arg));
    } else if (arg == "--metrics") {
      opt.metrics_path = require_value(arg);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (opt.mode != "pair" && opt.mode != "fanin") {
    throw std::runtime_error("--mode must be pair or fanin");
  }
  if (opt.device != "cpu" && opt.device != "hip") {
    throw std::runtime_error("--device must be cpu or hip");
  }
  if (opt.payload_mib == 0 || opt.chunk_mib == 0 || opt.iters <= 0 || opt.warmup < 0) {
    throw std::runtime_error("payload, chunk, iters, and warmup must be positive");
  }
  if (opt.dtype_bytes <= 0) {
    throw std::runtime_error("--dtype-bytes must be positive");
  }
  return opt;
}

class Buffer {
 public:
  Buffer(std::uint64_t bytes, const std::string& device) : bytes_(bytes), device_(device) {
    if (device_ == "cpu") {
      host_.resize(static_cast<std::size_t>(bytes_));
      std::fill(host_.begin(), host_.end(), 0x5a);
      ptr_ = host_.data();
    } else {
#if defined(USE_HIP)
      hipError_t err = hipMalloc(&ptr_, bytes_);
      if (err != hipSuccess) {
        throw std::runtime_error(std::string("hipMalloc failed: ") + hipGetErrorString(err));
      }
      err = hipMemset(ptr_, 0x5a, bytes_);
      if (err != hipSuccess) {
        throw std::runtime_error(std::string("hipMemset failed: ") + hipGetErrorString(err));
      }
#else
      throw std::runtime_error("benchmark was not compiled with USE_HIP");
#endif
    }
  }

  ~Buffer() {
#if defined(USE_HIP)
    if (device_ == "hip" && ptr_) {
      hipFree(ptr_);
    }
#endif
  }

  void* at(std::uint64_t offset) {
    return static_cast<void*>(static_cast<char*>(ptr_) + offset);
  }

  std::uint64_t bytes() const { return bytes_; }

 private:
  std::uint64_t bytes_ = 0;
  std::string device_;
  void* ptr_ = nullptr;
  std::vector<unsigned char> host_;
};

void local_single_rank_copy(Buffer& x, Buffer& z, std::uint64_t payload_bytes,
                            std::uint64_t chunk_bytes, const std::string& device) {
  for (std::uint64_t off = 0; off < payload_bytes; off += chunk_bytes) {
    const std::uint64_t n = std::min(chunk_bytes, payload_bytes - off);
    if (device == "cpu") {
      std::memcpy(x.at(off), z.at(off), static_cast<std::size_t>(n));
      std::memcpy(z.at(off), x.at(off), static_cast<std::size_t>(n));
    } else {
#if defined(USE_HIP)
      hipMemcpy(x.at(off), z.at(off), n, hipMemcpyDeviceToDevice);
      hipMemcpy(z.at(off), x.at(off), n, hipMemcpyDeviceToDevice);
#endif
    }
  }
}

void exchange_pair(Buffer& x, Buffer& z, std::uint64_t payload_bytes,
                   std::uint64_t chunk_bytes, int rank, int world) {
  if (world < 2) {
    local_single_rank_copy(x, z, payload_bytes, chunk_bytes, "cpu");
    return;
  }
  const int partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
  if (partner >= world) {
    return;
  }
  for (std::uint64_t off = 0; off < payload_bytes; off += chunk_bytes) {
    const int n = static_cast<int>(std::min(chunk_bytes, payload_bytes - off));
    MPI_Request reqs[4];
    MPI_Irecv(x.at(off), n, MPI_BYTE, partner, 100, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(z.at(off), n, MPI_BYTE, partner, 101, MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(x.at(off), n, MPI_BYTE, partner, 101, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(z.at(off), n, MPI_BYTE, partner, 100, MPI_COMM_WORLD, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
  }
}

void exchange_fanin(Buffer& x, Buffer& z, std::uint64_t payload_bytes,
                    std::uint64_t chunk_bytes, int rank, int world) {
  if (world < 2) {
    local_single_rank_copy(x, z, payload_bytes, chunk_bytes, "cpu");
    return;
  }
  for (std::uint64_t off = 0; off < payload_bytes; off += chunk_bytes) {
    const int n = static_cast<int>(std::min(chunk_bytes, payload_bytes - off));
    if (rank == 0) {
      for (int src = 1; src < world; ++src) {
        MPI_Request reqs[2];
        MPI_Irecv(x.at(off), n, MPI_BYTE, src, 200, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(z.at(off), n, MPI_BYTE, src, 201, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
      }
    } else {
      MPI_Request reqs[2];
      MPI_Isend(x.at(off), n, MPI_BYTE, 0, 200, MPI_COMM_WORLD, &reqs[0]);
      MPI_Isend(z.at(off), n, MPI_BYTE, 0, 201, MPI_COMM_WORLD, &reqs[1]);
      MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    }
  }
}

std::string join_strings(const std::vector<std::string>& values) {
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ", ";
    out << "\"" << json_escape(values[i]) << "\"";
  }
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  Options opt;
  try {
    opt = parse_options(argc, argv);
    if (opt.device == "hip") {
#if !defined(USE_HIP)
      throw std::runtime_error("--device hip requires a USE_HIP build");
#endif
    }
  } catch (const std::exception& exc) {
    if (rank == 0) {
      std::cerr << "argument error: " << exc.what() << "\n";
    }
    usage_and_exit(argv[0], 2);
  }

  const std::uint64_t payload_bytes = opt.payload_mib * kMiB;
  const std::uint64_t chunk_bytes = opt.chunk_mib * kMiB;
  const std::uint64_t xz_payload_bytes = payload_bytes * 2ull;
  std::vector<double> measured;
  std::string failure;

  try {
    Buffer x(payload_bytes, opt.device);
    Buffer z(payload_bytes, opt.device);

    for (int iter = -opt.warmup; iter < opt.iters; ++iter) {
      MPI_Barrier(MPI_COMM_WORLD);
      const double t0 = MPI_Wtime();
      if (world == 1) {
        local_single_rank_copy(x, z, payload_bytes, chunk_bytes, opt.device);
      } else if (opt.mode == "pair") {
        exchange_pair(x, z, payload_bytes, chunk_bytes, rank, world);
      } else {
        exchange_fanin(x, z, payload_bytes, chunk_bytes, rank, world);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      const double dt = MPI_Wtime() - t0;
      if (iter >= 0) {
        measured.push_back(dt);
      }
    }
  } catch (const std::exception& exc) {
    failure = exc.what();
  }

  const int failed = failure.empty() ? 0 : 1;
  int any_failed = 0;
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  const double sum = [&]() {
    double s = 0.0;
    for (double v : measured) s += v;
    return s;
  }();
  const double min_v = measured.empty() ? 0.0 : *std::min_element(measured.begin(), measured.end());
  const double max_v = measured.empty() ? 0.0 : *std::max_element(measured.begin(), measured.end());
  const double avg = measured.empty() ? 0.0 : sum / static_cast<double>(measured.size());
  const double effective_bytes =
      world == 1 ? static_cast<double>(xz_payload_bytes)
                 : (opt.mode == "pair"
                        ? static_cast<double>(xz_payload_bytes) * static_cast<double>((world / 2) * 2)
                        : static_cast<double>(xz_payload_bytes) * static_cast<double>(std::max(0, world - 1)));
  const double gib = effective_bytes / static_cast<double>(1024ull * 1024ull * 1024ull);
  const double bandwidth = avg > 0.0 ? gib / avg : 0.0;

  if (rank == 0) {
    std::vector<std::string> failure_modes;
    if (any_failed) {
      failure_modes.push_back(failure.empty() ? "non-root rank failed" : failure);
    }
    if (opt.device == "hip" && env_or_empty("MPICH_GPU_SUPPORT_ENABLED") != "1") {
      failure_modes.push_back("MPICH_GPU_SUPPORT_ENABLED is not 1; GPU-aware MPI may be disabled");
    }
    if (world == 1) {
      failure_modes.push_back("single-rank validation exercises allocation and chunk loop only; no network peer");
    }

    const std::string staging =
        opt.device == "hip" ? "hip_device_buffer_gpu_aware_mpi_required"
                            : "cpu_host_buffer_no_gpu_staging";
    const std::string status = any_failed ? "fail" : "pass";

    std::ostringstream json;
    json << std::fixed << std::setprecision(9);
    json << "{\n";
    json << "  \"benchmark\": \"async_diloco_transport_bench\",\n";
    json << "  \"status\": \"" << status << "\",\n";
    json << "  \"transport\": \"cray_mpich_point_to_point\",\n";
    json << "  \"merge_mechanism\": \"mpi_p2p_no_torch_distributed_collectives\",\n";
    json << "  \"dense_data_plane\": \"mpi_memory_buffers_no_lustre_payload_files\",\n";
    json << "  \"mode\": \"" << opt.mode << "\",\n";
    json << "  \"world_size\": " << world << ",\n";
    json << "  \"device\": \"" << opt.device << "\",\n";
    json << "  \"staging_behavior\": \"" << staging << "\",\n";
    json << "  \"payload_mib_per_state\": " << opt.payload_mib << ",\n";
    json << "  \"xz_payload_bytes_per_rank\": " << xz_payload_bytes << ",\n";
    json << "  \"chunk_mib\": " << opt.chunk_mib << ",\n";
    json << "  \"dtype_bytes\": " << opt.dtype_bytes << ",\n";
    json << "  \"iters\": " << opt.iters << ",\n";
    json << "  \"warmup\": " << opt.warmup << ",\n";
    json << "  \"avg_latency_s\": " << avg << ",\n";
    json << "  \"min_latency_s\": " << min_v << ",\n";
    json << "  \"max_latency_s\": " << max_v << ",\n";
    json << "  \"effective_bandwidth_gib_s\": " << bandwidth << ",\n";
    json << "  \"mpich_gpu_support_enabled\": \"" << json_escape(env_or_empty("MPICH_GPU_SUPPORT_ENABLED")) << "\",\n";
    json << "  \"fi_cxi_rx_match_mode\": \"" << json_escape(env_or_empty("FI_CXI_RX_MATCH_MODE")) << "\",\n";
    json << "  \"fi_mr_cache_monitor\": \"" << json_escape(env_or_empty("FI_MR_CACHE_MONITOR")) << "\",\n";
    json << "  \"failure_modes\": [" << join_strings(failure_modes) << "]\n";
    json << "}\n";

    std::cout << "ASYNC_DILOCO_TRANSPORT_METRICS " << json.str();
    if (!opt.metrics_path.empty()) {
      FILE* fp = std::fopen(opt.metrics_path.c_str(), "w");
      if (!fp) {
        std::cerr << "failed to open metrics path: " << opt.metrics_path << "\n";
        MPI_Abort(MPI_COMM_WORLD, 3);
      }
      const std::string payload = json.str();
      std::fwrite(payload.data(), 1, payload.size(), fp);
      std::fclose(fp);
    }
  }

  MPI_Finalize();
  return any_failed ? 1 : 0;
}
