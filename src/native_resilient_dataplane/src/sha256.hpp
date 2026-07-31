#ifndef EMENDER_NDP_SHA256_HPP
#define EMENDER_NDP_SHA256_HPP

#include <openssl/evp.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>

namespace emender_ndp {

/*
 * The service hashes multi-GiB immutable trainer and owner buffers on every
 * admission boundary, and hashes the exact binary64 numerator into the result
 * root.  OpenSSL selects the optimized SHA-256 implementation for the active
 * Frontier CPU; a scalar block loop makes those mandatory passes dominate the
 * fixed redistribution stage even after all dense reductions are parallel.
 */
class Sha256 {
public:
    Sha256() : context_(EVP_MD_CTX_new()) {
        if (context_ == nullptr) throw std::bad_alloc();
        reset();
    }

    ~Sha256() { EVP_MD_CTX_free(context_); }
    Sha256(const Sha256&) = delete;
    Sha256& operator=(const Sha256&) = delete;
    Sha256(Sha256&&) = delete;
    Sha256& operator=(Sha256&&) = delete;

    void reset() {
        if (EVP_DigestInit_ex(context_, EVP_sha256(), nullptr) != 1)
            throw std::runtime_error("OpenSSL SHA-256 initialization failed");
    }

    void update(const void* input, std::size_t bytes) {
        if (bytes != 0 && (input == nullptr
                || EVP_DigestUpdate(context_, input, bytes) != 1))
            throw std::runtime_error("OpenSSL SHA-256 update failed");
    }

    std::array<std::uint8_t, 32> finish() {
        std::array<std::uint8_t, 32> result{};
        unsigned int result_bytes = 0;
        if (EVP_DigestFinal_ex(context_, result.data(), &result_bytes) != 1
                || result_bytes != result.size())
            throw std::runtime_error("OpenSSL SHA-256 finalization failed");
        reset();
        return result;
    }

    static std::array<std::uint8_t, 32> digest(const void* data,
                                               std::size_t bytes) {
        Sha256 hash;
        hash.update(data, bytes);
        return hash.finish();
    }

private:
    EVP_MD_CTX* context_ = nullptr;
};

}  // namespace emender_ndp

#endif
