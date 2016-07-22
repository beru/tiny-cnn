/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <vector>
#include "tiny_cnn/core/params/conv_params.h"
#include "tiny_cnn/core/kernels/avx_kernel_common.h"

#if _MSC_VER
#else
#define __assume(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#endif

namespace tiny_cnn {
namespace core {
namespace kernels {

// float ver
template <typename Allocator>
void accumulate_db(const index3d<cnn_size_t>&           param_out,
                   const std::vector<float, Allocator>& curr_delta,
                   std::vector<float, Allocator>&       db) {
    //fvec_t& db = *in_grad[2];
    if (param_out.width_ == 1 && param_out.height_ == 1) {
        size_t nblocks = param_out.depth_ / 8;
        size_t remainder = param_out.depth_ & 7;
        for (size_t i = 0; i < nblocks; ++i) {
            _mm256_storeu_ps(&db[i*8], _mm256_add_ps(_mm256_loadu_ps(&db[i*8]), _mm256_loadu_ps(&curr_delta[i*8])));
        }
        for (cnn_size_t outc = nblocks * 8; outc < param_out.depth_; outc++) {
            db[outc] += curr_delta[outc];
        }
    } else {
        auto area = param_out.area();
        size_t nblocks = area / 8;
        size_t remainder = area & 7;
        // prepare load-mask beforehand
        static const int32_t masks[] = {
            -1, -1, -1, -1,
            -1, -1, -1, -1,
            0, 0, 0, 0,
            0, 0, 0, 0,
        };
        __m256i mask = _mm256_loadu_si256((const __m256i*)(masks + 8 - remainder));
        for (cnn_size_t outc = 0; outc < param_out.depth_; outc++) {
            const float *delta = &curr_delta[param_out.get_index(0, 0, outc)];
            __m256 sum = _mm256_setzero_ps();
            for (size_t i=0; i<nblocks; ++i) {
                sum = _mm256_add_ps(sum, _mm256_loadu_ps(delta + i*8));
            }
            __m256 sum1 = _mm256_loadu_ps(delta + nblocks*8);
            sum1 = _mm256_and_ps(sum1, _mm256_castsi256_ps(mask));
            sum = _mm256_add_ps(sum, sum1);
            db[outc] += _mm_cvtss_f32(hsum256_ps(sum));
        }
    }
}

inline void accumulate_db(const index3d<cnn_size_t>& param_out,
                          const vec_t&               curr_delta,
                          vec_t&                     db) {
    //vec_t& db = *in_grad[2];
    for (cnn_size_t outc = 0; outc < param_out.depth_; outc++) {
        cnn_size_t idx = param_out.get_index(0, 0, outc);
        const float_t * delta = &curr_delta[idx];
        const float_t * deltaa = delta + param_out.width_ *
                                         param_out.height_;
        db[outc] += std::accumulate(delta, deltaa, float_t(0));
    }
}

// float ver
template <typename Allocator>
void avx_conv2d_3x3_back_kernel(const conv_params& params,
                                const std::vector<float, Allocator>& prev_out,
                                const std::vector<float, Allocator>& W,
                                std::vector<float, Allocator>&       dW,
                                std::vector<float, Allocator>&       db,
                                std::vector<float, Allocator>&       curr_delta,
                                std::vector<float, Allocator>*       prev_delta) {
    assert(params.weight.height_ == 3 && params.weight.width_ == 3);
    
    auto& in        = params.in;
    auto& out       = params.out;
    auto& in_padded = params.in_padded;
    auto& tbl       = params.tbl;
    auto  w_stride  = params.w_stride;
    const size_t in_padded_area = in_padded.area();
    float* pdelta_dst_org = &(*prev_delta)[0];
    const size_t  h_stride2 = params.h_stride * in_padded.width_;
    static const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0));
    // propagate delta to previous layer
    if (w_stride == 1 && out.width_ >= 4) {
        const cnn_size_t nblocks = out.width_ / 4;
        for (size_t inc = 0; inc < in.depth_; ++inc, pdelta_dst_org += in_padded_area) {
            for (cnn_size_t outc = 0; outc < out.depth_; outc++) {
                if (!tbl.is_connected(outc, inc)) continue;
                const float* pw = &W[9 * (in.depth_ * outc + inc)];
                const float* pdelta_src = &curr_delta[out.get_index(0, 0, outc)];
                float* pdelta_dst = pdelta_dst_org;
                __m128 w0a = _mm_blend_ps(_mm128_loadu_ps(pw+0), _mm_setzero_ps(), 0x07 /* 0b00000111 */);
                __m128 w1a = _mm_blend_ps(_mm128_loadu_ps(pw+3), _mm_setzero_ps(), 0x07 /* 0b00000111 */);
                __m128 w2a = _mm_blend_ps(_mm128_loadu_ps(pw+6), _mm_setzero_ps(), 0x07 /* 0b00000111 */);
                __m256 w0b = leftShift<4>(w0a);
                __m256 w1b = leftShift<4>(w1a);
                __m256 w2b = leftShift<4>(w2a);
                __m256 w0c = leftShift<8>(w0a);
                __m256 w1c = leftShift<8>(w1a);
                __m256 w2c = leftShift<8>(w2a);
                __m256 w0d = leftShift<12>(w0a);
                __m256 w1d = leftShift<12>(w1a);
                __m256 w2d = leftShift<12>(w2a);
                for (cnn_size_t y = 0; y < out.height_; y++) {
                    float* delta_dst0 = pdelta_dst;
                    float* delta_dst1 = &pdelta_dst[in_padded.width_ * 1];
                    float* delta_dst2 = &pdelta_dst[in_padded.width_ * 2];
                    if (nblocks > 0) {
                        __m256 delta_src = _mm256_broadcast_ps((const __m128*)pdelta_src);
                        for (cnn_size_t n = 0; n < nblocks; ++n) {
                            __m256 delta_src0 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(0, 0, 0, 0));
                            __m256 delta_src1 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(1, 1, 1, 1));
                            __m256 next_delta_src = _mm256_broadcast_ps((const __m128*)(pdelta_src + 4 * n + 4));
                            __m256 tmp0 = _mm256_mul_ps(w0a, delta_src0);
                            __m256 tmp1 = _mm256_mul_ps(w1a, delta_src0);
                            __m256 tmp2 = _mm256_mul_ps(w2a, delta_src0);
                            __m256 delta_src2 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(2, 2, 2, 2));
                            tmp0 = madd(w0b, delta_src1, tmp0);
                            tmp1 = madd(w1b, delta_src1, tmp1);
                            tmp2 = madd(w2b, delta_src1, tmp2);
                            __m256 delta_src3 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(3, 3, 3, 3));
                            tmp0 = madd(w0c, delta_src2, tmp0);
                            tmp1 = madd(w1c, delta_src2, tmp1);
                            tmp2 = madd(w2c, delta_src2, tmp2);
                            tmp0 = madd(w0d, delta_src3, tmp0);
                            tmp1 = madd(w1d, delta_src3, tmp1);
                            tmp2 = madd(w2d, delta_src3, tmp2);
                            tmp0 = _mm256_add_ps(tmp0, _mm256_loadu_ps(delta_dst0 + 4 * n));
                            tmp1 = _mm256_add_ps(tmp1, _mm256_loadu_ps(delta_dst1 + 4 * n));
                            tmp2 = _mm256_add_ps(tmp2, _mm256_loadu_ps(delta_dst2 + 4 * n));
                            _mm256_storeu_ps(delta_dst0 + 4 * n, tmp0);
                            _mm256_storeu_ps(delta_dst1 + 4 * n, tmp1);
                            _mm256_storeu_ps(delta_dst2 + 4 * n, tmp2);
                            delta_src = next_delta_src;
                        }
                    }
                    cnn_size_t x = nblocks * 4;
                    if (x < out.width_) {
                        __m256 delta_src = _mm256_broadcast_ss(pdelta_src + x);
                        do {
                            __m256 next_delta_src = _mm256_broadcast_ss(pdelta_src + x + 1);
                            __m256 tmp0 = _mm256_mul_ps(w0a, delta_src);
                            __m256 tmp1 = _mm256_mul_ps(w1a, delta_src);
                            __m256 tmp2 = _mm256_mul_ps(w2a, delta_src);
                            tmp0 = _mm256_add_ps(tmp0, _mm256_loadu_ps(delta_dst0 + x));
                            tmp1 = _mm256_add_ps(tmp1, _mm256_loadu_ps(delta_dst1 + x));
                            tmp2 = _mm256_add_ps(tmp2, _mm256_loadu_ps(delta_dst2 + x));
                            _mm256_storeu_ps(delta_dst0 + x, tmp0);
                            _mm256_storeu_ps(delta_dst1 + x, tmp1);
                            _mm256_storeu_ps(delta_dst2 + x, tmp2);
                            delta_src = next_delta_src;
                            ++x;
                        } while (x < out.width_);
                    }
                    pdelta_src += out.width_;
                    pdelta_dst += h_stride2;
                }
            }
        }
    } else if (out.height_ == 1 && out.width_ == 1) {
        for (size_t inc = 0; inc < in.depth_; ++inc, pdelta_dst_org += in_padded_area) {
            __m256 sum0 = _mm256_setzero_ps();
            __m128 sum1 = _mm_setzero_ps();
            
            size_t widx = 9 * inc;
            size_t wstep = 9 * in.depth_;
            const __m256 mask2 = mask;
            if (tbl.is_empty()) {
                for (cnn_size_t outc = 0; outc < out.depth_; outc++, widx+=wstep) {
                    __m256 delta_src = _mm256_broadcast_ss(&curr_delta[outc]);
                    const float* pw = (const float*)&W[widx];
                    __m256 w0 = _mm256_loadu_ps(pw+0);
                    __m128 w1 = _mm_load_ss(pw+8);
                    sum0 = madd(w0, delta_src, sum0);
                    sum1 = madd_ss(w1, _mm256_castps256_ps128(delta_src), sum1);
                }
            } else {
                for (cnn_size_t outc = 0; outc < out.depth_; outc++, widx += wstep) {
                    if (!tbl.is_connected(outc, inc)) {
                        continue;
                    }
                    __m256 delta_src = _mm256_broadcast_ss(&curr_delta[outc]);
                    const float* pw = (const float*)&W[widx];
                    __m256 w0 = _mm256_loadu_ps(pw+0);
                    __m128 w1 = _mm_load_ss(pw+8);
                    sum0 = madd(w0, delta_src, sum0);
                    sum1 = madd_ss(w1, _mm256_castps256_ps128(delta_src), sum1);
                }
            }

            // *FROM
            // 2211 1000
            //      ---2
            //
            // *TO
            //      -000
            //      -111
            //      -222
            __m128 new_sum0 = _mm_blend_ps(
                _mm_setzero_ps(),
                _mm256_castps256_ps128(sum0),
                0x07 /* 0b00000111 */
            );
            __m128 new_sum1 = _mm_blend_ps(
                _mm_setzero_ps(),
                _mm256_castps256_ps128(rightShift<12>(sum0)),
                0x07 /* 0b00000111 */
            );
            __m128 new_sum2 = _mm_blend_ps(
                _mm_setzero_ps(),
                _mm_or_ps(
                    _mm256_castps256_ps128(rightShift<24>(sum0)),
                    leftShift<8>(sum1)
                ),
                0x07 /* 0b00000111 */
            );
            float* delta_dst0 = pdelta_dst_org;
            float* delta_dst1 = &pdelta_dst_org[in_padded.width_ * 1];
            float* delta_dst2 = &pdelta_dst_org[in_padded.width_ * 2];
            __m128 dst0 = _mm_add_ps(_mm_loadu_ps(delta_dst0), new_sum0);
            __m128 dst1 = _mm_add_ps(_mm_loadu_ps(delta_dst1), new_sum1);
            __m128 dst2 = _mm_add_ps(_mm_loadu_ps(delta_dst2), new_sum2);
            _mm_storeu_ps(delta_dst0, dst0);
            _mm_storeu_ps(delta_dst1, dst1);
            _mm_storeu_ps(delta_dst2, dst2);
        } // for
    } else {
        for (size_t inc = 0; inc < in.depth_; ++inc, pdelta_dst_org += in_padded_area) {
            for (cnn_size_t outc = 0; outc < out.depth_; outc++) {
                if (!tbl.is_connected(outc, inc)) continue;

                const float* pw = &W[9 * (in.depth_ * outc + inc)];
                const float* pdelta_src = &curr_delta[out.get_index(0, 0, outc)];
                float* pdelta_dst = pdelta_dst_org;
                __m128 w0a = _mm_and_ps(_mm_loadu_ps(pw+0), mask);
                __m128 w1a = _mm_and_ps(_mm_loadu_ps(pw+3), mask);
                __m128 w2a = _mm_and_ps(_mm_loadu_ps(pw+6), mask);
                for (cnn_size_t y = 0; y < out.height_; y++) {
                    float* delta_dst0 = pdelta_dst;
                    float* delta_dst1 = &pdelta_dst[in_padded.width_ * 1];
                    float* delta_dst2 = &pdelta_dst[in_padded.width_ * 2];
                    for (cnn_size_t x = 0; x < out.width_; x++) {
                        __m128 delta_src = _mm_broadcast_ss(pdelta_src + x);
                        __m128 dst0 = _mm_loadu_ps(delta_dst0);
                        __m128 dst1 = _mm_loadu_ps(delta_dst1);
                        __m128 dst2 = _mm_loadu_ps(delta_dst2);
                        dst0 = madd(w0a, delta_src, dst0);
                        dst1 = madd(w1a, delta_src, dst1);
                        dst2 = madd(w2a, delta_src, dst2);
                        _mm_storeu_ps(delta_dst0, dst0);
                        _mm_storeu_ps(delta_dst1, dst1);
                        _mm_storeu_ps(delta_dst2, dst2);
                        delta_dst0 += w_stride;
                        delta_dst1 += w_stride;
                        delta_dst2 += w_stride;
                    } // for x
                    pdelta_src += out.width_;
                    pdelta_dst += h_stride2;
                } // for y
            } // for outc
        } // for inc
    }

    // accumulate dw
    if (out.width_ == 1 && out.height_ == 1) {
        const float* pprev_out = &prev_out[0];
        for (size_t inc = 0; inc < in.depth_; ++inc, pprev_out += in_padded_area) {
            VECTORIZE_ALIGN(32) float floats[16];
            size_t in_padded_width = in_padded.width_;
            _mm256_store_ps(&floats[0], _mm256_loadu_ps(pprev_out + in_padded_width * 0));
            _mm256_storeu_ps(&floats[3], _mm256_loadu_ps(pprev_out + in_padded_width * 1));
            _mm256_storeu_ps(&floats[6], _mm256_loadu_ps(pprev_out + in_padded_width * 2));
            __m256 prevos0 = _mm256_load_ps(&floats[0]);
            __m128 prevos1 = _mm_load_ss(&floats[8]);
            cnn_size_t widx = 9 * inc;
            cnn_size_t widx_delta = 9 * in.depth_;
            float* pdW = &dW[widx];
            const float* pcurr_delta = &curr_delta[0];
            __m256 delta = _mm256_broadcast_ss(pcurr_delta);
            for (cnn_size_t outc = 0; outc < out.depth_; outc++, pdW += widx_delta) {
                __m256 next_delta = _mm256_broadcast_ss(pcurr_delta + outc + 1);
                if (tbl.is_connected(outc, inc)) {
                    __m256 w0 = _mm256_loadu_ps(pdW+0);
                    __m128 w1 = _mm_load_ss(pdW+8);
                    w0 = madd(prevos0, delta, w0);
                    w1 = madd_ss(prevos3, _mm256_castps256_ps128(delta), w1);
                    _mm256_storeu_ps(pdW+0, w0);
                    _mm_store_ss(pdW+8, w1);
                }
                delta = next_delta;
            }
        }
    } else {
        const size_t nblocks = out.width_ / 8;
        const size_t remainder = out.width_ & 7;
        // prepare load-mask beforehand
        static const int32_t masks[] = {
            -1, -1, -1, -1,
            -1, -1, -1, -1,
            0, 0, 0, 0,
            0, 0, 0, 0,
        };
        __m256i mask = _mm256_loadu_si256((const __m256i*)(masks + 8 - remainder));
        auto& weight = params.weight;
        for (size_t inc = 0; inc < in.depth_; ++inc) {
            for (cnn_size_t outc = 0; outc < out.depth_; outc++) {

                if (!tbl.is_connected(outc, inc)) continue;
                const float* delta = &curr_delta[out.get_index(0, 0, outc)];

                cnn_size_t widx = weight.get_index(0, 0, in.depth_ * outc + inc);
                for (cnn_size_t wy = 0; wy < 3 /* weight.height_ */; wy++) {
                    for (cnn_size_t wx = 0; wx < 3 /* weight.width_ */; wx++) {
                        const float* prevo = &prev_out[in_padded.get_index(wx, wy, inc)];
                        __m256 sum0 = _mm256_setzero_ps();
                        __m256 sum1 = _mm256_setzero_ps();
                        const float* pa = prevo;
                        const float* pb = delta;
                        for (cnn_size_t y = 0; y < out.height_; y++) {
                            // vectorize::dot
                            for (size_t i=0; i<nblocks; ++i) {
                                __m256 a = _mm256_loadu_ps(pa+8*i);
                                __m256 b = _mm256_loadu_ps(pb+8*i);
                                sum0 = madd(a, b, sum0);
                            }
                            if (remainder) {
                                __m256 a = _mm256_loadu_ps(pa+8*nblocks);
                                __m256 b = _mm256_loadu_ps(pb+8*nblocks);
                                sum1 = madd(a, b, sum1);
                            }
                            pa += in_padded.width_;
                            pb += out.width_;
                        }
                        sum1 = _mm256_and_ps(sum1, _mm256_castsi256_ps(mask));
                        __m256 sum = _mm256_add_ps(sum0, sum1);
                        _mm_store_ss(&dW[widx], _mm_add_ps(_mm_load_ss(&dW[widx]), hsum256_ps(sum)));
                        ++widx;
                    }
                }
            }
        }
    }

    // accumulate db
    if (params.has_bias) {
        accumulate_db(params.out, curr_delta, db);
    }
} // avx_conv2d_3x3_back_kernel float ver

// float ver
template <typename Allocator>
void avx_conv2d_5x5_back_kernel(const conv_params& params,
                                const std::vector<float, Allocator>& prev_out,
                                const std::vector<float, Allocator>& W,
                                std::vector<float, Allocator>&       dW,
                                std::vector<float, Allocator>&       db,
                                std::vector<float, Allocator>&       curr_delta,
                                std::vector<float, Allocator>*       prev_delta) {
    assert(params.weight.height_ == 5 && params.weight.width_ == 5);
    auto& in        = params.in;
    auto& out       = params.out;
    auto& in_padded = params.in_padded;
    auto& tbl       = params.tbl;
    auto  w_stride  = params.w_stride;
    const size_t in_padded_area = in_padded.area();
    float* pdelta_dst_org = &(*prev_delta)[0];
    const size_t  h_stride2 = params.h_stride * in_padded.width_;
    static const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0));
    // propagate delta to previous layer
    if (w_stride == 1 && out.width_ >= 4) {
/*
s 4
    0123 0123
    
    0000 0000
    1111 1111
    2222 2222
    3333 3333
    
w 5*5 4
    01234---
    01234---
    01234---
    01234---
    01234---

    -01234--
    -01234--
    -01234--
    -01234--
    -01234--
    
    --01234-
    --01234-
    --01234-
    --01234-
    --01234-
    
    ---01234
    ---01234
    ---01234
    ---01234
    ---01234
    
d 8*5
    0123 4567
    0123 4567
    0123 4567
    0123 4567
    0123 4567
*/
        const cnn_size_t nblocks = out.width_ / 4;
        const cnn_size_t remainder = out.width_ & 3;
        for (size_t inc = 0; inc < in.depth_; ++inc, pdelta_dst_org += in_padded_area) {
            for (cnn_size_t outc = 0; outc < out.depth_; outc++) {
                if (!tbl.is_connected(outc, inc)) continue;
                const float* pw = &W[25 * (in.depth_ * outc + inc)];
                const float* pdelta_src = &curr_delta[out.get_index(0, 0, outc)];
                float* pdelta_dst = pdelta_dst_org;
                __m256 w0a = _mm256_and_ps(_mm256_loadu_ps(pw+0), mask);
                __m256 w1a = _mm256_and_ps(_mm256_loadu_ps(pw+5), mask);
                __m256 w2a = _mm256_and_ps(_mm256_loadu_ps(pw+10), mask);
                __m256 w3a = _mm256_and_ps(_mm256_loadu_ps(pw+15), mask);
                __m256 w4a = _mm256_and_ps(_mm256_loadu_ps(pw+20), mask);
                __m256 w0b = leftShift<4>(w0a);
                __m256 w1b = leftShift<4>(w1a);
                __m256 w2b = leftShift<4>(w2a);
                __m256 w3b = leftShift<4>(w3a);
                __m256 w4b = leftShift<4>(w4a);
                __m256 w0c = leftShift<8>(w0a);
                __m256 w1c = leftShift<8>(w1a);
                __m256 w2c = leftShift<8>(w2a);
                __m256 w3c = leftShift<8>(w3a);
                __m256 w4c = leftShift<8>(w4a);
                __m256 w0d = leftShift<12>(w0a);
                __m256 w1d = leftShift<12>(w1a);
                __m256 w2d = leftShift<12>(w2a);
                __m256 w3d = leftShift<12>(w3a);
                __m256 w4d = leftShift<12>(w4a);
                for (cnn_size_t y = 0; y < out.height_; y++) {
                    float* delta_dst0 = pdelta_dst;
                    float* delta_dst1 = &pdelta_dst[in_padded.width_ * 1];
                    float* delta_dst2 = &pdelta_dst[in_padded.width_ * 2];
                    float* delta_dst3 = &pdelta_dst[in_padded.width_ * 3];
                    float* delta_dst4 = &pdelta_dst[in_padded.width_ * 4];
                    __m256 delta_src = _mm256_broadcast_ps((const __m128*)pdelta_src);
                    if (nblocks > 0) {
                        for (cnn_size_t n = 0; n < nblocks; ++n) {
                            __m256 delta_src0 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(0, 0, 0, 0));
                            __m256 delta_src1 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(1, 1, 1, 1));
                            __m256 next_delta_src = _mm256_broadcast_ps((const __m128*)(pdelta_src + 4 * n + 4));
                            __m256 tmp0 = _mm256_mul_ps(w0a, delta_src0);
                            __m256 tmp1 = _mm256_mul_ps(w1a, delta_src0);
                            __m256 tmp2 = _mm256_mul_ps(w2a, delta_src0);
                            __m256 tmp3 = _mm256_mul_ps(w3a, delta_src0);
                            __m256 tmp4 = _mm256_mul_ps(w4a, delta_src0);
                            __m256 delta_src2 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(2, 2, 2, 2));
                            tmp0 = madd(w0b, delta_src1, tmp0);
                            tmp1 = madd(w1b, delta_src1, tmp1);
                            tmp2 = madd(w2b, delta_src1, tmp2);
                            tmp3 = madd(w3b, delta_src1, tmp3);
                            tmp4 = madd(w4b, delta_src1, tmp4);
                            __m256 delta_src3 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(3, 3, 3, 3));
                            tmp0 = madd(w0c, delta_src2, tmp0);
                            tmp1 = madd(w1c, delta_src2, tmp1);
                            tmp2 = madd(w2c, delta_src2, tmp2);
                            tmp3 = madd(w3c, delta_src2, tmp3);
                            tmp4 = madd(w4c, delta_src2, tmp4);
                            tmp0 = madd(w0d, delta_src3, tmp0);
                            tmp1 = madd(w1d, delta_src3, tmp1);
                            tmp2 = madd(w2d, delta_src3, tmp2);
                            tmp3 = madd(w3d, delta_src3, tmp3);
                            tmp4 = madd(w4d, delta_src3, tmp4);
                            tmp0 = _mm256_add_ps(tmp0, _mm256_loadu_ps(delta_dst0 + 4 * n));
                            tmp1 = _mm256_add_ps(tmp1, _mm256_loadu_ps(delta_dst1 + 4 * n));
                            tmp2 = _mm256_add_ps(tmp2, _mm256_loadu_ps(delta_dst2 + 4 * n));
                            tmp3 = _mm256_add_ps(tmp3, _mm256_loadu_ps(delta_dst3 + 4 * n));
                            tmp4 = _mm256_add_ps(tmp4, _mm256_loadu_ps(delta_dst4 + 4 * n));
                            _mm256_storeu_ps(delta_dst0 + 4 * n, tmp0);
                            _mm256_storeu_ps(delta_dst1 + 4 * n, tmp1);
                            _mm256_storeu_ps(delta_dst2 + 4 * n, tmp2);
                            _mm256_storeu_ps(delta_dst3 + 4 * n, tmp3);
                            _mm256_storeu_ps(delta_dst4 + 4 * n, tmp4);
                            delta_src = next_delta_src;
                        }
                    }
                    if (remainder) {
                        __m256 delta_src0 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(0, 0, 0, 0));
                        __m256 delta_src1 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(1, 1, 1, 1));
                        __m256 delta_src2 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(2, 2, 2, 2));
                        __m256 tmp0 = _mm256_mul_ps(w0a, delta_src0);
                        __m256 tmp1 = _mm256_mul_ps(w1a, delta_src0);
                        __m256 tmp2 = _mm256_mul_ps(w2a, delta_src0);
                        __m256 tmp3 = _mm256_mul_ps(w3a, delta_src0);
                        __m256 tmp4 = _mm256_mul_ps(w4a, delta_src0);
                        switch (remainder) {
                        case 3:
                            tmp0 = madd(w0c, delta_src2, tmp0);
                            tmp1 = madd(w1c, delta_src2, tmp1);
                            tmp2 = madd(w2c, delta_src2, tmp2);
                            tmp3 = madd(w3c, delta_src2, tmp3);
                            tmp4 = madd(w4c, delta_src2, tmp4);
                        case 2:
                            tmp0 = madd(w0b, delta_src1, tmp0);
                            tmp1 = madd(w1b, delta_src1, tmp1);
                            tmp2 = madd(w2b, delta_src1, tmp2);
                            tmp3 = madd(w3b, delta_src1, tmp3);
                            tmp4 = madd(w4b, delta_src1, tmp4);
                        case 1:
                            tmp0 = _mm256_add_ps(tmp0, _mm256_loadu_ps(delta_dst0 + nblocks * 4));
                            tmp1 = _mm256_add_ps(tmp1, _mm256_loadu_ps(delta_dst1 + nblocks * 4));
                            tmp2 = _mm256_add_ps(tmp2, _mm256_loadu_ps(delta_dst2 + nblocks * 4));
                            tmp3 = _mm256_add_ps(tmp3, _mm256_loadu_ps(delta_dst3 + nblocks * 4));
                            tmp4 = _mm256_add_ps(tmp4, _mm256_loadu_ps(delta_dst4 + nblocks * 4));
                            _mm256_storeu_ps(delta_dst0 + nblocks * 4, tmp0);
                            _mm256_storeu_ps(delta_dst1 + nblocks * 4, tmp1);
                            _mm256_storeu_ps(delta_dst2 + nblocks * 4, tmp2);
                            _mm256_storeu_ps(delta_dst3 + nblocks * 4, tmp3);
                            _mm256_storeu_ps(delta_dst4 + nblocks * 4, tmp4);
                            break;
                        default:
                            __assume(0);
                        }
                    }
                    pdelta_src += out.width_;
                    pdelta_dst += h_stride2;
                }
            }
        }
    } else if (out.height_ == 1 && out.width_ == 1) {
        for (size_t inc = 0; inc < in.depth_; ++inc, pdelta_dst_org += in_padded_area) {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m128 sum3 = _mm_setzero_ps();

            size_t widx = 25 * inc;
            size_t wstep = 25 * in.depth_;
            const __m256 mask2 = mask;
            if (tbl.is_empty()) {
                for (cnn_size_t outc = 0; outc < out.depth_; outc++, widx+=wstep) {
                    __m256 delta_src = _mm256_broadcast_ss(&curr_delta[outc]);
                    const float* pw = (const float*)&W[widx];
                    __m256 w0 = _mm256_loadu_ps(pw+0);
                    __m256 w1 = _mm256_loadu_ps(pw+8);
                    __m256 w2 = _mm256_loadu_ps(pw+16);
                    __m128 w3 = _mm_load_ss(pw+24);
                    sum0 = madd(w0, delta_src, sum0);
                    sum1 = madd(w1, delta_src, sum1);
                    sum2 = madd(w2, delta_src, sum2);
                    sum3 = madd_ss(w3, _mm256_castps256_ps128(delta_src), sum3);
                }
            } else {
                for (cnn_size_t outc = 0; outc < out.depth_; outc++, widx += wstep) {
                    if (!tbl.is_connected(outc, inc)) {
                        continue;
                    }
                    __m256 delta_src = _mm256_broadcast_ss(&curr_delta[outc]);
                    const float* pw = (const float*)&W[widx];
                    __m256 w0 = _mm256_loadu_ps(pw+0);
                    __m256 w1 = _mm256_loadu_ps(pw+8);
                    __m256 w2 = _mm256_loadu_ps(pw+16);
                    __m128 w3 = _mm_load_ss(pw+24);
                    sum0 = madd(w0, delta_src, sum0);
                    sum1 = madd(w1, delta_src, sum1);
                    sum2 = madd(w2, delta_src, sum2);
                    sum3 = madd_ss(w3, _mm256_castps256_ps128(delta_src), sum3);
                }
            }

            float* delta_dst0 = pdelta_dst_org;
            float* delta_dst1 = &pdelta_dst_org[in_padded.width_ * 1];
            float* delta_dst2 = &pdelta_dst_org[in_padded.width_ * 2];
            float* delta_dst3 = &pdelta_dst_org[in_padded.width_ * 3];
            float* delta_dst4 = &pdelta_dst_org[in_padded.width_ * 4];
            __m256 dst0 = _mm256_loadu_ps(delta_dst0);
            __m256 dst1 = _mm256_loadu_ps(delta_dst1);
            __m256 dst2 = _mm256_loadu_ps(delta_dst2);
            __m256 dst3 = _mm256_loadu_ps(delta_dst3);
            __m256 dst4 = _mm256_loadu_ps(delta_dst4);

            // *FROM
            // 1110 0000
            // 3222 2211
            // 4444 3333
            // ---- ---4
            //
            // *TO
            // ---0 0000
            // ---1 1111
            // ---2 2222
            // ---3 3333
            // ---4 4444
            __m256 new_sum0 = _mm256_blend_ps(
                _mm256_setzero_ps(),
                sum0,
                0x1F /* 0b00011111 */
            );
            __m256 new_sum1 = _mm256_blend_ps(
                _mm256_setzero_ps(),
                _mm256_or_ps(
                    rightShift<20>(sum0),
                    leftShift<12>(sum1)
                ),
                0x1F /* 0b00011111 */
            );
            __m256 new_sum2 = _mm256_blend_ps(
                _mm256_setzero_ps(),
                rightShift<8>(sum1),
                0x1F /* 0b00011111 */
            );
            __m256 new_sum3 = _mm256_blend_ps(
                _mm256_setzero_ps(),
                _mm256_or_ps(
                    rightShift<28>(sum1),
                    leftShift<4>(sum2)
                ),
                0x1F /* 0b00011111 */
            );
            __m256 new_sum4 = _mm256_blend_ps(
                _mm256_setzero_ps(),
                _mm256_set_m128(
                    sum3,
                    _mm256_extractf128_ps(sum2, 1)
                ),
                0x1F /* 0b00011111 */
            );
            dst0 = _mm256_add_ps(dst0, new_sum0);
            dst1 = _mm256_add_ps(dst1, new_sum1);
            dst2 = _mm256_add_ps(dst2, new_sum2);
            dst3 = _mm256_add_ps(dst3, new_sum3);
            dst4 = _mm256_add_ps(dst4, new_sum4);

            _mm256_storeu_ps(delta_dst0, dst0);
            _mm256_storeu_ps(delta_dst1, dst1);
            _mm256_storeu_ps(delta_dst2, dst2);
            _mm256_storeu_ps(delta_dst3, dst3);
            _mm256_storeu_ps(delta_dst4, dst4);
        } // for
    } else {
/*
s 1
    0000 0000
    
w 5*5
    01234---
    01234---
    01234---
    01234---

d 5*5
    0123 4567
    0123 4567
    0123 4567
    0123 4567
    0123 4567
*/
        for (size_t inc = 0; inc < in.depth_; ++inc, pdelta_dst_org += in_padded_area) {
            for (cnn_size_t outc = 0; outc < out.depth_; outc++) {
                if (!tbl.is_connected(outc, inc)) continue;

                const float* pw = &W[25 * (in.depth_ * outc + inc)];
                const float* pdelta_src = &curr_delta[out.get_index(0, 0, outc)];
                float* pdelta_dst = pdelta_dst_org;
                __m256 w0a = _mm256_and_ps(_mm256_loadu_ps(pw+0), mask);
                __m256 w1a = _mm256_and_ps(_mm256_loadu_ps(pw+5), mask);
                __m256 w2a = _mm256_and_ps(_mm256_loadu_ps(pw+10), mask);
                __m256 w3a = _mm256_and_ps(_mm256_loadu_ps(pw+15), mask);
                __m256 w4a = _mm256_and_ps(_mm256_loadu_ps(pw+20), mask);
                for (cnn_size_t y = 0; y < out.height_; y++) {
                    float* delta_dst0 = pdelta_dst;
                    float* delta_dst1 = &pdelta_dst[in_padded.width_ * 1];
                    float* delta_dst2 = &pdelta_dst[in_padded.width_ * 2];
                    float* delta_dst3 = &pdelta_dst[in_padded.width_ * 3];
                    float* delta_dst4 = &pdelta_dst[in_padded.width_ * 4];
                    for (cnn_size_t x = 0; x < out.width_; x++) {
                        __m256 delta_src = _mm256_broadcast_ss(pdelta_src + x);
                        __m256 dst0 = _mm256_loadu_ps(delta_dst0);
                        __m256 dst1 = _mm256_loadu_ps(delta_dst1);
                        __m256 dst2 = _mm256_loadu_ps(delta_dst2);
                        __m256 dst3 = _mm256_loadu_ps(delta_dst3);
                        __m256 dst4 = _mm256_loadu_ps(delta_dst4);
                        dst0 = madd(w0a, delta_src, dst0);
                        dst1 = madd(w1a, delta_src, dst1);
                        dst2 = madd(w2a, delta_src, dst2);
                        dst3 = madd(w3a, delta_src, dst3);
                        dst4 = madd(w4a, delta_src, dst4);
                        _mm256_storeu_ps(delta_dst0, dst0);
                        _mm256_storeu_ps(delta_dst1, dst1);
                        _mm256_storeu_ps(delta_dst2, dst2);
                        _mm256_storeu_ps(delta_dst3, dst3);
                        _mm256_storeu_ps(delta_dst4, dst4);
                        delta_dst0 += w_stride;
                        delta_dst1 += w_stride;
                        delta_dst2 += w_stride;
                        delta_dst3 += w_stride;
                        delta_dst4 += w_stride;
                    } // for x
                    pdelta_src += out.width_;
                    pdelta_dst += h_stride2;
                } // for y
            } // for outc
        } // for inc
    }

    // prepare load-mask beforehand
    static const int32_t masks[] = {
        -1, -1, -1, -1,
        -1, -1, -1, -1,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    // accumulate dw
    if (out.width_ == 1 && out.height_ == 1) {
        const float* pprev_out = &prev_out[0];
        for (size_t inc = 0; inc < in.depth_; ++inc, pprev_out += in_padded_area) {
            VECTORIZE_ALIGN(32) float floats[28];
            size_t in_padded_width = in_padded.width_;
            _mm256_store_ps(&floats[0], _mm256_loadu_ps(pprev_out + in_padded_width * 0));
            _mm256_storeu_ps(&floats[5], _mm256_loadu_ps(pprev_out + in_padded_width * 1));
            _mm256_storeu_ps(&floats[10], _mm256_loadu_ps(pprev_out + in_padded_width * 2));
            _mm256_storeu_ps(&floats[15], _mm256_loadu_ps(pprev_out + in_padded_width * 3));
            _mm256_storeu_ps(&floats[20], _mm256_loadu_ps(pprev_out + in_padded_width * 4));
            __m256 prevos0 = _mm256_load_ps(&floats[0]);
            __m256 prevos1 = _mm256_load_ps(&floats[8]);
            __m256 prevos2 = _mm256_load_ps(&floats[16]);
            __m128 prevos3 = _mm_load_ss(&floats[24]);
            cnn_size_t widx = 25 * inc;
            cnn_size_t widx_delta = 25 * in.depth_;
            float* pdW = &dW[widx];
            const float* pcurr_delta = &curr_delta[0];
            __m256 delta = _mm256_broadcast_ss(pcurr_delta);
            for (cnn_size_t outc = 0; outc < out.depth_; outc++, pdW += widx_delta) {
                __m256 next_delta = _mm256_broadcast_ss(pcurr_delta + outc + 1);
                if (tbl.is_connected(outc, inc)) {
                    __m256 w0 = _mm256_loadu_ps(pdW+0);
                    __m256 w1 = _mm256_loadu_ps(pdW+8);
                    __m256 w2 = _mm256_loadu_ps(pdW+16);
                    __m128 w3 = _mm_load_ss(pdW+24);
                    w0 = madd(prevos0, delta, w0);
                    w1 = madd(prevos1, delta, w1);
                    w2 = madd(prevos2, delta, w2);
                    w3 = madd_ss(prevos3, _mm256_castps256_ps128(delta), w3);
                    _mm256_storeu_ps(pdW+0, w0);
                    _mm256_storeu_ps(pdW+8, w1);
                    _mm256_storeu_ps(pdW+16, w2);
                    _mm_store_ss(pdW+24, w3);
                }
                delta = next_delta;
            }
        }
    } else {
        const size_t nblocks = out.width_ / 8;
        const size_t remainder = out.width_ & 7;
        __m256i mask = _mm256_loadu_si256((const __m256i*)(masks + 8 - remainder));
        auto& weight = params.weight;
        for (size_t inc = 0; inc < in.depth_; ++inc) {
            for (cnn_size_t outc = 0; outc < out.depth_; outc++) {

                if (!tbl.is_connected(outc, inc)) continue;
                const float* delta = &curr_delta[out.get_index(0, 0, outc)];

                cnn_size_t widx = weight.get_index(0, 0, in.depth_ * outc + inc);
                for (cnn_size_t wy = 0; wy < 5 /* weight.height_ */; wy++) {
                    for (cnn_size_t wx = 0; wx < 5 /* weight.width_ */; wx++) {
                        const float* prevo = &prev_out[in_padded.get_index(wx, wy, inc)];
                        __m256 sum0 = _mm256_setzero_ps();
                        __m256 sum1 = _mm256_setzero_ps();
                        const float* pa = prevo;
                        const float* pb = delta;
                        for (cnn_size_t y = 0; y < out.height_; y++) {
                            // vectorize::dot
                            for (size_t i=0; i<nblocks; ++i) {
                                __m256 a = _mm256_loadu_ps(pa+8*i);
                                __m256 b = _mm256_loadu_ps(pb+8*i);
                                sum0 = madd(a, b, sum0);
                            }
                            if (remainder) {
                                __m256 a = _mm256_loadu_ps(pa+8*nblocks);
                                __m256 b = _mm256_loadu_ps(pb+8*nblocks);
                                sum1 = madd(a, b, sum1);
                            }
                            pa += in_padded.width_;
                            pb += out.width_;
                        }
                        sum1 = _mm256_and_ps(sum1, _mm256_castsi256_ps(mask));
                        __m256 sum = _mm256_add_ps(sum0, sum1);
                        _mm_store_ss(&dW[widx], _mm_add_ps(_mm_load_ss(&dW[widx]), hsum256_ps(sum)));
                        ++widx;
                    }
                }
            }
        }
    }

    // accumulate db
    if (params.has_bias) {
        accumulate_db(params.out, curr_delta, db);
    }
} // avx_conv2d_5x5_back_kernel float ver

// double ver
template <typename Allocator>
void avx_conv2d_5x5_back_kernel(const conv_params& params,
                                const std::vector<double, Allocator>& prev_out,
                                const std::vector<double, Allocator>& W,
                                std::vector<double, Allocator>&       dW,
                                std::vector<double, Allocator>&       db,
                                std::vector<double, Allocator>&       curr_delta,
                                std::vector<double, Allocator>*       prev_delta) {
    assert(params.weight.height_ == 5 && params.weight.width_ == 5);
    // propagate delta to previous layer
    for_i(params.in.depth_, [&](int inc) {
        for (cnn_size_t outc = 0; outc < params.out.depth_; outc++) {
            if (!params.tbl.is_connected(outc, inc)) continue;

            cnn_size_t idx = 0;
            idx = params.in.depth_ * outc + inc;
            idx = params.weight.get_index(0, 0, idx);
            const float_t *pw = &W[idx];

            idx = params.out.get_index(0, 0, outc);
            const float_t *pdelta_src = &curr_delta[idx];

            idx = params.in_padded.get_index(0, 0, inc);
            float_t *pdelta_dst = &(*prev_delta)[idx];

            for (cnn_size_t y = 0; y < params.out.height_; y++) {
                for (cnn_size_t x = 0; x < params.out.width_; x++) {
                    const float_t * ppw = pw;

                    idx = y * params.out.width_ + x;
                    const float_t ppdelta_src = pdelta_src[idx];

                    float_t * ppdelta_dst = pdelta_dst +
                          y * params.h_stride * params.in_padded.width_ +
                          x * params.w_stride;

                    for (cnn_size_t wy = 0; wy < 5 /* params.weight.height_ */; wy++) {    // NOLINT
                        for (cnn_size_t wx = 0; wx < 5 /* params.weight.width_ */; wx++) { // NOLINT
                            idx = wy * params.in_padded.width_ + wx;
                            ppdelta_dst[idx] += *ppw++ * ppdelta_src;
                        }
                    }
                }
            }
        }
    });

    // accumulate dw
    for_i(params.in.depth_, [&](int inc) {
        for (cnn_size_t outc = 0; outc < params.out.depth_; outc++) {
            if (!params.tbl.is_connected(outc, inc)) continue;

            for (cnn_size_t wy = 0; wy < params.weight.height_; wy++) {
                for (cnn_size_t wx = 0; wx < params.weight.width_; wx++) {
                    float_t dst = float_t(0);

                    cnn_size_t idx = 0;
                    idx = params.in_padded.get_index(wx, wy, inc);
                    const float_t * prevo = &prev_out[idx];

                    idx = params.out.get_index(0, 0, outc);
                    const float_t * delta = &curr_delta[idx];

                    for (cnn_size_t y = 0; y < params.out.height_; y++) {
                        dst += vectorize::dot(
                            prevo + y * params.in_padded.width_,
                            delta + y * params.out.width_,
                            params.out.width_);
                    }

                    idx = params.in.depth_ * outc + inc;
                    dW[params.weight.get_index(wx, wy, idx)] += dst;
                }
            }
        }
    });

    // accumulate db
    if (params.has_bias) {
        accumulate_db(params.out, curr_delta, db);
    }
} // avx_conv2d_5x5_back_kernel double ver

inline void avx_conv2d_back_kernel(const conv_params& params,
                                   const vec_t& prev_out,
                                   const vec_t& W,
                                   vec_t&       dW,
                                   vec_t&       db,
                                   vec_t&       curr_delta,
                                   vec_t*       prev_delta) {
    if (params.weight.height_ == 5 && params.weight.width_ == 5) {
        avx_conv2d_5x5_back_kernel(params, prev_out, W, dW, db, curr_delta, prev_delta);
        return;
    }
    
    // propagate delta to previous layer
    for_i(params.in.depth_, [&](int inc) {
        for (cnn_size_t outc = 0; outc < params.out.depth_; outc++) {
            if (!params.tbl.is_connected(outc, inc)) continue;

            cnn_size_t idx = 0;
            idx = params.in.depth_ * outc + inc;
            idx = params.weight.get_index(0, 0, idx);
            const float_t *pw = &W[idx];

            idx = params.out.get_index(0, 0, outc);
            const float_t *pdelta_src = &curr_delta[idx];

            idx = params.in_padded.get_index(0, 0, inc);
            float_t *pdelta_dst = &(*prev_delta)[idx];

            for (cnn_size_t y = 0; y < params.out.height_; y++) {
                for (cnn_size_t x = 0; x < params.out.width_; x++) {
                    const float_t * ppw = pw;

                    idx = y * params.out.width_ + x;
                    const float_t ppdelta_src = pdelta_src[idx];

                    float_t * ppdelta_dst = pdelta_dst +
                          y * params.h_stride * params.in_padded.width_ +
                          x * params.w_stride;

                    for (cnn_size_t wy = 0; wy < params.weight.height_; wy++) {    // NOLINT
                        for (cnn_size_t wx = 0; wx < params.weight.width_; wx++) { // NOLINT
                            idx = wy * params.in_padded.width_ + wx;
                            ppdelta_dst[idx] += *ppw++ * ppdelta_src;
                        }
                    }
                }
            }
        }
    });

    // accumulate dw
    for_i(params.in.depth_, [&](int inc) {
        for (cnn_size_t outc = 0; outc < params.out.depth_; outc++) {
            if (!params.tbl.is_connected(outc, inc)) continue;

            for (cnn_size_t wy = 0; wy < params.weight.height_; wy++) {
                for (cnn_size_t wx = 0; wx < params.weight.width_; wx++) {
                    float_t dst = float_t(0);

                    cnn_size_t idx = 0;
                    idx = params.in_padded.get_index(wx, wy, inc);
                    const float_t * prevo = &prev_out[idx];

                    idx = params.out.get_index(0, 0, outc);
                    const float_t * delta = &curr_delta[idx];

                    for (cnn_size_t y = 0; y < params.out.height_; y++) {
                        dst += vectorize::dot(
                            prevo + y * params.in_padded.width_,
                            delta + y * params.out.width_,
                            params.out.width_);
                    }

                    idx = params.in.depth_ * outc + inc;
                    dW[params.weight.get_index(wx, wy, idx)] += dst;
                }
            }
        }
    });

    // accumulate db
    if (params.has_bias) {
        accumulate_db(params.out, curr_delta, db);
    }
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_cnn
