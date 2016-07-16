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

namespace tiny_cnn {
namespace core {
namespace kernels {

// float ver
template <typename Allocator>
void avx_conv2d_3x3_kernel(const conv_params& params,
                           const std::vector<float, Allocator>& in,
                           const std::vector<float, Allocator>& W,
                           const std::vector<float, Allocator>& bias,
                           std::vector<float, Allocator>&       a,
                           const bool layer_parallelize) {
    assert(params.weight.height_ == 3 && params.weight.width_ == 3 && params.weight.area() == 9);
    
    auto& out       = params.out;
    auto& in_padded = params.in_padded;
    auto& tbl       = params.tbl;
    auto  w_stride  = params.w_stride;
    
    const size_t out_area = out.area();
    cnn_size_t oidx = 0;
    float bias_scale = params.has_bias ? 1.0f : 0.0f;
    const size_t stride = params.h_stride * in_padded.width_;
    const size_t inarea = in_padded.area();

    static const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0));
    const __m128 y_bias_scale = _mm_set_ss(bias_scale);
    if (out.height_ == 1 && out.width_ == 1) {
        if (stride == 3) {
            const float* pw = (const float*) &W[0];
            for (size_t o=0; o<out.depth_; ++o) {
                __m256 sum0 = _mm256_setzero_ps();
                __m128 sum1 = _mm_setzero_ps();
                const float* pi = (const float*) &in[0];
                for (cnn_size_t inc = 0; inc < params.in.depth_; ++inc, pw += 9, pi += inarea) {
                    if (!tbl.is_connected(o, inc)) {
                        continue;
                    }
                    __m256 w0 = _mm256_loadu_ps(pw+0);
                    __m256 i0 = _mm256_loadu_ps(pi+0);
                    __m128 w1 = _mm_load_ss(pw+8);
                    __m128 i1 = _mm_load_ss(pi+8);
                    __m256 tmp0 = _mm256_mul_ps(w0, i0);
                    __m128 tmp1 = _mm_mul_ss(w1, i1);
                    sum0 = _mm256_add_ps(tmp0, sum0);
                    sum1 = _mm_add_ss(tmp1, sum1);
                }
                __m128 b = _mm_load_ss(&bias[o]);
                __m128 hsum = hsum256_ps(sum0);
                b = madd_ss(b, y_bias_scale, sum1);
                _mm_store_ss(&a[o], _mm_add_ss(hsum, b));
            }
        } else {
            for (size_t o = 0; o < out.depth_; ++o) {
                __m128 sum0 = _mm_setzero_ps();
                __m128 sum1 = _mm_setzero_ps();
                __m128 sum2 = _mm_setzero_ps();
                size_t widx = 9/* params.weight.area() */ * params.in.depth_ * o;
                size_t inidx = 0;
                for (cnn_size_t inc = 0; inc < params.in.depth_; ++inc, widx += 9, inidx += inarea) {
                    if (!tbl.is_connected(o, inc)) {
                        continue;
                    }
                    const float* pw = (const float*) &W[widx];
                    __m128 w0 = _mm_loadu_ps(pw + 0);
                    __m128 w1 = _mm_loadu_ps(pw + 3);
                    __m128 w2 = _mm_loadu_ps(pw + 6);
                    const float* pi = (const float*) &in[inidx];
                    __m128 i0 = _mm_loadu_ps(pi + 0 * stride);
                    __m128 i1 = _mm_loadu_ps(pi + 1 * stride);
                    __m128 i2 = _mm_loadu_ps(pi + 2 * stride);
                    sum0 = madd(w0, i0, sum0);
                    sum1 = madd(w1, i1, sum1);
                    sum2 = madd(w2, i2, sum2);
                }
                __m128 sum = _mm_add_ps(_mm_add_ps(sum0, sum1), sum2);
                __m128 b = _mm_load_ss(&bias[o]);
                __m128 hsum = hsum128_ps(sum);
                hsum = madd(b, y_bias_scale, hsum);
                _mm_store_ss(&a[o], hsum);
            }
        }
    } else {
        const size_t nblocks = out.width_ / 8;
        for (size_t o = 0; o < out.depth_; ++o, oidx += out_area) {
            float* pa = &a[oidx];
            // init to bias value
            float b = bias[o] * bias_scale;
            {
                size_t headSize = 0;
                __m256 b2 = _mm256_set1_ps(b);
                if (oidx & 7) {
                    headSize = 8 - (oidx & 7);
                    assert(headSize < out_area);
                    for (size_t i=0; i<headSize; ++i) {
                        _mm_store_ss(&pa[i], _mm256_castps256_ps128(b2));
                    }
                }
                size_t cnt = (out_area - headSize) / 16;
                float* pa2 = pa + headSize;
                for (size_t i=0; i<cnt; ++i) {
                    _mm256_store_ps(&pa2[i*16+0], b2);
                    _mm256_store_ps(&pa2[i*16+8], b2);
                }
                for (size_t i=headSize+cnt*16; i<out_area; ++i) {
                    pa[i] = b;
                }
            }
            for (cnn_size_t inc = 0; inc < params.in.depth_; ++inc) {
                if (!tbl.is_connected(o, inc)) continue;

                const float* pw = (const float*) &W[9 * (params.in.depth_ * o + inc)];
                const float* pi = (const float*) &in[in_padded.get_index(0, 0, inc)];
/*
== Weight
abc abc
 abc abc

== Input
01234567
01234567
== Input Shifted
23456789
23456789

== W * I
012 456
 123 567
234 678
 345 789

== vdpps
0   4
 1   5
  2   6
   3   7
*/
                const __m256 w0 = _mm256_broadcast_ps((const __m128*)(pw+0));
                const __m256 w1 = _mm256_broadcast_ps((const __m128*)(pw+3));
                const __m256 w2 = _mm256_broadcast_ps((const __m128*)(pw+6));
                const __m256 w0a = leftShift<4>(w0);
                const __m256 w1a = leftShift<4>(w1);
                const __m256 w2a = leftShift<4>(w2);
                const __m256 mask0 = _mm256_setr_ps(1,1,1,0,1,1,1,0);
                const __m256 mask1 = _mm256_setr_ps(0,1,1,1,0,1,1,1);
                float* ppa = pa;
                for (cnn_size_t y = 0; y < out.height_; y++) {
                    const float* pi0 = (pi + y * stride);
                    const float* pi1 = pi0 + 1 * stride;
                    const float* pi2 = pi0 + 2 * stride;
                    cnn_size_t x = 0;
                    if (w_stride == 1) {
                        __m256 dst04, dst15, dst26, dst37;
                        float* ppa2 = ppa;
                        for (size_t i = 0; i < nblocks; ++i) {
                            __m256 i0 = _mm256_loadu_ps(pi0);
                            __m256 i1 = _mm256_loadu_ps(pi1);
                            __m256 i2 = _mm256_loadu_ps(pi2);
                            __m256 i0a = _mm256_loadu_ps(pi0+2);
                            __m256 i1a = _mm256_loadu_ps(pi0+2);
                            __m256 i2a = _mm256_loadu_ps(pi0+2);
                            __m256 sum = _mm256_loadu_ps(ppa2);
                            dst04 = _mm256_mul_ps(w0, i0);
                            dst04 = madd(w1, i1, dst04);
                            dst04 = madd(w2, i2, dst04);
                            dst15 = _mm256_mul_ps(w0a, i0);
                            dst15 = madd(w1a, i1, dst15);
                            dst15 = madd(w2a, i2, dst15);
                            dst26 = _mm256_mul_ps(w0, i0a);
                            dst26 = madd(w1, i1a, dst26);
                            dst26 = madd(w2, i2a, dst26);
                            dst37 = _mm256_mul_ps(w0a, i0a);
                            dst37 = madd(w1a, i1a, dst37);
                            dst37 = madd(w2a, i2a, dst37);
                            dst04 = _mm256_dp_ps(dst04, mask0, 0x71 /* 0b01110001 */);
                            dst15 = _mm256_dp_ps(dst15, mask1, 0xE2 /* 0b11100010 */);
                            dst26 = _mm256_dp_ps(dst26, mask0, 0x74 /* 0b01110100 */);
                            dst37 = _mm256_dp_ps(dst37, mask1, 0xE8 /* 0b11101000 */);
                            sum = _mm256_add_ps(_mm256_add_ps(dst04, dst15), _mm256_add_ps(dst26, dst37));
                            _mm256_storeu_ps(ppa2, sum);
                            pi0 += 8;
                            pi1 += 8;
                            pi2 += 8;
                            ppa2 += 8;
                        }
                        x = nblocks * 8;
                    }
                    for (; x < out.width_; ++x) {
                        __m128 sum = _mm_load_ss(&ppa[x]);
                        __m256 i0 = _mm256_loadu_ps(pi0);
                        __m256 i1 = _mm256_loadu_ps(pi1);
                        __m256 i2 = _mm256_loadu_ps(pi2);
                        __m256 sum0 = _mm256_mul_ps(w0a, i0);
                        __m256 sum1 = _mm256_mul_ps(w1a, i1);
                        sum0 = madd(w2a, i2, sum0);
                        sum0 = _mm256_add_ps(sum0, sum1);
                        _mm_store_ss(&ppa[x], _mm_add_ss(sum, hsum256_ps(sum0)));
//                      printf("%d %d %d %f\n", inc, y, x, ppa[x]);
                        pi0 += w_stride;
                        pi1 += w_stride;
                        pi2 += w_stride;
                    } // x loop
                    ppa += out.width_;
                } // y loop
            } // in depth loop
        } // out depth loop
    } // else
}

// float ver
template <typename Allocator>
void avx_conv2d_5x5_kernel(const conv_params& params,
                           const std::vector<float, Allocator>& in,
                           const std::vector<float, Allocator>& W,
                           const std::vector<float, Allocator>& bias,
                           std::vector<float, Allocator>&       a,
                           const bool layer_parallelize) {
    assert(params.weight.height_ == 5 && params.weight.width_ == 5);
    
    auto& out       = params.out;
    auto& in_padded = params.in_padded;
    auto& tbl       = params.tbl;
    auto  w_stride  = params.w_stride;
    
    const size_t out_area = out.area();
    cnn_size_t oidx = 0;
    float bias_scale = params.has_bias ? 1.0f : 0.0f;
    const size_t stride = params.h_stride * in_padded.width_;
    const size_t inarea = in_padded.area();

    static const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0));
    const __m128 y_bias_scale = _mm_set_ss(bias_scale);
    if (out.height_ == 1 && out.width_ == 1) {
        if (stride == 5) {
            const float* pw = (const float*) &W[0];
            for (size_t o=0; o<out.depth_; ++o) {
                __m256 sum0 = _mm256_setzero_ps();
                __m256 sum1 = _mm256_setzero_ps();
                __m256 sum2 = _mm256_setzero_ps();
                __m128 sum3 = _mm_setzero_ps();
                const float* pi = (const float*) &in[0];
                for (cnn_size_t inc = 0; inc < params.in.depth_; ++inc, pw += 25, pi += inarea) {
                    if (!tbl.is_connected(o, inc)) {
                        continue;
                    }
                    __m256 w0 = _mm256_loadu_ps(pw+0);
                    __m256 w1 = _mm256_loadu_ps(pw+8);
                    __m256 w2 = _mm256_loadu_ps(pw+16);
                    __m256 i0 = _mm256_loadu_ps(pi+0);
                    __m256 i1 = _mm256_loadu_ps(pi+8);
                    __m256 i2 = _mm256_loadu_ps(pi+16);
                    __m128 w3 = _mm_load_ss(pw+24);
                    __m128 i3 = _mm_load_ss(pi+24);
                    __m256 tmp0 = _mm256_mul_ps(w0, i0);
                    __m256 tmp1 = _mm256_mul_ps(w1, i1);
                    __m256 tmp2 = _mm256_mul_ps(w2, i2);
                    __m128 tmp3 = _mm_mul_ps(w3, i3);
                    sum0 = _mm256_add_ps(tmp0, sum0);
                    sum1 = _mm256_add_ps(tmp1, sum1);
                    sum2 = _mm256_add_ps(tmp2, sum2);
                    sum3 = _mm_add_ps(tmp3, sum3);
                }
                __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), sum2);
                __m128 b = _mm_load_ss(&bias[o]);
                __m128 hsum = hsum256_ps(sum);
                b = madd_ss(b, y_bias_scale, sum3);
                _mm_store_ss(&a[o], _mm_add_ss(hsum, b));
            }
        } else {
            for (size_t o = 0; o < out.depth_; ++o) {
                __m256 sum = _mm256_setzero_ps();
                size_t widx = 25/* weight_.area() */ * params.in.depth_ * o;
                size_t inidx = 0;
                for (cnn_size_t inc = 0; inc < params.in.depth_; ++inc, widx += 25, inidx += inarea) {
                    if (!tbl.is_connected(o, inc)) {
                        continue;
                    }
                    const float* pw = (const float*) &W[widx];
                    __m256 w0 = _mm256_loadu_ps(pw+0);
                    __m256 w1 = _mm256_loadu_ps(pw+5);
                    __m256 w2 = _mm256_loadu_ps(pw+10);
                    __m256 w3 = _mm256_loadu_ps(pw+15);
                    __m256 w4 = _mm256_loadu_ps(pw+20);
                    w0 = _mm256_and_ps(w0, mask);
                    w1 = _mm256_and_ps(w1, mask);
                    w2 = _mm256_and_ps(w2, mask);
                    w3 = _mm256_and_ps(w3, mask);
                    w4 = _mm256_and_ps(w4, mask);
                    const float* pi = (const float*) &in[inidx];
                    __m256 i0 = _mm256_loadu_ps(pi + 0 * stride);
                    __m256 i1 = _mm256_loadu_ps(pi + 1 * stride);
                    __m256 i2 = _mm256_loadu_ps(pi + 2 * stride);
                    __m256 i3 = _mm256_loadu_ps(pi + 3 * stride);
                    __m256 i4 = _mm256_loadu_ps(pi + 4 * stride);
                    __m256 sum0 = madd(w0, i0, sum);
                    __m256 sum1 = _mm256_mul_ps(w1, i1);
                    sum0 = madd(w2, i2, sum0);
                    sum1 = madd(w3, i3, sum1);
                    sum0 = madd(w4, i4, sum0);
                    sum = _mm256_add_ps(sum0, sum1);
                }
                __m128 b = _mm_load_ss(&bias[o]);
                __m128 hsum = hsum256_ps(sum);
                hsum = madd(b, y_bias_scale, hsum);
                _mm_store_ss(&a[o], hsum);
            }
        }
    } else {
        const size_t nblocks = out.width_ / 4;
        for (size_t o = 0; o < out.depth_; ++o, oidx += out_area) {
            float* pa = &a[oidx];
            // init to bias value
            float b = bias[o] * bias_scale;
            {
                size_t headSize = 0;
                __m256 b2 = _mm256_set1_ps(b);
                if (oidx & 7) {
                    headSize = 8 - (oidx & 7);
                    assert(headSize < out_area);
                    for (size_t i=0; i<headSize; ++i) {
                        _mm_store_ss(&pa[i], _mm256_castps256_ps128(b2));
                    }
                }
                size_t cnt = (out_area - headSize) / 16;
                float* pa2 = pa + headSize;
                for (size_t i=0; i<cnt; ++i) {
                    _mm256_store_ps(&pa2[i*16+0], b2);
                    _mm256_store_ps(&pa2[i*16+8], b2);
                }
                for (size_t i=headSize+cnt*16; i<out_area; ++i) {
                    pa[i] = b;
                }
            }
            for (cnn_size_t inc = 0; inc < params.in.depth_; ++inc) {
                if (!tbl.is_connected(o, inc)) continue;

                const float* pw = (const float*) &W[25 * (params.in.depth_ * o + inc)];
                const float* pi = (const float*) &in[in_padded.get_index(0, 0, inc)];

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
                float* ppa = pa;
                for (cnn_size_t y = 0; y < out.height_; y++) {
                    const float* pi0 = (pi + y * stride);
                    const float* pi1 = pi0 + 1 * stride;
                    const float* pi2 = pi0 + 2 * stride;
                    const float* pi3 = pi0 + 3 * stride;
                    const float* pi4 = pi0 + 4 * stride;
                    cnn_size_t x = 0;
                    if (w_stride == 1) {
                        __m256 dst0, dst1, dst2, dst3;
                        float* ppa2 = ppa;
                        for (size_t i = 0; i < nblocks; ++i) {
                            __m256 i0 = _mm256_loadu_ps(pi0);
                            __m256 i1 = _mm256_loadu_ps(pi1);
                            __m256 i2 = _mm256_loadu_ps(pi2);
                            __m256 i3 = _mm256_loadu_ps(pi3);
                            __m256 i4 = _mm256_loadu_ps(pi4);
                            __m128 sum = _mm_loadu_ps(ppa2);
                            dst0 = _mm256_mul_ps(w0a, i0);
                            dst1 = _mm256_mul_ps(w0b, i0);
                            dst2 = _mm256_mul_ps(w0c, i0);
                            dst3 = _mm256_mul_ps(w0d, i0);
                            dst0 = madd(w1a, i1, dst0);
                            dst1 = madd(w1b, i1, dst1);
                            dst2 = madd(w1c, i1, dst2);
                            dst3 = madd(w1d, i1, dst3);
                            dst0 = madd(w2a, i2, dst0);
                            dst1 = madd(w2b, i2, dst1);
                            dst2 = madd(w2c, i2, dst2);
                            dst3 = madd(w2d, i2, dst3);
                            dst0 = madd(w3a, i3, dst0);
                            dst1 = madd(w3b, i3, dst1);
                            dst2 = madd(w3c, i3, dst2);
                            dst3 = madd(w3d, i3, dst3);
                            dst0 = madd(w4a, i4, dst0);
                            dst1 = madd(w4b, i4, dst1);
                            __m128 hsum01 = hsum2x256_ps(dst0, dst1);
                            dst2 = madd(w4c, i4, dst2);
                            dst3 = madd(w4d, i4, dst3);
                            __m128 hsum23 = hsum2x256_ps(dst2, dst3);
                            __m128 sum2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(hsum01), _mm_castps_pd(hsum23)));
                            sum = _mm_add_ps(sum, sum2);
                            _mm_storeu_ps(ppa2, sum);
                            pi0 += 4;
                            pi1 += 4;
                            pi2 += 4;
                            pi3 += 4;
                            pi4 += 4;
                            ppa2 += 4;
                        }
                        x = nblocks * 4;
                    }
                    for (; x < out.width_; ++x) {
                        __m128 sum = _mm_load_ss(&ppa[x]);
                        __m256 i0 = _mm256_loadu_ps(pi0);
                        __m256 i1 = _mm256_loadu_ps(pi1);
                        __m256 i2 = _mm256_loadu_ps(pi2);
                        __m256 i3 = _mm256_loadu_ps(pi3);
                        __m256 i4 = _mm256_loadu_ps(pi4);
                        __m256 sum0 = _mm256_mul_ps(w0a, i0);
                        __m256 sum1 = _mm256_mul_ps(w1a, i1);
                        sum0 = madd(w2a, i2, sum0);
                        sum1 = madd(w3a, i3, sum1);
                        sum0 = madd(w4a, i4, sum0);
                        sum0 = _mm256_add_ps(sum0, sum1);
                        _mm_store_ss(&ppa[x], _mm_add_ss(sum, hsum256_ps(sum0)));
//                      printf("%d %d %d %f\n", inc, y, x, ppa[x]);
                        pi0 += w_stride;
                        pi1 += w_stride;
                        pi2 += w_stride;
                        pi3 += w_stride;
                        pi4 += w_stride;
                    } // x loop
                    ppa += out.width_;
                } // y loop
            } // in depth loop
        } // out depth loop
    } // else
} // avx_conv2d_5x5_kernel float ver

// double ver
template <typename Allocator>
void avx_conv2d_5x5_kernel(const conv_params& params,
                           const std::vector<double, Allocator>& in,
                           const std::vector<double, Allocator>& W,
                           const std::vector<double, Allocator>& bias,
                           std::vector<double, Allocator>&       a,
                           const bool layer_parallelize) {
    assert(params.weight.height_ == 5 && params.weight.width_ == 5);
    
    auto& out       = params.out;
    auto& in_padded = params.in_padded;
    auto& tbl       = params.tbl;
    auto  w_stride  = params.w_stride;
    
    const size_t out_area = out.area();
    double bias_scale = params.has_bias ? 1.0 : 0.0;
    const __m128d y_bias_scale = _mm_set_sd(bias_scale);
    cnn_size_t oidx = 0;

    const double* pw = &W[0];
    const size_t in_stride = params.h_stride * in_padded.width_;
    const size_t in_padded_area = in_padded.area();
    if (out.height_ == 1 && out.width_ == 1) {
        if (in_stride == 5) {
            for (size_t o = 0; o < out.depth_; ++o) {
                __m256d sum0 = _mm256_setzero_pd();
                __m256d sum1 = _mm256_setzero_pd();
                __m256d sum2 = _mm256_setzero_pd();
                __m256d sum3 = _mm256_setzero_pd();
                __m256d sum4 = _mm256_setzero_pd();
                __m256d sum5 = _mm256_setzero_pd();
                __m128d sum6 = _mm_setzero_pd();
                size_t inidx = 0;
                for (cnn_size_t inc = 0; inc < params.in.depth_; ++inc, pw += 25, inidx += in_padded_area) {
                    if (!tbl.is_connected(o, inc)) {
                        continue;
                    }
                    __m256d w0 = _mm256_loadu_pd(pw+0);
                    __m256d w1 = _mm256_loadu_pd(pw+4);
                    __m256d w2 = _mm256_loadu_pd(pw+8);
                    __m256d w3 = _mm256_loadu_pd(pw+12);
                    __m256d w4 = _mm256_loadu_pd(pw+16);
                    __m256d w5 = _mm256_loadu_pd(pw+20);
                    __m128d w6 = _mm_load_sd(pw+24);
                    const double* pi = (const double*) &in[inidx];
                    __m256d i0 = _mm256_loadu_pd(pi+0);
                    __m256d i1 = _mm256_loadu_pd(pi+4);
                    __m256d i2 = _mm256_loadu_pd(pi+8);
                    __m256d i3 = _mm256_loadu_pd(pi+12);
                    __m256d i4 = _mm256_loadu_pd(pi+16);
                    __m256d i5 = _mm256_loadu_pd(pi+20);
                    __m128d i6 = _mm_load_sd(pi+24);
                    __m256d tmp0 = _mm256_mul_pd(w0, i0);
                    __m256d tmp1 = _mm256_mul_pd(w1, i1);
                    __m256d tmp2 = _mm256_mul_pd(w2, i2);
                    __m256d tmp3 = _mm256_mul_pd(w3, i3);
                    __m256d tmp4 = _mm256_mul_pd(w4, i4);
                    __m256d tmp5 = _mm256_mul_pd(w5, i5);
                    __m128d tmp6 = _mm_mul_pd(w6, i6);
                    sum0 = _mm256_add_pd(tmp0, sum0);
                    sum1 = _mm256_add_pd(tmp1, sum1);
                    sum2 = _mm256_add_pd(tmp2, sum2);
                    sum3 = _mm256_add_pd(tmp3, sum3);
                    sum4 = _mm256_add_pd(tmp4, sum4);
                    sum5 = _mm256_add_pd(tmp5, sum5);
                    sum6 = _mm_add_pd(tmp6, sum6);
                }
                sum0 = _mm256_add_pd(sum0, sum1);
                sum2 = _mm256_add_pd(sum2, sum3);
                sum4 = _mm256_add_pd(sum4, sum5);
                sum0 = _mm256_add_pd(sum0, sum2);
                __m256d sum = _mm256_add_pd(sum0, sum4);
                __m128d b = _mm_load_sd(&bias[o]);
                __m128d hsum = hsum256_pd(sum);
                b = madd_sd(b, y_bias_scale, sum6);
                _mm_store_sd(&a[o], _mm_add_sd(hsum, b));
            }
        } else {
            for (size_t o=0; o<out.depth_; ++o) {
                __m256d sum_a = _mm256_setzero_pd();
                __m128d sum_b = _mm_setzero_pd();
                size_t inidx = 0;
                for (cnn_size_t inc = 0; inc < params.in.depth_; ++inc, pw += 25, inidx += in_padded_area) {
                    if (!tbl.is_connected(o, inc)) {
                        continue;
                    }
                    __m256d w0a = _mm256_loadu_pd(pw+0);
                    __m128d w0b = _mm_load_sd(pw+4);
                    __m256d w1a = _mm256_loadu_pd(pw+5);
                    __m128d w1b = _mm_load_sd(pw+9);
                    __m256d w2a = _mm256_loadu_pd(pw+10);
                    __m128d w2b = _mm_load_sd(pw+14);
                    __m256d w3a = _mm256_loadu_pd(pw+15);
                    __m128d w3b = _mm_load_sd(pw+19);
                    __m256d w4a = _mm256_loadu_pd(pw+20);
                    __m128d w4b = _mm_load_sd(pw+24);
                    const double* pi = (const double*) &in[inidx];
                    __m256d i0a = _mm256_loadu_pd(pi + 0 * in_stride);
                    __m128d i0b = _mm_load_sd(pi + 0 * in_stride + 4);
                    __m256d i1a = _mm256_loadu_pd(pi + 1 * in_stride);
                    __m128d i1b = _mm_load_sd(pi + 1 * in_stride + 4);
                    __m256d i2a = _mm256_loadu_pd(pi + 2 * in_stride);
                    __m128d i2b = _mm_load_sd(pi + 2 * in_stride + 4);
                    __m256d i3a = _mm256_loadu_pd(pi + 3 * in_stride);
                    __m128d i3b = _mm_load_sd(pi + 3 * in_stride + 4);
                    __m256d i4a = _mm256_loadu_pd(pi + 4 * in_stride);
                    __m128d i4b = _mm_load_sd(pi + 4 * in_stride + 4);
                    sum_a = madd(w0a, i0a, sum_a);
                    sum_b = madd(w0b, i0b, sum_b);
                    sum_a = madd(w1a, i1a, sum_a);
                    sum_b = madd(w1b, i1b, sum_b);
                    sum_a = madd(w2a, i2a, sum_a);
                    sum_b = madd(w2b, i2b, sum_b);
                    sum_a = madd(w3a, i3a, sum_a);
                    sum_b = madd(w3b, i3b, sum_b);
                    sum_a = madd(w4a, i4a, sum_a);
                    sum_b = madd(w4b, i4b, sum_b);
                }
                __m128d b = _mm_load_sd(&bias[o]);
                __m128d hsum = hsum256_pd(sum_a);
                b = _mm_fmadd_sd(b, y_bias_scale, sum_b);
                _mm_store_sd(&a[o], _mm_add_sd(hsum, b));
            }
        }
    } else {
        for (cnn_size_t o = 0; o < out.depth_; ++o, oidx += out_area) {
            double* pa = &a[oidx];
            double b = bias[o] * bias_scale;
            {
                size_t headSize = 0;
                __m256d b2 = _mm256_set1_pd(b);
                if (oidx & 3) {
                    headSize = 4 - (oidx & 3);
                    assert(headSize < out_area);
                    for (size_t i = 0; i < headSize; ++i) {
                        _mm_store_sd(&pa[i], _mm256_castpd256_pd128(b2));
                    }
                }
                size_t cnt = (out_area - headSize) / 8;
                double* pa2 = pa + headSize;
                for (size_t i = 0; i < cnt; ++i) {
                    _mm256_store_pd(&pa2[i*8+0], b2);
                    _mm256_store_pd(&pa2[i*8+4], b2);
                }
                for (size_t i = headSize + cnt*8; i < out_area; ++i) {
                    _mm_store_sd(&pa[i], _mm256_castpd256_pd128(b2));
                }
            }
            const double* pi0 = &in[0];
            for (cnn_size_t inc = 0; inc < params.in.depth_; ++inc, pw += 25, pi0 += in_padded_area) {
                if (!tbl.is_connected(o, inc)) continue;
                __m256d w0a = _mm256_loadu_pd(pw+0);
                __m128d w0b = _mm_load_sd(pw+4);
                __m256d w1a = _mm256_loadu_pd(pw+5);
                __m128d w1b = _mm_load_sd(pw+9);
                __m256d w2a = _mm256_loadu_pd(pw+10);
                __m128d w2b = _mm_load_sd(pw+14);
                __m256d w3a = _mm256_loadu_pd(pw+15);
                __m128d w3b = _mm_load_sd(pw+19);
                __m256d w4a = _mm256_loadu_pd(pw+20);
                __m128d w4b = _mm_load_sd(pw+24);
                const double* pi = pi0;
                double* pa2 = pa;
                for (cnn_size_t y = 0; y < out.height_; ++y, pi += in_stride, pa2 += out.width_) {
                    const double* pi1 = pi + 1 * in_stride;
                    const double* pi2 = pi + 2 * in_stride;
                    const double* pi3 = pi + 3 * in_stride;
                    const double* pi4 = pi + 4 * in_stride;
                    for (cnn_size_t x = 0; x < out.width_; ++x) {
                        __m128d sum = _mm_load_sd(&pa2[x]);
                        __m256d i0a = _mm256_loadu_pd(pi);
                        __m128d i0b = _mm_load_sd(pi + 4);
                        __m256d i1a = _mm256_loadu_pd(pi1);
                        __m128d i1b = _mm_load_sd(pi1 + 4);
                        __m256d i2a = _mm256_loadu_pd(pi2);
                        __m128d i2b = _mm_load_sd(pi2 + 4);
                        __m256d i3a = _mm256_loadu_pd(pi3);
                        __m128d i3b = _mm_load_sd(pi3 + 4);
                        __m256d i4a = _mm256_loadu_pd(pi4);
                        __m128d i4b = _mm_load_sd(pi4 + 4);
                        __m256d sum_a = _mm256_mul_pd(w0a, i0a);
                        __m128d sum_b = _mm_mul_sd(w0b, i0b);
                        sum_a = madd(w1a, i1a, sum_a);
                        sum_b = madd(w1b, i1b, sum_b);
                        sum_a = madd(w2a, i2a, sum_a);
                        sum_b = madd(w2b, i2b, sum_b);
                        sum_a = madd(w3a, i3a, sum_a);
                        sum_b = madd(w3b, i3b, sum_b);
                        sum_a = madd(w4a, i4a, sum_a);
                        sum_b = madd(w4b, i4b, sum_b);
                        __m128d sum_c = hsum256_pd(sum_a);
                        sum = _mm_add_sd(sum, sum_b);
                        _mm_store_sd(&pa2[x], _mm_add_sd(sum, sum_c));
                        pi0 += w_stride;
                        pi1 += w_stride;
                        pi2 += w_stride;
                        pi3 += w_stride;
                        pi4 += w_stride;
                    } // x loop
                } // y loop
            } // in depth loop
        } // out depth loop
    } // else
} // avx_conv2d_5x5_kernel double ver

inline void avx_conv2d_kernel(const conv_params& params,
                              const vec_t& in,
                              const vec_t& W,
                              const vec_t& bias,
                              vec_t&       a,
                              const bool   layer_parallelize) {
    
    if (params.weight.height_ == 3 && params.weight.width_ == 3) {
        avx_conv2d_3x3_kernel(params, in, W, bias, a, layer_parallelize);
        return;
    }else if (params.weight.height_ == 5 && params.weight.width_ == 5) {
        avx_conv2d_5x5_kernel(params, in, W, bias, a, layer_parallelize);
        return;
    }
    
    for_i(layer_parallelize, params.out.depth_, [&](int o) {
        for (cnn_size_t inc = 0; inc < params.in.depth_; inc++) {
            if (!params.tbl.is_connected(o, inc)) continue;

            cnn_size_t idx = 0;
            idx = params.in.depth_ * o + inc;
            idx = params.weight.get_index(0, 0, idx);
            const float_t *pw = &W[idx];

            idx = params.in_padded.get_index(0, 0, inc);
            const float_t *pi = &in[idx];

            idx = params.out.get_index(0, 0, o);
            float_t *pa = &a[idx];

            for (cnn_size_t y = 0; y < params.out.height_; y++) {
                for (cnn_size_t x = 0; x < params.out.width_; x++) {
                    const float_t * ppw = pw;
                    const float_t * ppi = pi + params.in_padded.width_ *
                                        (y * params.h_stride) +
                                         x * params.w_stride;
                    float_t sum = float_t(0);

                    // should be optimized for small kernel(3x3,5x5)
                    for (cnn_size_t wy = 0; wy < params.weight.height_; wy++) {    // NOLINT
                        for (cnn_size_t wx = 0; wx < params.weight.width_; wx++) { // NOLINT
                            idx = wy * params.in_padded.width_ + wx;
                            sum += *ppw++ * ppi[idx];
                        }
                    }
                    pa[y * params.out.width_ + x] += sum;
                }
            }
        }

        if (params.has_bias) {
            float_t * pa  = &a[params.out.get_index(0, 0, o)];
            float_t * paa = pa + params.out.width_ * params.out.height_;
            std::for_each(pa, paa, [&](float_t& f) { f += bias[o]; });
        }
    });
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_cnn
