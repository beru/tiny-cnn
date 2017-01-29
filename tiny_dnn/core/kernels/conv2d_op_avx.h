/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>
#include "tiny_dnn/core/kernels/conv2d_op_internal.h"
#include "tiny_dnn/core/params/conv_params.h"

#ifdef CNN_USE_AVX
#include "tiny_dnn/core/kernels/avx_kernel_common.h"
#endif

namespace tiny_dnn {
namespace kernels {

#ifdef CNN_USE_AVX

// float ver
template <typename Allocator>
void avx_conv2d_5x5_kernel(const core::conv_params &params,
                           const std::vector<float, Allocator> &in,
                           const std::vector<float, Allocator> &W,
                           const std::vector<float, Allocator> &bias,
                           std::vector<float, Allocator> &a,
                           const bool layer_parallelize) {
  CNN_UNREFERENCED_PARAMETER(layer_parallelize);
  assert(params.weight.height_ == 5 && params.weight.width_ == 5);

  auto &out       = params.out;
  auto &in_padded = params.in_padded;
  auto &tbl       = params.tbl;
  auto w_stride   = params.w_stride;

  const serial_size_t out_area = out.area();
  serial_size_t oidx           = 0;
  float bias_scale             = params.has_bias ? 1.0f : 0.0f;
  const serial_size_t stride   = params.h_stride * in_padded.width_;
  const serial_size_t inarea   = in_padded.area();

  static const __m256i imask = _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0);
  // static const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1,
  // -1, -1, -1, 0, 0, 0));

  const __m128 y_bias_scale = _mm_set_ss(bias_scale);
  if (out.height_ == 1 && out.width_ == 1) {
    const float *pw = (const float *)&W[0];
    for (serial_size_t o = 0; o < out.depth_; ++o) {
      __m256 sum0     = _mm256_setzero_ps();
      __m256 sum1     = _mm256_setzero_ps();
      __m256 sum2     = _mm256_setzero_ps();
      __m128 sum3     = _mm_setzero_ps();
      const float *pi = (const float *)&in[0];
      for (serial_size_t inc = 0; inc < params.in.depth_;
           ++inc, pw += 25, pi += inarea) {
        if (!tbl.is_connected(o, inc)) {
          continue;
        }
        __m256 w0   = _mm256_loadu_ps(pw + 0);
        __m256 w1   = _mm256_loadu_ps(pw + 8);
        __m256 w2   = _mm256_loadu_ps(pw + 16);
        __m256 i0   = _mm256_loadu_ps(pi + 0);
        __m256 i1   = _mm256_loadu_ps(pi + 8);
        __m256 i2   = _mm256_loadu_ps(pi + 16);
        __m128 w3   = _mm_load_ss(pw + 24);
        __m128 i3   = _mm_load_ss(pi + 24);
        __m256 tmp0 = _mm256_mul_ps(w0, i0);
        __m256 tmp1 = _mm256_mul_ps(w1, i1);
        __m256 tmp2 = _mm256_mul_ps(w2, i2);
        __m128 tmp3 = _mm_mul_ps(w3, i3);
        sum0        = _mm256_add_ps(tmp0, sum0);
        sum1        = _mm256_add_ps(tmp1, sum1);
        sum2        = _mm256_add_ps(tmp2, sum2);
        sum3        = _mm_add_ps(tmp3, sum3);
      }
      __m256 sum  = _mm256_add_ps(_mm256_add_ps(sum0, sum1), sum2);
      __m128 b    = _mm_load_ss(&bias[o]);
      __m128 hsum = hsum256_ps(sum);
      b           = madd128_ss(b, y_bias_scale, sum3);
      _mm_store_ss(&a[o], _mm_add_ss(hsum, b));
    }
  } else {
    const size_t n12 = out.width_ / 12;
    const size_t n4 = (out.width_ % 12) / 4;
    const size_t n1 = out.width_ % 4;
    //const serial_size_t n4 = out.width_ / 4;
    alignas(32) float w0[24], w1[24], w2[24], w3[24], w4[24];
    auto zero = _mm256_setzero_ps();
    for (serial_size_t o = 0; o < out.depth_; ++o, oidx += out_area) {
      float *pa = &a[oidx];
#if 1
      // init to bias value
      float b = bias[o] * bias_scale;
      {
        size_t headSize = 0;
        __m256 b2       = _mm256_set1_ps(b);
        if (oidx & 7) {
          headSize = 8 - (oidx & 7);
          assert(headSize < out_area);
          for (size_t i = 0; i < headSize; ++i) {
            _mm_store_ss(&pa[i], _mm256_castps256_ps128(b2));
          }
        }
        size_t cnt = (out_area - headSize) / 16;
        float *pa2 = pa + headSize;
        for (size_t i = 0; i < cnt; ++i) {
          _mm256_store_ps(&pa2[i * 16 + 0], b2);
          _mm256_store_ps(&pa2[i * 16 + 8], b2);
        }
        for (size_t i = headSize + cnt * 16; i < out_area; ++i) {
          pa[i] = b;
        }
      }
#endif
      for (serial_size_t inc = 0; inc < params.in.depth_; ++inc) {
        if (!tbl.is_connected(o, inc)) continue;

        int widx = 25 * (params.in.depth_ * o + inc);
        const float *pw = (const float *)&W[widx];
        assert(widx + 25 <= W.size());
        const float *pi = (const float *)&in[in_padded.get_index(0, 0, inc)];
        float *ppa = pa;
        if (w_stride == 1) {
          __m256 w0s = _mm256_loadu_ps(pw);
          __m256 w1s = _mm256_loadu_ps(pw + 5);
          __m256 w2s = _mm256_loadu_ps(pw + 10);
          __m256 w3s = _mm256_loadu_ps(pw + 15);
          __m256 w4s = _mm256_loadu_ps(pw + 20);
          _mm256_store_ps(w0 + 0, w0s);
          _mm256_storeu_ps(w0 + 5, w0s);
          _mm256_storeu_ps(w0 + 10, w0s);
          _mm256_storeu_ps(w0 + 15, w0s);
          _mm_store_ps(w0 + 20, _mm256_castps256_ps128(w0s));
          _mm256_store_ps(w1 + 0, w1s);
          _mm256_storeu_ps(w1 + 5, w1s);
          _mm256_storeu_ps(w1 + 10, w1s);
          _mm256_storeu_ps(w1 + 15, w1s);
          _mm_store_ps(w1 + 20, _mm256_castps256_ps128(w1s));
          _mm256_store_ps(w2 + 0, w2s);
          _mm256_storeu_ps(w2 + 5, w2s);
          _mm256_storeu_ps(w2 + 10, w2s);
          _mm256_storeu_ps(w2 + 15, w2s);
          _mm_store_ps(w2 + 20, _mm256_castps256_ps128(w2s));
          _mm256_store_ps(w3 + 0, w3s);
          _mm256_storeu_ps(w3 + 5, w3s);
          _mm256_storeu_ps(w3 + 10, w3s);
          _mm256_storeu_ps(w3 + 15, w3s);
          _mm_store_ps(w3 + 20, _mm256_castps256_ps128(w3s));
          _mm256_store_ps(w4 + 0, w4s);
          _mm256_storeu_ps(w4 + 5, w4s);
          _mm256_storeu_ps(w4 + 10, w4s);
          _mm256_storeu_ps(w4 + 15, w4s);
          _mm_store_ps(w4 + 20, _mm256_castps256_ps128(w4s));
          for (serial_size_t y = 0; y < out.height_; ++y, ppa += out.width_) {
            const float *pi0 = (pi + y * stride);
            const float *pi1 = pi0 + 1 * in_padded.width_;
            const float *pi2 = pi0 + 2 * in_padded.width_;
            const float *pi3 = pi0 + 3 * in_padded.width_;
            const float *pi4 = pi0 + 4 * in_padded.width_;
            for (size_t i = 0; i < n12; ++i) {
              __m256 i0 = _mm256_loadu_ps(pi0 + i * 12);
              __m256 i1 = _mm256_loadu_ps(pi1 + i * 12);
              __m256 i2 = _mm256_loadu_ps(pi2 + i * 12);
              __m256 i3 = _mm256_loadu_ps(pi3 + i * 12);
              __m256 i4 = _mm256_loadu_ps(pi4 + i * 12);
              static const int o00 = 0;
              static const int o01 = 4;
              static const int o02 = 8;
              static const int o03 = 12;
              static const int o04 = 16;
              __m256 s00 = _mm256_mul_ps(i0, _mm256_load_ps(w0 + o00));
              __m256 s01 = _mm256_mul_ps(i0, _mm256_loadu_ps(w0 + o01));
              __m256 s02 = _mm256_mul_ps(i0, _mm256_load_ps(w0 + o02));
              __m256 s03 = _mm256_mul_ps(i0, _mm256_loadu_ps(w0 + o03));
              __m256 s04 = _mm256_mul_ps(i0, _mm256_load_ps(w0 + o04));
              s00 = madd256_ps(i1, _mm256_load_ps (w1 + o00), s00);
              s01 = madd256_ps(i1, _mm256_loadu_ps(w1 + o01), s01);
              s02 = madd256_ps(i1, _mm256_load_ps (w1 + o02), s02);
              s03 = madd256_ps(i1, _mm256_loadu_ps(w1 + o03), s03);
              s04 = madd256_ps(i1, _mm256_load_ps (w1 + o04), s04);
              s00 = madd256_ps(i2, _mm256_load_ps (w2 + o00), s00);
              s01 = madd256_ps(i2, _mm256_loadu_ps(w2 + o01), s01);
              s02 = madd256_ps(i2, _mm256_load_ps (w2 + o02), s02);
              s03 = madd256_ps(i2, _mm256_loadu_ps(w2 + o03), s03);
              s04 = madd256_ps(i2, _mm256_load_ps (w2 + o04), s04);
              s00 = madd256_ps(i3, _mm256_load_ps (w3 + o00), s00);
              s01 = madd256_ps(i3, _mm256_loadu_ps(w3 + o01), s01);
              s02 = madd256_ps(i3, _mm256_load_ps (w3 + o02), s02);
              s03 = madd256_ps(i3, _mm256_loadu_ps(w3 + o03), s03);
              s04 = madd256_ps(i3, _mm256_load_ps (w3 + o04), s04);
              s00 = madd256_ps(i4, _mm256_load_ps (w4 + o00), s00);
              s01 = madd256_ps(i4, _mm256_loadu_ps(w4 + o01), s01);
              s02 = madd256_ps(i4, _mm256_load_ps (w4 + o02), s02);
              s03 = madd256_ps(i4, _mm256_loadu_ps(w4 + o03), s03);
              s04 = madd256_ps(i4, _mm256_load_ps (w4 + o04), s04);
              {
                __m256 v0 = _mm256_blend_ps(zero, s00, 0b00011111);
                __m256 v1 = _mm256_blend_ps(zero, s01, 0b00111110);
                __m256 v2 = _mm256_blend_ps(zero, s02, 0b01111100);
                __m256 v3 = _mm256_blend_ps(zero, s03, 0b11111000);
                __m128 sum      = _mm_loadu_ps(ppa + i * 12);
                __m128 hsum0123 = hsum4x256_ps(v0, v1, v2, v3);
                sum      = _mm_add_ps(sum, hsum0123);
                _mm_storeu_ps(ppa + i * 12, sum);
              }
              i0 = _mm256_loadu_ps(pi0 + i * 12 + 8);
              i1 = _mm256_loadu_ps(pi1 + i * 12 + 8);
              i2 = _mm256_loadu_ps(pi2 + i * 12 + 8);
              i3 = _mm256_loadu_ps(pi3 + i * 12 + 8);
              i4 = _mm256_loadu_ps(pi4 + i * 12 + 8);
              static const int o10 = 8;
              static const int o11 = 12;
              static const int o12 = 16;
              static const int o13 = 0;
              static const int o14 = 4;
              __m256 s10 = _mm256_mul_ps(i0, _mm256_load_ps (w0 + o10));
              __m256 s11 = _mm256_mul_ps(i0, _mm256_loadu_ps(w0 + o11));
              __m256 s12 = _mm256_mul_ps(i0, _mm256_load_ps (w0 + o12));
              __m256 s13 = _mm256_mul_ps(i0, _mm256_load_ps (w0 + o13));
              __m256 s14 = _mm256_mul_ps(i0, _mm256_loadu_ps(w0 + o14));
              s10 = madd256_ps(i1, _mm256_load_ps (w1 + o10), s10);
              s11 = madd256_ps(i1, _mm256_loadu_ps(w1 + o11), s11);
              s12 = madd256_ps(i1, _mm256_load_ps (w1 + o12), s12);
              s13 = madd256_ps(i1, _mm256_loadu_ps(w1 + o13), s13);
              s14 = madd256_ps(i1, _mm256_load_ps (w1 + o14), s14);
              s10 = madd256_ps(i2, _mm256_load_ps (w2 + o10), s10);
              s11 = madd256_ps(i2, _mm256_loadu_ps(w2 + o11), s11);
              s12 = madd256_ps(i2, _mm256_load_ps (w2 + o12), s12);
              s13 = madd256_ps(i2, _mm256_loadu_ps(w2 + o13), s13);
              s14 = madd256_ps(i2, _mm256_load_ps (w2 + o14), s14);
              s10 = madd256_ps(i3, _mm256_load_ps (w3 + o10), s10);
              s11 = madd256_ps(i3, _mm256_loadu_ps(w3 + o11), s11);
              s12 = madd256_ps(i3, _mm256_load_ps (w3 + o12), s12);
              s13 = madd256_ps(i3, _mm256_loadu_ps(w3 + o13), s13);
              s14 = madd256_ps(i3, _mm256_load_ps (w3 + o14), s14);
              s10 = madd256_ps(i4, _mm256_load_ps (w4 + o10), s10);
              s11 = madd256_ps(i4, _mm256_loadu_ps(w4 + o11), s11);
              s12 = madd256_ps(i4, _mm256_load_ps (w4 + o12), s12);
              s13 = madd256_ps(i4, _mm256_loadu_ps(w4 + o13), s13);
              s14 = madd256_ps(i4, _mm256_load_ps (w4 + o14), s14);
              {
                s04 = _mm256_blend_ps(zero, s04, 0b11110000);
                s00 = _mm256_blend_ps(zero, s00, 0b11100000);
                s01 = _mm256_blend_ps(zero, s01, 0b11000000);
                s02 = _mm256_blend_ps(zero, s02, 0b10000000);
                __m256 v0 = _mm256_blend_ps(s14, s04, 0b00000001);
                __m256 v1 = _mm256_blend_ps(s10, s00, 0b00000011);
                __m256 v2 = _mm256_blend_ps(s11, s01, 0b00000111);
                __m256 v3 = _mm256_blend_ps(s12, s02, 0b00001111);
                __m128 sum      = _mm_loadu_ps(ppa + i * 12 + 4);
                __m128 hsum0123 = hsum4x256_ps(v0, v1, v2, v3);
                sum      = _mm_add_ps(sum, hsum0123);
                _mm_storeu_ps(ppa + i * 12 + 4, sum);
                v0 = _mm256_blend_ps(zero, s13, 0b11100000);
                v1 = _mm256_blend_ps(zero, s14, 0b11000001);
                v2 = _mm256_blend_ps(zero, s10, 0b10000011);
                v3 = _mm256_blend_ps(zero, s11, 0b00000111);
                sum      = _mm_loadu_ps(ppa + i * 12 + 8);
                hsum0123 = hsum4x256_ps(v0, v1, v2, v3);
                sum      = _mm_add_ps(sum, hsum0123);
                _mm_storeu_ps(ppa + i * 12 + 8, sum);
              }
            }
            pi0 += n12 * 12;
            pi1 += n12 * 12;
            pi2 += n12 * 12;
            pi3 += n12 * 12;
            pi4 += n12 * 12;
#if 0
            for (size_t i = 0; i < n4; ++i) {
              __m256 i0       = _mm256_loadu_ps(pi0 + i * 4);
              __m256 i1       = _mm256_loadu_ps(pi1 + i * 4);
              __m256 i2       = _mm256_loadu_ps(pi2 + i * 4);
              __m256 i3       = _mm256_loadu_ps(pi3 + i * 4);
              __m256 i4       = _mm256_loadu_ps(pi4 + i * 4);
              __m256 dst0     = _mm256_mul_ps(w0a, i0);
              __m256 dst1     = _mm256_mul_ps(w0b, i0);
              __m256 dst2     = _mm256_mul_ps(w0c, i0);
              __m256 dst3     = _mm256_mul_ps(w0d, i0);
              dst0     = madd256_ps(w1a, i1, dst0);
              dst1     = madd256_ps(w1b, i1, dst1);
              dst2     = madd256_ps(w1c, i1, dst2);
              dst3     = madd256_ps(w1d, i1, dst3);
              dst0     = madd256_ps(w2a, i2, dst0);
              dst1     = madd256_ps(w2b, i2, dst1);
              dst2     = madd256_ps(w2c, i2, dst2);
              dst3     = madd256_ps(w2d, i2, dst3);
              dst0     = madd256_ps(w3a, i3, dst0);
              dst1     = madd256_ps(w3b, i3, dst1);
              dst2     = madd256_ps(w3c, i3, dst2);
              dst3     = madd256_ps(w3d, i3, dst3);
              dst0     = madd256_ps(w4a, i4, dst0);
              dst1     = madd256_ps(w4b, i4, dst1);
              dst2     = madd256_ps(w4c, i4, dst2);
              dst3     = madd256_ps(w4d, i4, dst3);
              __m128 sum      = _mm_loadu_ps(ppa + i * 4);
              __m128 hsum0123 = hsum4x256_ps(dst0, dst1, dst2, dst3);
              sum      = _mm_add_ps(sum, hsum0123);
              _mm_storeu_ps(ppa + i * 4, sum);
            }
            pi0 += n4 * 4;
            pi1 += n4 * 4;
            pi2 += n4 * 4;
            pi3 += n4 * 4;
            pi4 += n4 * 4;
            ppa += n4 * 4;
            for (size_t i = 0; i < n1; ++i) {
              __m128 sum         = _mm_load_ss(&ppa[i]);
              __m256 i0          = _mm256_loadu_ps(pi0 + i);
              __m256 i1          = _mm256_loadu_ps(pi1 + i);
              __m256 i2          = _mm256_loadu_ps(pi2 + i);
              __m256 i3          = _mm256_loadu_ps(pi3 + i);
              __m256 i4          = _mm256_loadu_ps(pi4 + i);
              __m256 sum0 = _mm256_mul_ps(w0a, i0);
              __m256 sum1 = _mm256_mul_ps(w1a, i1);
              sum0        = madd256_ps(w2a, i2, sum0);
              sum1        = madd256_ps(w3a, i3, sum1);
              sum0        = madd256_ps(w4a, i4, sum0);
              sum0        = _mm256_add_ps(sum0, sum1);
              _mm_store_ss(&ppa[i], _mm_add_ss(sum, hsum256_ps(sum0)));
            }     // x loop
#endif
          }       // y loop
        } else {  // if (w_stride == 1) {
#if 0
          for (serial_size_t y = 0; y < out.height_; ++y, ppa += out.width_) {
            const float *pi0 = (pi + y * stride);
            const float *pi1 = pi0 + 1 * in_padded.width_;
            const float *pi2 = pi0 + 2 * in_padded.width_;
            const float *pi3 = pi0 + 3 * in_padded.width_;
            const float *pi4 = pi0 + 4 * in_padded.width_;
            for (serial_size_t x = 0; x < out.width_; ++x) {
              __m128 sum  = _mm_load_ss(&ppa[x]);
              __m256 i0   = _mm256_loadu_ps(pi0);
              __m256 i1   = _mm256_loadu_ps(pi1);
              __m256 i2   = _mm256_loadu_ps(pi2);
              __m256 i3   = _mm256_loadu_ps(pi3);
              __m256 i4   = _mm256_loadu_ps(pi4);
              __m256 sum0 = _mm256_mul_ps(w0a, i0);
              __m256 sum1 = _mm256_mul_ps(w1a, i1);
              sum0        = madd256_ps(w2a, i2, sum0);
              sum1        = madd256_ps(w3a, i3, sum1);
              sum0        = madd256_ps(w4a, i4, sum0);
              sum0        = _mm256_add_ps(sum0, sum1);
              _mm_store_ss(&ppa[x], _mm_add_ss(sum, hsum256_ps(sum0)));
              pi0 += w_stride;
              pi1 += w_stride;
              pi2 += w_stride;
              pi3 += w_stride;
              pi4 += w_stride;
            }  // x loop
          }    // y loop
#endif
        }
      }  // in depth loop
    }    // out depth loop
  }      // else
}  // avx_conv2d_5x5_kernel float ver

// double ver
template <typename Allocator>
void avx_conv2d_5x5_kernel(const core::conv_params &params,
                           const std::vector<double, Allocator> &in,
                           const std::vector<double, Allocator> &W,
                           const std::vector<double, Allocator> &bias,
                           std::vector<double, Allocator> &a,
                           const bool layer_parallelize) {
  assert(params.weight.height_ == 5 && params.weight.width_ == 5);

  auto &out       = params.out;
  auto &in_padded = params.in_padded;
  auto &tbl       = params.tbl;
  auto w_stride   = params.w_stride;

  const size_t out_area      = out.area();
  double bias_scale          = params.has_bias ? 1.0 : 0.0;
  const __m128d y_bias_scale = _mm_set_sd(bias_scale);
  serial_size_t oidx         = 0;

  const size_t in_stride      = params.h_stride * in_padded.width_;
  const size_t in_padded_area = in_padded.area();

  if (out.height_ == 1 && out.width_ == 1) {
    const double *pw = &W[0];
    for (serial_size_t o = 0; o < out.depth_; ++o) {
      __m256d sum0 = _mm256_setzero_pd();
      __m256d sum1 = _mm256_setzero_pd();
      __m256d sum2 = _mm256_setzero_pd();
      __m256d sum3 = _mm256_setzero_pd();
      __m256d sum4 = _mm256_setzero_pd();
      __m256d sum5 = _mm256_setzero_pd();
      __m128d sum6 = _mm_setzero_pd();
      size_t inidx = 0;
      for (serial_size_t inc = 0; inc < params.in.depth_;
           ++inc, pw += 25, inidx += in_padded_area) {
        if (!tbl.is_connected(o, inc)) {
          continue;
        }
        __m256d w0       = _mm256_loadu_pd(pw + 0);
        __m256d w1       = _mm256_loadu_pd(pw + 4);
        __m256d w2       = _mm256_loadu_pd(pw + 8);
        __m256d w3       = _mm256_loadu_pd(pw + 12);
        __m256d w4       = _mm256_loadu_pd(pw + 16);
        __m256d w5       = _mm256_loadu_pd(pw + 20);
        __m128d w6       = _mm_load_sd(pw + 24);
        const double *pi = (const double *)&in[inidx];
        __m256d i0       = _mm256_loadu_pd(pi + 0);
        __m256d i1       = _mm256_loadu_pd(pi + 4);
        __m256d i2       = _mm256_loadu_pd(pi + 8);
        __m256d i3       = _mm256_loadu_pd(pi + 12);
        __m256d i4       = _mm256_loadu_pd(pi + 16);
        __m256d i5       = _mm256_loadu_pd(pi + 20);
        __m128d i6       = _mm_load_sd(pi + 24);
        __m256d tmp0     = _mm256_mul_pd(w0, i0);
        __m256d tmp1     = _mm256_mul_pd(w1, i1);
        __m256d tmp2     = _mm256_mul_pd(w2, i2);
        __m256d tmp3     = _mm256_mul_pd(w3, i3);
        __m256d tmp4     = _mm256_mul_pd(w4, i4);
        __m256d tmp5     = _mm256_mul_pd(w5, i5);
        __m128d tmp6     = _mm_mul_pd(w6, i6);
        sum0             = _mm256_add_pd(tmp0, sum0);
        sum1             = _mm256_add_pd(tmp1, sum1);
        sum2             = _mm256_add_pd(tmp2, sum2);
        sum3             = _mm256_add_pd(tmp3, sum3);
        sum4             = _mm256_add_pd(tmp4, sum4);
        sum5             = _mm256_add_pd(tmp5, sum5);
        sum6             = _mm_add_pd(tmp6, sum6);
      }
      sum0         = _mm256_add_pd(sum0, sum1);
      sum2         = _mm256_add_pd(sum2, sum3);
      sum4         = _mm256_add_pd(sum4, sum5);
      sum0         = _mm256_add_pd(sum0, sum2);
      __m256d sum  = _mm256_add_pd(sum0, sum4);
      __m128d b    = _mm_load_sd(&bias[o]);
      __m128d hsum = hsum256_pd(sum);
      b            = madd128_sd(b, y_bias_scale, sum6);
      _mm_store_sd(&a[o], _mm_add_sd(hsum, b));
    }
  } else {
    for (serial_size_t o = 0; o < out.depth_; ++o, oidx += out_area) {
      double *pa = &a[oidx];
      double b   = bias[o] * bias_scale;
      {
        size_t headSize = 0;
        __m256d b2      = _mm256_set1_pd(b);
        if (oidx & 3) {
          headSize = 4 - (oidx & 3);
          assert(headSize < out_area);
          for (size_t i = 0; i < headSize; ++i) {
            _mm_store_sd(&pa[i], _mm256_castpd256_pd128(b2));
          }
        }
        size_t cnt  = (out_area - headSize) / 8;
        double *pa2 = pa + headSize;
        for (size_t i = 0; i < cnt; ++i) {
          _mm256_store_pd(&pa2[i * 8 + 0], b2);
          _mm256_store_pd(&pa2[i * 8 + 4], b2);
        }
        for (size_t i = headSize + cnt * 8; i < out_area; ++i) {
          _mm_store_sd(&pa[i], _mm256_castpd256_pd128(b2));
        }
      }

      for (serial_size_t inc = 0; inc < params.in.depth_; ++inc) {
        if (!tbl.is_connected(o, inc)) continue;

        const double *pw =
          (const double *)&W[25 * (params.in.depth_ * o + inc)];
        const double *pi = &in[in_padded.get_index(0, 0, inc)];

        __m256d w0a = _mm256_loadu_pd(pw + 0);
        __m128d w0b = _mm_load_sd(pw + 4);
        __m256d w1a = _mm256_loadu_pd(pw + 5);
        __m128d w1b = _mm_load_sd(pw + 9);
        __m256d w2a = _mm256_loadu_pd(pw + 10);
        __m128d w2b = _mm_load_sd(pw + 14);
        __m256d w3a = _mm256_loadu_pd(pw + 15);
        __m128d w3b = _mm_load_sd(pw + 19);
        __m256d w4a = _mm256_loadu_pd(pw + 20);
        __m128d w4b = _mm_load_sd(pw + 24);

        double *ppa = pa;
        for (serial_size_t y = 0; y < out.height_;
             ++y, pi += in_stride, ppa += out.width_) {
          const double *pi0 = pi + 0 * in_padded.width_;
          const double *pi1 = pi + 1 * in_padded.width_;
          const double *pi2 = pi + 2 * in_padded.width_;
          const double *pi3 = pi + 3 * in_padded.width_;
          const double *pi4 = pi + 4 * in_padded.width_;
          for (serial_size_t x = 0; x < out.width_; ++x) {
            __m128d sum   = _mm_load_sd(&ppa[x]);
            __m256d i0a   = _mm256_loadu_pd(pi0);
            __m128d i0b   = _mm_load_sd(pi0 + 4);
            __m256d i1a   = _mm256_loadu_pd(pi1);
            __m128d i1b   = _mm_load_sd(pi1 + 4);
            __m256d i2a   = _mm256_loadu_pd(pi2);
            __m128d i2b   = _mm_load_sd(pi2 + 4);
            __m256d i3a   = _mm256_loadu_pd(pi3);
            __m128d i3b   = _mm_load_sd(pi3 + 4);
            __m256d i4a   = _mm256_loadu_pd(pi4);
            __m128d i4b   = _mm_load_sd(pi4 + 4);
            __m256d sum_a = _mm256_mul_pd(w0a, i0a);
            __m128d sum_b = _mm_mul_sd(w0b, i0b);
            sum_a         = madd256_pd(w1a, i1a, sum_a);
            sum_b         = madd128_pd(w1b, i1b, sum_b);
            sum_a         = madd256_pd(w2a, i2a, sum_a);
            sum_b         = madd128_pd(w2b, i2b, sum_b);
            sum_a         = madd256_pd(w3a, i3a, sum_a);
            sum_b         = madd128_pd(w3b, i3b, sum_b);
            sum_a         = madd256_pd(w4a, i4a, sum_a);
            sum_b         = madd128_pd(w4b, i4b, sum_b);
            __m128d sum_c = hsum256_pd(sum_a);
            sum           = _mm_add_sd(sum, sum_b);
            _mm_store_sd(&ppa[x], _mm_add_sd(sum, sum_c));
            pi0 += w_stride;
            pi1 += w_stride;
            pi2 += w_stride;
            pi3 += w_stride;
            pi4 += w_stride;
          }  // x loop
        }    // y loop
      }      // in depth loop
    }        // out depth loop
  }          // else
}  // avx_conv2d_5x5_kernel double ver

#endif  // CNN_USE_AVX

inline void conv2d_op_avx(const tensor_t &in_data,
                          const vec_t &W,
                          const vec_t &bias,
                          tensor_t &out_data,
                          const core::conv_params &params,
                          const bool layer_parallelize) {
#ifdef CNN_USE_AVX
  if (params.weight.height_ == 5 && params.weight.width_ == 5) {
    // @todo consider better parallelization
    for (int i=0; i<in_data.size(); ++i) {
      avx_conv2d_5x5_kernel(params, in_data[i], W, bias, out_data[i],
                            layer_parallelize);
    }
    return;
  }
#endif
  conv2d_op_internal(in_data, W, bias, out_data, params, layer_parallelize);
}

}  // namespace kernels
}  // namespace tiny_dnn
