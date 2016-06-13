/*
    Copyright (c) 2013, Taiga Nomi
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
#include "convolutional_layer.h"

//#undef CNN_USE_AVX

#ifdef CNN_USE_AVX

// sum __m256 horizontally (sadly, _mm256_hadd_ps isn't good enough)
// http://stackoverflow.com/a/13222410/4699324
// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
inline float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3+x7, x2+x6, x1+x5, x0+x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1+x5, x0+x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3+x7, x2+x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1+x3 + x5+x7, x0+x2 + x4+x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0+x2 + x4+x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1+x3 + x5+x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0+x1+x2+x3 + x4+x5+x6+x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

#endif // #ifdef CNN_USE_AVX

// byte shifting YMM register across 128-bit lanes (shift amount is immediate)
#if defined(CNN_USE_AVX2)

template <int n>
inline __m256i leftShift(__m256i a)
{
	return _mm256_alignr_epi8(
		a,
		_mm256_permute2x128_si256(a, a, _MM_SHUFFLE(0, 0, 2, 0)),
		16 - n
	);
}

template <int n>
inline __m256 leftShift(__m256 a)
{
	return _mm256_castsi256_ps(leftShift<n>(_mm256_castps_si256(a)));
}

#elif defined(CNN_USE_AVX)

template <int n>
inline __m256 leftShift(__m256 a)
{
	static_assert(false);
}

// http://stackoverflow.com/q/19516585
template <>
inline __m256 leftShift<4>(__m256 x)
{
	__m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
	__m256 t1 = _mm256_permute2f128_ps(t0, t0, 41);
	__m256 y = _mm256_blend_ps(t0, t1, 0x11);
	return y;
}

template <>
inline __m256 leftShift<8>(__m256 x)
{
	__m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
	__m256 t1 = _mm256_permute2f128_ps(t0, t0, 41);
	__m256 y = _mm256_blend_ps(t0, t1, 0x33);
	return y;
}

template <>
inline __m256 leftShift<12>(__m256 x)
{
	__m256 y = _mm256_permute2f128_ps(x, x, 41);
	return y;
}

#endif // #if defined(CNN_USE_AVX2) #elif defined(CNN_USE_AVX)

// idkw but using _mm256_fmadd_ps isn't any faster than the combi of _mm256_mul_ps and _mm256_add_ps in this case with my cpu(i5-4670), though YMMV.
#if 0//defined(CNN_USE_AVX2)

// fused multiply add.
// As a matter of fact, fmadd isn't part of AVX2 instruction set but meh...
// https://en.wikipedia.org/wiki/FMA_instruction_set#CPUs_with_FMA3

__m256 madd(__m256 a, __m256 b, __m256 c)
{
	return _mm256_fmadd_ps(a, b, c);
}

#elif defined(CNN_USE_AVX)

// multiply and add. apparently.
__m256 madd(__m256 a, __m256 b, __m256 c)
{
	return _mm256_add_ps(
		_mm256_mul_ps(a, b),
		c
	);
}

#endif

namespace tiny_cnn {

// optimized for 5x5 kernel

/**
 * 2D convolution layer
 *
 * take input as two-dimensional *image* and applying filtering operation.
 **/
template<typename Activation = activation::identity>
class convolutional_layer_5x5 : public feedforward_layer<Activation> {
public:
    typedef feedforward_layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    /**
    * constructing convolutional layer
    *
    * @param in_width     [in] input image width
    * @param in_height    [in] input image height
    * @param window_size  [in] window(kernel) size of convolution
    * @param in_channels  [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels [in] output image channels
    * @param padding      [in] rounding strategy
    *                          valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
    *                          same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    * @param has_bias     [in] whether to add a bias vector to the filter outputs
    * @param w_stride     [in] specify the horizontal interval at which to apply the filters to the input
    * @param h_stride     [in] specify the vertical interval at which to apply the filters to the input
    **/
    convolutional_layer_5x5(cnn_size_t in_width,
							cnn_size_t in_height,
							cnn_size_t window_size,
							cnn_size_t in_channels,
							cnn_size_t out_channels,
							padding    pad_type = padding::valid,
							bool       has_bias = true,
							cnn_size_t w_stride = 1,
							cnn_size_t h_stride = 1)
        : Base(std_input_order(has_bias))
    {
    	assert(window_size == 5);
        conv_set_params(shape3d(in_width, in_height, in_channels), window_size, window_size,
                        out_channels, pad_type, has_bias, w_stride, h_stride);
    }

    /**
    * constructing convolutional layer
    *
    * @param in_width      [in] input image width
    * @param in_height     [in] input image height
    * @param window_width  [in] window_width(kernel) size of convolution
    * @param window_height [in] window_height(kernel) size of convolution
    * @param in_channels   [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels  [in] output image channels
    * @param padding       [in] rounding strategy
    *                          valid: use valid pixels of input only. output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels
    *                          same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    * @param has_bias     [in] whether to add a bias vector to the filter outputs
    * @param w_stride     [in] specify the horizontal interval at which to apply the filters to the input
    * @param h_stride     [in] specify the vertical interval at which to apply the filters to the input
    **/
    convolutional_layer_5x5(cnn_size_t in_width,
							cnn_size_t in_height,
							cnn_size_t window_width,
							cnn_size_t window_height,
							cnn_size_t in_channels,
							cnn_size_t out_channels,
							padding    pad_type = padding::valid,
							bool       has_bias = true,
							cnn_size_t w_stride = 1,
							cnn_size_t h_stride = 1)
        : Base(std_input_order(has_bias))
    {
    	assert(window_width == 5);
    	assert(window_height == 5);
        conv_set_params(shape3d(in_width, in_height, in_channels), window_width, window_height,
                        out_channels, pad_type, has_bias, w_stride, h_stride);
    }

    /**
    * constructing convolutional layer
    *
    * @param in_width         [in] input image width
    * @param in_height        [in] input image height
    * @param window_size      [in] window(kernel) size of convolution
    * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels     [in] output image channels
    * @param connection_table [in] definition of connections between in-channels and out-channels
    * @param pad_type         [in] rounding strategy
    *                               valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
    *                               same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    * @param has_bias         [in] whether to add a bias vector to the filter outputs
    * @param w_stride         [in] specify the horizontal interval at which to apply the filters to the input
    * @param h_stride         [in] specify the vertical interval at which to apply the filters to the input
    **/
    convolutional_layer_5x5(cnn_size_t              in_width,
							cnn_size_t              in_height,
							cnn_size_t              window_size,
							cnn_size_t              in_channels,
							cnn_size_t              out_channels,
							const connection_table& connection_table,
							padding                 pad_type = padding::valid,
							bool                    has_bias = true,
							cnn_size_t              w_stride = 1,
							cnn_size_t              h_stride = 1)
        : Base(std_input_order(has_bias)), tbl_(connection_table)
    {
    	assert(window_size == 5);
        conv_set_params(shape3d(in_width, in_height, in_channels), window_size, window_size,
                        out_channels, pad_type, has_bias, w_stride, h_stride);
    }

    /**
    * constructing convolutional layer
    *
    * @param in_width         [in] input image width
    * @param in_height        [in] input image height
    * @param window_width     [in] window_width(kernel) size of convolution
    * @param window_height    [in] window_height(kernel) size of convolution
    * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels     [in] output image channels
    * @param connection_table [in] definition of connections between in-channels and out-channels
    * @param pad_type         [in] rounding strategy
    *                               valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
    *                               same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    * @param has_bias         [in] whether to add a bias vector to the filter outputs
    * @param w_stride         [in] specify the horizontal interval at which to apply the filters to the input
    * @param h_stride         [in] specify the vertical interval at which to apply the filters to the input
    **/
    convolutional_layer_5x5(cnn_size_t              in_width,
							cnn_size_t              in_height,
							cnn_size_t              window_width,
							cnn_size_t              window_height,
							cnn_size_t              in_channels,
							cnn_size_t              out_channels,
							const connection_table& connection_table,
							padding                 pad_type = padding::valid,
							bool                    has_bias = true,
							cnn_size_t              w_stride = 1,
							cnn_size_t              h_stride = 1)
        : Base(has_bias ? 3 : 2, 1, std_input_order(has_bias)), tbl_(connection_table)
    {
    	assert(window_width == 5);
    	assert(window_height == 5);
        conv_set_params(shape3d(in_width, in_height, in_channels), window_width, window_height,
                        out_channels, pad_type, has_bias, w_stride, h_stride);
    }

    ///< number of incoming connections for each output unit
    virtual size_t fan_in_size() const override
    {
        return 25 /* weight_.width_ * weight_.height_ */ * in_.depth_;
    }

    ///< number of outgoing connections for each input unit
    virtual size_t fan_out_size() const override
    {
        return (5 /* weight_.width_ */ / w_stride_) * (5 /* weight_.height_ */ / h_stride_) * out_.depth_;
    }

    void forward_propagation(cnn_size_t index,
                             const std::vector<vec_t*>& in_data,
                             std::vector<vec_t*>& out_data) override {
		forward_propagation_impl(index, in_data, out_data);
	}

    void forward_propagation_impl(cnn_size_t index,
                                  const std::vector<dvec_t*>& in_data,
                                  std::vector<dvec_t*>& out_data) {
        copy_and_pad_input(*in_data[0], static_cast<int>(index));
        const dvec_t& W   = *in_data[1];
        dvec_t&       out = *out_data[0];
        dvec_t&       a   = *out_data[1];
        const dvec_t &in  = *(conv_layer_worker_storage_[index].prev_out_padded_); // input
        
        std::fill(a.begin(), a.end(), double(0));

        for_i(parallelize_, out_.depth_, [&](int o) {
            for (cnn_size_t inc = 0; inc < in_.depth_; inc++) {
                if (!tbl_.is_connected(o, inc)) continue;

                const double *pw = &W[weight_.get_index(0, 0, in_.depth_ * o + inc)];
                const double *pi = &in[in_padded_.get_index(0, 0, inc)];
                double *pa = &a[out_.get_index(0, 0, o)];

                for (cnn_size_t y = 0; y < out_.height_; y++) {
                    for (cnn_size_t x = 0; x < out_.width_; x++) {
                        const double * ppw = pw;
                        const double * ppi = pi + (y * h_stride_) * in_padded_.width_ + x * w_stride_;
                        double sum = double(0);

                        // should be optimized for small kernel(3x3,5x5)
                        for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                            for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                                sum += *ppw++ * ppi[wy * in_padded_.width_ + wx];
                            }
                        }
                        pa[y * out_.width_ + x] += sum;
						//printf("%d %d %d %f\n", inc, y, x, sum);
                    }
                }
            }

            if (has_bias_) {
                const dvec_t& bias = *in_data[2];
                double *pa = &a[out_.get_index(0, 0, o)];
                double b = bias[o];
                std::for_each(pa, pa + out_.width_ * out_.height_, [&](double& f) { f += b; });
            }
        });

        for_i(parallelize_, out_.size(), [&](int i) {
            out[i] = h_.f(a, i);
        });
    }

    void forward_propagation_impl(cnn_size_t index,
                                  const std::vector<fvec_t*>& in_data,
                                  std::vector<fvec_t*>& out_data) {
        copy_and_pad_input(*in_data[0], static_cast<int>(index));
        const fvec_t& w   = *in_data[1];
		const fvec_t& bias = *in_data[2];
        fvec_t&       out = *out_data[0];
        fvec_t&       a   = *out_data[1];
        const fvec_t& in  = *(conv_layer_worker_storage_[index].prev_out_padded_); // input

#ifdef CNN_USE_AVX
		static const __m256i mask = _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0);
		if (out_.height_ == 1 && out_.width_ == 1) {
			const size_t stride = h_stride_ * in_padded_.width_;
	        for_i(parallelize_, out_.depth_, [&](int o) {
				__m256 sum = _mm256_setzero_ps();
				size_t widx = 25/* weight_.area() */ * in_.depth_ * o;
				size_t inidx = 0;
				size_t inarea = in_padded_.area();
				for (cnn_size_t inc=0; inc<in_.depth_; ++inc, widx+=25, inidx+=inarea) {
	                if (!tbl_.is_connected(o, inc)) {
						continue;
					}
					const float* pw = (const float*) &w[widx];
					__m256 w0 = _mm256_maskload_ps(pw+0, mask);
					__m256 w1 = _mm256_maskload_ps(pw+5, mask);
					__m256 w2 = _mm256_maskload_ps(pw+10, mask);
					__m256 w3 = _mm256_maskload_ps(pw+15, mask);
					__m256 w4 = _mm256_maskload_ps(pw+20, mask);
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
				a[o] = sum8(sum) + (has_bias_ ? bias[o] : 0);
	        });
		}else
#endif // #ifdef CNN_USE_AVX
		{
	        for_i(parallelize_, out_.depth_, [&](int o) {
				float* pa = &a[out_.get_index(0, 0, o)];
				float b = has_bias_ ? bias[o] : 0;
				const size_t area = out_.area();
#ifdef CNN_USE_AVX
				__m256 b2 = _mm256_set1_ps(b);
				size_t cnt = area / 16;
				for (size_t i=0; i<cnt; ++i) {
					_mm256_storeu_ps(&pa[i*16+0], b2);
					_mm256_storeu_ps(&pa[i*16+8], b2);
				}
				for (size_t i=cnt*16; i<area; ++i) {
					pa[i] = b;
				}
#else // #ifdef CNN_USE_AVX
				for (size_t i=0; i<area; ++i) {
					pa[i] = b;
				}
#endif // #ifdef CNN_USE_AVX
				for (cnn_size_t inc=0; inc<in_.depth_; ++inc) {
	                if (!tbl_.is_connected(o, inc)) continue;

	                const float* pw = (const float*) &w[25 * (in_.depth_ * o + inc)];
	                const float* pi = (const float*) &in[in_padded_.get_index(0, 0, inc)];

#ifdef CNN_USE_AVX
					__m256 w0 = _mm256_maskload_ps(pw+0, mask);
					__m256 w1 = _mm256_maskload_ps(pw+5, mask);
					__m256 w2 = _mm256_maskload_ps(pw+10, mask);
					__m256 w3 = _mm256_maskload_ps(pw+15, mask);
					__m256 w4 = _mm256_maskload_ps(pw+20, mask);
					size_t stride = h_stride_ * in_padded_.width_;
#else // #ifdef CNN_USE_AVX
					float w00 = *pw++;
					float w01 = *pw++;
					float w02 = *pw++;
					float w03 = *pw++;
					float w04 = *pw++;
					float w10 = *pw++;
					float w11 = *pw++;
					float w12 = *pw++;
					float w13 = *pw++;
					float w14 = *pw++;
					float w20 = *pw++;
					float w21 = *pw++;
					float w22 = *pw++;
					float w23 = *pw++;
					float w24 = *pw++;
					float w30 = *pw++;
					float w31 = *pw++;
					float w32 = *pw++;
					float w33 = *pw++;
					float w34 = *pw++;
					float w40 = *pw++;
					float w41 = *pw++;
					float w42 = *pw++;
					float w43 = *pw++;
					float w44 = *pw++;
#endif // #ifdef CNN_USE_AVX
					float* ppa = pa;
	                for (cnn_size_t y = 0; y < out_.height_; y++) {
#ifdef CNN_USE_AVX
						const float* pi0 = (pi + y * stride);
						const float* pi1 = pi0 + 1 * stride;
						const float* pi2 = pi0 + 2 * stride;
						const float* pi3 = pi0 + 3 * stride;
						const float* pi4 = pi0 + 4 * stride;
	                    for (cnn_size_t x=0; x<out_.width_; ++x) {
							__m256 i0 = _mm256_loadu_ps(pi0);
							__m256 i1 = _mm256_loadu_ps(pi1);
							__m256 i2 = _mm256_loadu_ps(pi2);
							__m256 i3 = _mm256_loadu_ps(pi3);
							__m256 i4 = _mm256_loadu_ps(pi4);
							__m256 sum0 = _mm256_mul_ps(w0, i0);
							__m256 sum1 = _mm256_mul_ps(w1, i1);
							sum0 = madd(w2, i2, sum0);
							sum1 = madd(w3, i3, sum1);
							sum0 = madd(w4, i4, sum0);
							sum0 = _mm256_add_ps(sum0, sum1);
							ppa[x] += sum8(sum0);
	//						printf("%d %d %d %f\n", inc, y, x, ppa[x]);
							pi0 += w_stride_;
							pi1 += w_stride_;
							pi2 += w_stride_;
							pi3 += w_stride_;
							pi4 += w_stride_;
	                    }
#else // #ifdef CNN_USE_AVX
	                    for (cnn_size_t x = 0; x < out_.width_; x++) {
	                        const float * ppw = pw;
	                        const float * ppi = pi + (y * h_stride_) * in_padded_.width_ + x * w_stride_;
							float sum = float(0);
							sum += w00 * ppi[0];
							sum += w01 * ppi[1];
							sum += w02 * ppi[2];
							sum += w03 * ppi[3];
							sum += w04 * ppi[4];
							ppi += in_padded_.width_;
							sum += w10 * ppi[0];
							sum += w11 * ppi[1];
							sum += w12 * ppi[2];
							sum += w13 * ppi[3];
							sum += w14 * ppi[4];
							ppi += in_padded_.width_;
							sum += w20 * ppi[0];
							sum += w21 * ppi[1];
							sum += w22 * ppi[2];
							sum += w23 * ppi[3];
							sum += w24 * ppi[4];
							ppi += in_padded_.width_;
							sum += w30 * ppi[0];
							sum += w31 * ppi[1];
							sum += w32 * ppi[2];
							sum += w33 * ppi[3];
							sum += w34 * ppi[4];
							ppi += in_padded_.width_;
							sum += w40 * ppi[0];
							sum += w41 * ppi[1];
							sum += w42 * ppi[2];
							sum += w43 * ppi[3];
							sum += w44 * ppi[4];
							ppi += in_padded_.width_;
	                        ppa[x] += sum;
	//						printf("%d %d %d %f\n", inc, y, x, sum);
	                    }
#endif // #ifdef CNN_USE_AVX
						ppa += out_.width_;
	                }
	            }
	            
	        });
		}

		// apply acativation function
		h_.f(out, a);
    }

    float& weight_at(cnn_size_t in_channel, cnn_size_t out_channel, cnn_size_t kernel_x, cnn_size_t kernel_y) {
        vec_t* w = this->get_weights()[0];
        return w[weight_.get_index(kernel_x, kernel_y, in_.depth_ * out_channel + in_channel)];
    }

    void back_propagation(cnn_size_t                 index,
                          const std::vector<vec_t*>& in_data,
                          const std::vector<vec_t*>& out_data,
                          std::vector<vec_t*>&       out_grad,
                          std::vector<vec_t*>&       in_grad) override {
		back_propagation_impl(index, in_data, out_data, out_grad, in_grad);
	}

    void back_propagation_impl(cnn_size_t             index,
                          const std::vector<dvec_t*>& in_data,
                          const std::vector<dvec_t*>& out_data,
                          std::vector<dvec_t*>&       out_grad,
                          std::vector<dvec_t*>&       in_grad) {
        conv_layer_worker_specific_storage& cws = conv_layer_worker_storage_[index];

        const dvec_t& prev_out = *(cws.prev_out_padded_);
        const dvec_t& W = *in_data[1];
        dvec_t*       prev_delta = (pad_type_ == padding::same) ? &cws.prev_delta_padded_ : in_grad[0];
        dvec_t&       dW = *in_grad[1];
        dvec_t&       curr_delta = *out_grad[1];

        assert(W.size() == weight_.size());
        assert(dW.size() == weight_.size());
        assert(curr_delta.size() == out_shape()[0].size());

        this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

        std::fill(prev_delta->begin(), prev_delta->end(), double(0));

        // propagate delta to previous layer
        for_i(in_.depth_, [&](int inc) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
                if (!tbl_.is_connected(outc, inc)) continue;

                const double *pw = &W[weight_.get_index(0, 0, in_.depth_ * outc + inc)];
                const double *pdelta_src = &curr_delta[out_.get_index(0, 0, outc)];
                double *pdelta_dst = &(*prev_delta)[in_padded_.get_index(0, 0, inc)];

                for (cnn_size_t y = 0; y < out_.height_; y++) {
                    for (cnn_size_t x = 0; x < out_.width_; x++) {
                        const double * ppw = pw;
                        const double ppdelta_src = pdelta_src[y * out_.width_ + x];
                        double * ppdelta_dst = pdelta_dst + y * h_stride_ * in_padded_.width_ + x * w_stride_;

                        for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                            for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                                ppdelta_dst[wy * in_padded_.width_ + wx] += *ppw++ * ppdelta_src;
                            }
                        }
                    }
                }
            }
        });

        // accumulate dw
        for_i(in_.depth_, [&](int inc) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {

                if (!tbl_.is_connected(outc, inc)) continue;

                for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                    for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                        double dst = double(0);
                        const double * prevo = &prev_out[in_padded_.get_index(wx, wy, inc)];
                        const double * delta = &curr_delta[out_.get_index(0, 0, outc)];

                        for (cnn_size_t y = 0; y < out_.height_; y++) {
                            dst += vectorize::dot(prevo + y * in_padded_.width_, delta + y * out_.width_, out_.width_);
                        }
                        dW[weight_.get_index(wx, wy, in_.depth_ * outc + inc)] += dst;
                    }
                }
            }
        });

        // accumulate db
        if (has_bias_) {
            dvec_t& db = *in_grad[2];

            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
                const double *delta = &curr_delta[out_.get_index(0, 0, outc)];
                db[outc] += std::accumulate(delta, delta + out_.width_ * out_.height_, double(0));
            }
        }

        if (pad_type_ == padding::same)
            copy_and_unpad_delta(cws.prev_delta_padded_, *in_grad[0]);
    }

    void back_propagation_impl(cnn_size_t             index,
                          const std::vector<fvec_t*>& in_data,
                          const std::vector<fvec_t*>& out_data,
                          std::vector<fvec_t*>&       out_grad,
                          std::vector<fvec_t*>&       in_grad) {

        conv_layer_worker_specific_storage& cws = conv_layer_worker_storage_[index];

        const fvec_t& prev_out = *(cws.prev_out_padded_);
        const fvec_t& w = *in_data[1];
        fvec_t*       prev_delta = (pad_type_ == padding::same) ? &cws.prev_delta_padded_ : in_grad[0];
        fvec_t&       dW = *in_grad[1];
        fvec_t&       curr_delta = *out_grad[1];

        assert(w.size() == weight_.size());
        assert(dW.size() == weight_.size());
        assert(curr_delta.size() == out_shape()[0].size());

        this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

#ifdef CNN_USE_AVX
		__m256 z = _mm256_setzero_ps();
		size_t sz = prev_delta->size();
		size_t cnt = sz / 16;
		for (size_t i=0; i<cnt; ++i) {
			_mm256_store_ps(&(*prev_delta)[i*16+0], z);
			_mm256_store_ps(&(*prev_delta)[i*16+8], z);
		}
		for (size_t i=cnt*16; i<sz; ++i) {
			(*prev_delta)[i] = 0;
		}
#else // #ifdef CNN_USE_AVX
        std::fill(prev_delta->begin(), prev_delta->end(), float(0));
#endif // #ifdef CNN_USE_AVX

#ifdef CNN_USE_AVX
		static const __m256i mask = _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0);
		// propagate delta to previous layer
		if (w_stride_ == 1 && out_.width_ >= 4) {
			for_i(in_.depth_, [&](int inc) {
				for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
					if (!tbl_.is_connected(outc, inc)) continue;
					const float* pw = &w[25 * (in_.depth_ * outc + inc)];
					const float* pdelta_src = &curr_delta[out_.get_index(0, 0, outc)];
					float* pdelta_dst = &(*prev_delta)[in_padded_.get_index(0, 0, inc)];
					__m256 w0a = _mm256_maskload_ps((const float*)pw, mask); pw += 5;
					__m256 w1a = _mm256_maskload_ps((const float*)pw, mask); pw += 5;
					__m256 w2a = _mm256_maskload_ps((const float*)pw, mask); pw += 5;
					__m256 w3a = _mm256_maskload_ps((const float*)pw, mask); pw += 5;
					__m256 w4a = _mm256_maskload_ps((const float*)pw, mask); pw += 5;
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
					for (cnn_size_t y = 0; y < out_.height_; y++) {
						float* delta_dst0 = pdelta_dst;
						float* delta_dst1 = &pdelta_dst[in_padded_.width_ * 1];
						float* delta_dst2 = &pdelta_dst[in_padded_.width_ * 2];
						float* delta_dst3 = &pdelta_dst[in_padded_.width_ * 3];
						float* delta_dst4 = &pdelta_dst[in_padded_.width_ * 4];
						cnn_size_t nblocks = out_.width_ >> 2;
						for (cnn_size_t n = 0; n < nblocks; ++n) {
							__m128 delta_src = _mm_loadu_ps(pdelta_src + n * 4);
							__m256 dst0 = _mm256_loadu_ps(delta_dst0);
							__m256 dst1 = _mm256_loadu_ps(delta_dst1);
							__m256 dst2 = _mm256_loadu_ps(delta_dst2);
							__m256 dst3 = _mm256_loadu_ps(delta_dst3);
							__m256 dst4 = _mm256_loadu_ps(delta_dst4);
							__m256 tmp = _mm256_set_m128(delta_src, delta_src);
							__m256 delta_src0 = _mm256_permute_ps(tmp, _MM_SHUFFLE(0, 0, 0, 0));
							__m256 delta_src1 = _mm256_permute_ps(tmp, _MM_SHUFFLE(1, 1, 1, 1));
							__m256 delta_src2 = _mm256_permute_ps(tmp, _MM_SHUFFLE(2, 2, 2, 2));
							__m256 delta_src3 = _mm256_permute_ps(tmp, _MM_SHUFFLE(3, 3, 3, 3));
							dst0 = madd(w0a, delta_src0, dst0);
							dst1 = madd(w1a, delta_src0, dst1);
							dst2 = madd(w2a, delta_src0, dst2);
							dst3 = madd(w3a, delta_src0, dst3);
							dst4 = madd(w4a, delta_src0, dst4);
							dst0 = madd(w0b, delta_src1, dst0);
							dst1 = madd(w1b, delta_src1, dst1);
							dst2 = madd(w2b, delta_src1, dst2);
							dst3 = madd(w3b, delta_src1, dst3);
							dst4 = madd(w4b, delta_src1, dst4);
							dst0 = madd(w0c, delta_src2, dst0);
							dst1 = madd(w1c, delta_src2, dst1);
							dst2 = madd(w2c, delta_src2, dst2);
							dst3 = madd(w3c, delta_src2, dst3);
							dst4 = madd(w4c, delta_src2, dst4);
							dst0 = madd(w0d, delta_src3, dst0);
							dst1 = madd(w1d, delta_src3, dst1);
							dst2 = madd(w2d, delta_src3, dst2);
							dst3 = madd(w3d, delta_src3, dst3);
							dst4 = madd(w4d, delta_src3, dst4);
							_mm256_storeu_ps(delta_dst0, dst0);
							_mm256_storeu_ps(delta_dst1, dst1);
							_mm256_storeu_ps(delta_dst2, dst2);
							_mm256_storeu_ps(delta_dst3, dst3);
							_mm256_storeu_ps(delta_dst4, dst4);
							delta_dst0 += 4;
							delta_dst1 += 4;
							delta_dst2 += 4;
							delta_dst3 += 4;
							delta_dst4 += 4;
						}
						for (cnn_size_t x = nblocks * 4; x < out_.width_; x++) {
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
							++delta_dst0;
							++delta_dst1;
							++delta_dst2;
							++delta_dst3;
							++delta_dst4;
						}
						pdelta_src += out_.width_;
						pdelta_dst += h_stride_ * in_padded_.width_;
					}
				}
			});
		}else if (out_.height_ == 1 && out_.width_ == 1) {
			for_i(in_.depth_, [&](int inc) {
				float* pdelta_dst = &(*prev_delta)[in_padded_.get_index(0, 0, inc)];
				float* delta_dst0 = pdelta_dst;
				float* delta_dst1 = &pdelta_dst[in_padded_.width_ * 1];
				float* delta_dst2 = &pdelta_dst[in_padded_.width_ * 2];
				float* delta_dst3 = &pdelta_dst[in_padded_.width_ * 3];
				float* delta_dst4 = &pdelta_dst[in_padded_.width_ * 4];
				__m256 dst0 = _mm256_loadu_ps(delta_dst0);
				__m256 dst1 = _mm256_loadu_ps(delta_dst1);
				__m256 dst2 = _mm256_loadu_ps(delta_dst2);
				__m256 dst3 = _mm256_loadu_ps(delta_dst3);
				__m256 dst4 = _mm256_loadu_ps(delta_dst4);
				size_t widx = 25 * inc;
				size_t wstep = 25 * in_.depth_;
				for (cnn_size_t outc = 0; outc < out_.depth_; outc++, widx+=wstep) {
					if (!tbl_.is_connected(outc, inc)) {
						continue;
					}
					__m256 delta_src = _mm256_broadcast_ss(&curr_delta[outc]);
					const float* pw = (const float*)&w[widx];
					__m256 w0a = _mm256_maskload_ps(pw+0, mask);
					__m256 w1a = _mm256_maskload_ps(pw+5, mask);
					__m256 w2a = _mm256_maskload_ps(pw+10, mask);
					__m256 w3a = _mm256_maskload_ps(pw+15, mask);
					__m256 w4a = _mm256_maskload_ps(pw+20, mask);
					dst0 = madd(w0a, delta_src, dst0);
					dst1 = madd(w1a, delta_src, dst1);
					dst2 = madd(w2a, delta_src, dst2);
					dst3 = madd(w3a, delta_src, dst3);
					dst4 = madd(w4a, delta_src, dst4);
				}
				_mm256_storeu_ps(delta_dst0, dst0);
				_mm256_storeu_ps(delta_dst1, dst1);
				_mm256_storeu_ps(delta_dst2, dst2);
				_mm256_storeu_ps(delta_dst3, dst3);
				_mm256_storeu_ps(delta_dst4, dst4);
			});

		}else
#endif // #ifdef CNN_USE_AVX
		{
			for_i(in_.depth_, [&](int inc) {
				float* pdelta_dst_org = &(*prev_delta)[in_padded_.get_index(0, 0, inc)];
				for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
					if (!tbl_.is_connected(outc, inc)) continue;

					const float* pw = &w[25 * (in_.depth_ * outc + inc)];
					const float* pdelta_src = &curr_delta[out_.get_index(0, 0, outc)];
					float* pdelta_dst = pdelta_dst_org;
#ifdef CNN_USE_AVX
					__m256 w0a = _mm256_maskload_ps((const float*)pw, mask); pw += 5;
					__m256 w1a = _mm256_maskload_ps((const float*)pw, mask); pw += 5;
					__m256 w2a = _mm256_maskload_ps((const float*)pw, mask); pw += 5;
					__m256 w3a = _mm256_maskload_ps((const float*)pw, mask); pw += 5;
					__m256 w4a = _mm256_maskload_ps((const float*)pw, mask); pw += 5;
#else // #ifdef CNN_USE_AVX
					float w00 = *pw++;
					float w01 = *pw++;
					float w02 = *pw++;
					float w03 = *pw++;
					float w04 = *pw++;
					float w10 = *pw++;
					float w11 = *pw++;
					float w12 = *pw++;
					float w13 = *pw++;
					float w14 = *pw++;
					float w20 = *pw++;
					float w21 = *pw++;
					float w22 = *pw++;
					float w23 = *pw++;
					float w24 = *pw++;
					float w30 = *pw++;
					float w31 = *pw++;
					float w32 = *pw++;
					float w33 = *pw++;
					float w34 = *pw++;
					float w40 = *pw++;
					float w41 = *pw++;
					float w42 = *pw++;
					float w43 = *pw++;
					float w44 = *pw++;
#endif // #ifdef CNN_USE_AVX
					for (cnn_size_t y = 0; y < out_.height_; y++) {
#ifdef CNN_USE_AVX
						float* delta_dst0 = pdelta_dst;
						float* delta_dst1 = &pdelta_dst[in_padded_.width_ * 1];
						float* delta_dst2 = &pdelta_dst[in_padded_.width_ * 2];
						float* delta_dst3 = &pdelta_dst[in_padded_.width_ * 3];
						float* delta_dst4 = &pdelta_dst[in_padded_.width_ * 4];
						for (cnn_size_t x = 0; x < out_.width_; x++) {
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
							delta_dst0 += w_stride_;
							delta_dst1 += w_stride_;
							delta_dst2 += w_stride_;
							delta_dst3 += w_stride_;
							delta_dst4 += w_stride_;
						}
						pdelta_src += out_.width_;
						pdelta_dst += h_stride_ * in_padded_.width_;
#else // #ifdef CNN_USE_AVX
						float* delta_dst0 = pdelta_dst;
						float* delta_dst1 = &pdelta_dst[in_padded_.width_ * 1];
						float* delta_dst2 = &pdelta_dst[in_padded_.width_ * 2];
						float* delta_dst3 = &pdelta_dst[in_padded_.width_ * 3];
						float* delta_dst4 = &pdelta_dst[in_padded_.width_ * 4];
						for (cnn_size_t x = 0; x < out_.width_; x++) {
							const float delta_src = pdelta_src[x];
							delta_dst0[0] += w00 * delta_src;
							delta_dst0[1] += w01 * delta_src;
							delta_dst0[2] += w02 * delta_src;
							delta_dst0[3] += w03 * delta_src;
							delta_dst0[4] += w04 * delta_src;
							delta_dst1[0] += w10 * delta_src;
							delta_dst1[1] += w11 * delta_src;
							delta_dst1[2] += w12 * delta_src;
							delta_dst1[3] += w13 * delta_src;
							delta_dst1[4] += w14 * delta_src;
							delta_dst2[0] += w20 * delta_src;
							delta_dst2[1] += w21 * delta_src;
							delta_dst2[2] += w22 * delta_src;
							delta_dst2[3] += w23 * delta_src;
							delta_dst2[4] += w24 * delta_src;
							delta_dst3[0] += w30 * delta_src;
							delta_dst3[1] += w31 * delta_src;
							delta_dst3[2] += w32 * delta_src;
							delta_dst3[3] += w33 * delta_src;
							delta_dst3[4] += w34 * delta_src;
							delta_dst4[0] += w40 * delta_src;
							delta_dst4[1] += w41 * delta_src;
							delta_dst4[2] += w42 * delta_src;
							delta_dst4[3] += w43 * delta_src;
							delta_dst4[4] += w44 * delta_src;
							delta_dst0 += w_stride_;
							delta_dst1 += w_stride_;
							delta_dst2 += w_stride_;
							delta_dst3 += w_stride_;
							delta_dst4 += w_stride_;
						}
						pdelta_src += out_.width_;
						pdelta_dst += h_stride_ * in_padded_.width_;
#endif // #ifdef CNN_USE_AVX
					}
				}
			});
		}

        // accumulate dw
		if (out_.width_ == 1 && out_.height_ == 1) {
			for_i(in_.depth_, [&](int inc) {
#ifdef CNN_USE_AVX
				union {
					struct {
						__m256 prevos0;
						__m256 prevos1;
						__m256 prevos2;
						__m256 prevos3;
					} s;
					float floats[32];
				};
				size_t base_idx = inc * in_padded_.area();
				size_t in_padded_width = in_padded_.width_;
				_mm256_storeu_ps(&floats[0], _mm256_loadu_ps(&prev_out[base_idx + in_padded_width * 0]));
				_mm256_storeu_ps(&floats[5], _mm256_loadu_ps(&prev_out[base_idx + in_padded_width * 1]));
				_mm256_storeu_ps(&floats[10], _mm256_loadu_ps(&prev_out[base_idx + in_padded_width * 2]));
				_mm256_storeu_ps(&floats[15], _mm256_loadu_ps(&prev_out[base_idx + in_padded_width * 3]));
				_mm256_storeu_ps(&floats[20], _mm256_loadu_ps(&prev_out[base_idx + in_padded_width * 4]));
				cnn_size_t widx = 25 * inc;
				cnn_size_t widx_delta = 25 * in_.depth_;
				for (cnn_size_t outc = 0; outc < out_.depth_; outc++, widx+=widx_delta) {
					if (!tbl_.is_connected(outc, inc)) {
						continue;
					}
					__m256 delta = _mm256_broadcast_ss(&curr_delta[outc]);
					__m256 w0 = _mm256_loadu_ps(&dW[widx+0]);
					__m256 w1 = _mm256_loadu_ps(&dW[widx+8]);
					__m256 w2 = _mm256_loadu_ps(&dW[widx+16]);
					w0 = madd(s.prevos0, delta, w0);
					w1 = madd(s.prevos1, delta, w1);
					w2 = madd(s.prevos2, delta, w2);
					_mm256_storeu_ps(&dW[widx+0], w0);
					_mm256_storeu_ps(&dW[widx+8], w1);
					_mm256_storeu_ps(&dW[widx+16], w2);
					dW[widx+24] += s.prevos3.m256_f32[0] * delta.m256_f32[0];
				}
#else // #ifdef CNN_USE_AVX
				for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
					if (!tbl_.is_connected(outc, inc)) continue;
					const float delta = curr_delta[out_.get_index(0, 0, outc)];
					for (cnn_size_t wy = 0; wy < 5 /* weight_.height_ */; wy++) {
						for (cnn_size_t wx = 0; wx < 5 /* weight_.width_ */; wx++) {
							cnn_size_t idx = in_padded_.get_index(wx, wy, inc);
							const float prevo = prev_out[idx];
							// vectorize::dot
							cnn_size_t widx = weight_.get_index(wx, wy, in_.depth_ * outc + inc);
							dW[widx] += prevo * delta;
						}
					}
				}
#endif // #ifdef CNN_USE_AVX
			});
		}else {
			for_i(in_.depth_, [&](int inc) {
				for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {

					if (!tbl_.is_connected(outc, inc)) continue;
					const float* delta = &curr_delta[out_.get_index(0, 0, outc)];

#ifdef CNN_USE_AVX
					// prepare load-mask beforehand
					size_t nblocks = out_.width_ >> 3;
					static const int32_t masks[] = {
						-1, -1, -1, -1,
						-1, -1, -1, -1,
						0, 0, 0, 0,
						0, 0, 0, 0,
					};
					size_t remainder = out_.width_ & 7;
					__m256i mask = _mm256_loadu_si256((const __m256i*)(masks + 8 - remainder));
#endif // #ifdef CNN_USE_AVX
					cnn_size_t widx = weight_.get_index(0, 0, in_.depth_ * outc + inc);
					for (cnn_size_t wy = 0; wy < 5 /* weight_.height_ */; wy++) {
						for (cnn_size_t wx = 0; wx < 5 /* weight_.width_ */; wx++) {
							const float* prevo = &prev_out[in_padded_.get_index(wx, wy, inc)];
#ifdef CNN_USE_AVX
							__m256 dst = _mm256_setzero_ps();
							__m256 a, b;
							for (cnn_size_t y = 0; y < out_.height_; y++) {
								// vectorize::dot
								const float* pa = prevo + y * in_padded_.width_;
								const float* pb = delta + y * out_.width_;
								for (size_t i=0; i<nblocks; ++i) {
									a = _mm256_loadu_ps(pa+8*i);
									b = _mm256_loadu_ps(pb+8*i);
									dst = madd(a, b, dst);
								}
								if (remainder) {
									a = _mm256_maskload_ps(pa+8*nblocks, mask);
									b = _mm256_maskload_ps(pb+8*nblocks, mask);
									dst = madd(a, b, dst);
								}
							}
							dW[widx++] += sum8(dst);
#else // #ifdef CNN_USE_AVX
							float dst = vectorize::dot(prevo, delta, out_.width_);
							for (cnn_size_t y = 1; y < out_.height_; y++) {
								dst += vectorize::dot(prevo + y * in_padded_.width_, delta + y * out_.width_, out_.width_);
							}
							dW[widx++] += dst;
#endif // #ifdef CNN_USE_AVX
						}
					}
				}
			});
		}

        // accumulate db
        if (has_bias_) {
            fvec_t& db = *in_grad[2];
			if (out_.width_ == 1 && out_.height_ == 1) {
				for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
					db[outc] += curr_delta[outc];
				}
			}else {
				for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
					const float *delta = &curr_delta[out_.get_index(0, 0, outc)];
					db[outc] += std::accumulate(delta, delta + out_.width_ * out_.height_, float(0));
				}
			}
        }

        if (pad_type_ == padding::same)
            copy_and_unpad_delta(cws.prev_delta_padded_, *in_grad[0]);
    }

    std::vector<index3d<cnn_size_t>> in_shape() const override {
        if (has_bias_) {
            return{ in_, weight_, index3d<cnn_size_t>(1, 1, out_.depth_) };
        } else {
            return { in_, weight_ };
        }
    }

    std::vector<index3d<cnn_size_t>> out_shape() const override { return {out_, out_}; }
    std::string layer_type() const override { return "conv"; }

    image<> weight_to_image() const {
        image<> img;
        const cnn_size_t border_width = 1;
        const auto pitch = 5/* weight_.width_ */ + border_width;
        const auto width = out_.depth_ * pitch + border_width;
        const auto height = in_.depth_ * pitch + border_width;
        const image<>::intensity_t bg_color = 255;
        const vec_t& W = *this->get_weights()[0];

        img.resize(width, height);
        img.fill(bg_color);

        auto minmax = std::minmax_element(W.begin(), W.end());

        for (cnn_size_t r = 0; r < in_.depth_; ++r) {
            for (cnn_size_t c = 0; c < out_.depth_; ++c) {
                if (!tbl_.is_connected(c, r)) continue;

                const auto top = r * pitch + border_width;
                const auto left = c * pitch + border_width;

                for (cnn_size_t y = 0; y < 5 /* weight_.height_ */; ++y) {
                    for (cnn_size_t x = 0; x < 5 /* weight_.width_ */; ++x) {
                        const float_t w = W[weight_.get_index(x, y, c * in_.depth_ + r)];

                        img.at(left + x, top + y)
                            = static_cast<image<>::intensity_t>(rescale(w, *minmax.first, *minmax.second, 0, 255));
                    }
                }
            }
        }
        return img;
    }

    virtual void set_worker_count(cnn_size_t worker_count) override {
        Base::set_worker_count(worker_count);
        conv_layer_worker_storage_.resize(worker_count);
        init();
    }

private:
    void conv_set_params(const shape3d& in,
                         cnn_size_t     w_width,
                         cnn_size_t     w_height,
                         cnn_size_t     outc,
                         padding        ptype,
                         bool           has_bias,
                         cnn_size_t     w_stride,
                         cnn_size_t     h_stride) {
        in_ = in;
        in_padded_ = shape3d(in_length(in.width_, w_width, ptype),
                             in_length(in.height_, w_height, ptype),
                             in.depth_);
        out_ = shape3d(conv_out_length(in.width_, w_width, w_stride, ptype),
                       conv_out_length(in.height_, w_height, h_stride, ptype),
                       outc);
        weight_ = shape3d(w_width, w_height, in.depth_ * outc);
        has_bias_ = has_bias;
        pad_type_ = ptype;
        w_stride_ = w_stride;
        h_stride_ = h_stride;
    }

    void init() {
        for (conv_layer_worker_specific_storage& cws : conv_layer_worker_storage_) {
            if (pad_type_ == padding::same) {
                cws.prev_out_buf_.resize(in_padded_.size(), float_t(0));
                cws.prev_delta_padded_.resize(in_padded_.size(), float_t(0));
            }
            else {
                cws.prev_out_buf_.clear();
            }
        }
        if (pad_type_ == padding::same) {
            prev_delta2_padded_.resize(in_padded_.size(), float_t(0));
        }
    }

    cnn_size_t in_length(cnn_size_t in_length, cnn_size_t window_size, padding pad_type) const {
        return pad_type == padding::same ? (in_length + window_size - 1) : in_length;
    }

    static cnn_size_t conv_out_length(cnn_size_t in_length, cnn_size_t window_size, cnn_size_t stride, padding pad_type) {
        return pad_type == padding::same ? (cnn_size_t)ceil((double)in_length / stride) : (cnn_size_t)ceil((double)(in_length - window_size + 1) / stride);
    }

    static cnn_size_t conv_out_dim(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t window_size, cnn_size_t w_stride, cnn_size_t h_stride, padding pad_type) {
        return conv_out_length(in_width, window_size, w_stride, pad_type) * conv_out_length(in_height, window_size, h_stride, pad_type);
    }

    cnn_size_t conv_out_dim(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t window_width, cnn_size_t window_height, cnn_size_t w_stride, cnn_size_t h_stride, padding pad_type) const {
        return conv_out_length(in_width, window_width, w_stride, pad_type) * conv_out_length(in_height, window_height, h_stride, pad_type);
    }

    void copy_and_unpad_delta(const vec_t& delta, vec_t& dst) {
        if (pad_type_ == padding::valid) {
            dst = delta;
        }
        else {
            for (cnn_size_t c = 0; c < in_.depth_; c++) {
                float_t *pdst = &dst[in_.get_index(0, 0, c)];
                const float_t *pin = &delta[in_padded_.get_index(5 /* weight_.width_ */ / 2, 5 /* weight_.height_ */ / 2, c)];

                for (cnn_size_t y = 0; y < in_.height_; y++, pdst += in_.width_, pin += in_padded_.width_) {
                    std::copy(pin, pin + in_.width_, pdst);
                }
            }
        }
    }

    void copy_and_pad_input(const vec_t& in, cnn_size_t worker_index) {
        conv_layer_worker_specific_storage& cws = conv_layer_worker_storage_[worker_index];

        vec_t* dst = &cws.prev_out_buf_;

        if (pad_type_ == padding::valid) {
            cws.prev_out_padded_ = &in;
        }
        else {
            // make padded version in order to avoid corner-case in fprop/bprop
            for (cnn_size_t c = 0; c < in_.depth_; c++) {
                float_t *pimg = &(*dst)[in_padded_.get_index(5 /* weight_.width_ */ / 2, 5 /* weight_.height_ */ / 2, c)];
                const float_t *pin = &in[in_.get_index(0, 0, c)];

                for (cnn_size_t y = 0; y < in_.height_; y++, pin += in_.width_, pimg += in_padded_.width_) {
                    std::copy(pin, pin + in_.width_, pimg);
                }
            }
            cws.prev_out_padded_ = &cws.prev_out_buf_;
        }
    }

    struct conv_layer_worker_specific_storage {
        const vec_t* prev_out_padded_;
        vec_t prev_out_buf_;
        vec_t prev_delta_padded_;
    };

    std::vector<conv_layer_worker_specific_storage> conv_layer_worker_storage_;

    vec_t  prev_delta2_padded_;

    connection_table tbl_;
    index3d<cnn_size_t> in_;
    index3d<cnn_size_t> in_padded_;
    index3d<cnn_size_t> out_;
    index3d<cnn_size_t> weight_;
    bool has_bias_;
    padding pad_type_;
    size_t w_stride_;
    size_t h_stride_;
};

} // namespace tiny_cnn
