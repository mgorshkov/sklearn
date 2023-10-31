/*
⚡ ML methods in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU

Copyright (c) 2023-2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <memory>

#include <cmath>
#include <np/Array.hpp>

#include <sklearn/Exception.hpp>
#include <sklearn/metrics/Distance.hpp>
#include <sklearn/metrics/Math.hpp>

namespace sklearn {
    namespace metrics {
        //max(|x - y|)
        template<typename ArrayX, typename ArrayY = ArrayX>
        class ChebyshevDistance : public Distance<ArrayX, ArrayY> {
        public:
            virtual np::Array<np::float_> pairwise(const ArrayX &X) {
                if (X.shape().size() != 2) {
                    throw sklearn::RuntimeError("2D array expected");
                }
                np::Size n_samples = X.shape()[0];
                np::Size n_features = X.shape()[1];
                np::Shape shape{n_samples, n_samples};
                np::Array<np::float_> result{shape};
                for (np::Size i = 0; i < n_samples; ++i) {
                    for (np::Size j = 0; j < n_samples; ++j) {
                        np::float_ maxDiff = 0;
                        for (np::Size k = 0; k < n_features; ++k) {
                            np::float_ diff = std::abs(static_cast<np::float_>(X.at(i, k) -
                                                                               X.at(j, k)));
                            if (diff > maxDiff) maxDiff = diff;
                        }
                        result.set(i * n_samples + j, maxDiff);
                    }
                }
                return result;
            }

            virtual np::Array<np::float_> pairwise(const ArrayX &X, const ArrayY &Y) {
                if (X.shape().size() != 2 || Y.shape().size() != 2) {
                    throw sklearn::RuntimeError("2D arrays expected");
                }
                if (X.shape()[1] != Y.shape()[1]) {
                    throw sklearn::RuntimeError("Number of features is different");
                }
                np::Size n_samples_X = X.shape()[0];
                np::Size n_samples_Y = Y.shape()[0];
                np::Size n_features = X.shape()[1];
                np::Shape shape{n_samples_X, n_samples_Y};
                np::Array<np::float_> result{shape};
                for (np::Size i = 0; i < n_samples_X; ++i) {
                    for (np::Size j = 0; j < n_samples_Y; ++j) {
                        np::float_ maxDiff = 0;
                        for (np::Size k = 0; k < n_features; ++k) {
                            np::float_ diff = std::abs(static_cast<np::float_>(X.at(i, k) -
                                                                               Y.at(j, k)));
                            if (diff > maxDiff) maxDiff = diff;
                        }
                        result.set(i * n_samples_Y + j, maxDiff);
                    }
                }
                return result;
            }
        };

        template<typename ArrayX, typename ArrayY = ArrayX>
        using ChebyshevDistancePtr = std::shared_ptr<ChebyshevDistance<ArrayX, ArrayY>>;
    }// namespace metrics
}// namespace sklearn
