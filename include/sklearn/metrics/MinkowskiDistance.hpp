/*
ML Methods from scikit-learn library

Copyright (c) 2023 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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

#include <np/Array.hpp>

#include <sklearn/metrics/Distance.hpp>

namespace sklearn {
    namespace metrics {
        // sum(|x - y|^p)^(1/p)
        template<typename ArrayX, typename ArrayY = ArrayX>
        class MinkowskiDistance : public Distance<ArrayX, ArrayY> {
        public:
            explicit MinkowskiDistance(int p = 2) : m_p{p} {
                if (p != 2) {
                    throw std::runtime_error("Only p == 2 is currently supported");
                }
            }

            virtual np::Array<np::float_> pairwise(const ArrayX &X) {
                if (X.shape().size() != 2) {
                    throw std::runtime_error("2D array expected");
                }
                np::Shape shape{X.shape()[0], X.shape()[0]};
                np::Array<np::float_> result{shape};
                for (np::Size i = 0; i < shape[0]; ++i) {
                    for (np::Size j = 0; j < shape[1]; ++j) {
                        result.set(i * shape[1] + j, std::sqrt(static_cast<np::float_>(X[i].dot(X[i])) - 2 * static_cast<np::float_>(X[i].dot(X[j])) + static_cast<np::float_>(X[j].dot(X[j]))));
                    }
                }
                return result;
            }

            virtual np::Array<np::float_> pairwise(const ArrayX &X, const ArrayY &Y) {
                if (X.shape().size() != 2 || Y.shape().size() != 2) {
                    throw std::runtime_error("2D arrays expected");
                }
                if (X.shape()[1] != Y.shape()[1]) {
                    throw std::runtime_error("Number of features is different");
                }
                np::Shape shape{X.shape()[0], Y.shape()[0]};
                np::Array<np::float_> result{shape};
                for (np::Size i = 0; i < shape[0]; ++i) {
                    for (np::Size j = 0; j < shape[1]; ++j) {
                        result.set(i * shape[1] + j, std::sqrt(static_cast<np::float_>(X[i].dot(X[i])) - 2 * static_cast<np::float_>(X[i].dot(Y[j])) + static_cast<np::float_>(Y[j].dot(Y[j]))));
                    }
                }
                return result;
            }

        private:
            int m_p;
        };

        template<typename ArrayX, typename ArrayY = ArrayX>
        using MinkowskiDistancePtr = std::shared_ptr<MinkowskiDistance<ArrayX, ArrayY>>;
    }// namespace metrics
}// namespace sklearn
