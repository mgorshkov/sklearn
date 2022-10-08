/*
ML Methods from scikit-learn library

Copyright (c) 2022 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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
        // np.absolute(X - Y)

        template <typename DType, np::Size... Sizes>
        class ManhattanDistance;

        template <typename DType, np::Size NSamples1, np::Size NSamples2, np::Size NFeatures>
        class ManhattanDistance<DType, NSamples1, NSamples2, NFeatures> : public Distance<DType, NSamples1, NSamples2, NFeatures> {
        public:
            virtual np::Array<DType, NSamples1, NSamples1> pairwise(const np::Array<DType, NSamples1, NFeatures>& X) {
                return sqrt(dot(X, X) - 2 * dot(X, X) + dot(X, X));
            }

            virtual np::Array<DType, NSamples1, NSamples2> pairwise(const np::Array<DType, NSamples1, NFeatures>& X, const np::Array<DType, NSamples2, NFeatures>& Y) {
                //dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
                return sqrt(dot(X, X) - 2 * dot(X, Y) + dot(Y, Y));
            }
        };

        template <typename DType>
        class ManhattanDistance<DType> : public Distance<DType> {
        public:
            virtual np::Array<DType> pairwise(const np::Array<DType>& X) {
                if (X.shape().size() != 2) {
                    throw std::runtime_error("2D array expected");
                }
                np::Shape sh{X.shape()[0], X.shape()[0]};
                std::vector<DType> v;
                v.resize(X.shape()[0] * X.shape()[0]);
                for (np::Size i = 0; i < X.shape()[0]; ++i) {
                    for (np::Size j = 0; j < X.shape()[0]; ++j) {
                        v[i * X.shape()[0] + j] = sqrt(X[i].dot(X[i]) - 2 * X[i].dot(X[j]) + X[j].dot(X[j]));
                    }
                }
                return np::Array<DType>{v, sh};
            }

            virtual np::Array<DType> pairwise(const np::Array<DType>& X, const np::Array<DType>& Y) {
                if (X.shape().size() != 2 || Y.shape().size() != 2) {
                    throw std::runtime_error("2D arrays expected");
                }
                if (X.shape()[1] != Y.shape()[1]) {
                    throw std::runtime_error("Number of features is different");
                }
                np::Shape sh{X.shape()[0], Y.shape()[0]};
                std::vector<DType> v;
                v.resize(X.shape()[0] * Y.shape()[0]);
                for (np::Size i = 0; i < X.shape()[0]; ++i) {
                    for (np::Size j = 0; j < Y.shape()[0]; ++j) {
                        v[i * Y.shape()[0] + j] = sqrt(X[i].dot(X[i]) - 2 * X[i].dot(Y[j]) + Y[j].dot(Y[j]));
                    }
                }
                return np::Array<DType>{v, sh};
            }
        };

        template <typename DType>
        using ManhattanDistancePtr = std::shared_ptr<ManhattanDistance<DType>>;
    }
}
