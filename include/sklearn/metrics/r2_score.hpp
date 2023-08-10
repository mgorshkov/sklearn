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

#include <np/Array.hpp>
#include <sklearn/metrics/confusion_matrix.hpp>
#include <unordered_map>

namespace sklearn {
    namespace metrics {
        template<typename ArrayX, typename ArrayY = ArrayX>
        struct R2ScoreParameters {
            ArrayX y_true;
            ArrayY y_pred;
        };

        template<typename ArrayX, typename ArrayY = ArrayX>
        np::float_ r2_score(const R2ScoreParameters<ArrayX, ArrayY> &params = {}) {
            if (params.y_true.empty() && params.y_pred.empty()) {
                return 1.0;
            }
            if (params.y_true.ndim() != params.y_pred.ndim()) {
                throw std::runtime_error("Arrays must be of equal dimensions");
            }
            if (params.y_true.size() != params.y_pred.size()) {
                throw std::runtime_error("Arrays must be of equal sizes");
            }

            auto y_avg = params.y_true.mean();

            np::float_ numerator = 0.0;
            np::float_ denominator = 0.0;
            for (std::size_t i = 0; i < params.y_pred.size(); ++i) {
                numerator += (params.y_true.get(i) - params.y_pred.get(i)) * (params.y_true.get(i) - params.y_pred.get(i));
                denominator += (params.y_true.get(i) - y_avg) * (params.y_true.get(i) - y_avg);
            }
            np::float_ r2 = 1 - (denominator == 0 ? 0.0 : numerator / denominator);
            return r2;
        }

    }// namespace metrics
}// namespace sklearn