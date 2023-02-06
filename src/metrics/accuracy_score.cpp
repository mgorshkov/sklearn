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

#include <sklearn/metrics/accuracy_score.hpp>

namespace sklearn {
    namespace metrics {
        np::float_ accuracy_score(const pd::DataFrame &y_true, const pd::DataFrame &y_pred) {
            if (y_true.empty() && y_pred.empty()) {
                return 1.0;
            }
            if (y_true.ndim() != y_pred.ndim()) {
                throw std::runtime_error("Arrays must be of equal dimensions");
            }
            if (y_true.size() != y_pred.size()) {
                throw std::runtime_error("Arrays must be of equal sizes");
            }

            np::Size equal{0};
            for (np::Size i = 0; i < y_true.shape()[0]; ++i) {
                if (y_true.iloc(i) == y_pred.iloc(i)) {
                    ++equal;
                }
            }

            return static_cast<np::float_>(equal) / static_cast<np::float_>(y_true.shape()[0]);
        }

        np::float_ accuracy_score(const pd::Series &y_true, const pd::Series &y_pred) {
            if (y_true.empty() && y_pred.empty()) {
                return 1.0;
            }
            if (y_true.ndim() != y_pred.ndim()) {
                throw std::runtime_error("Arrays must be of equal dimensions");
            }
            if (y_true.size() != y_pred.size()) {
                throw std::runtime_error("Arrays must be of equal sizes");
            }

            np::Size equal{0};
            for (np::Size i = 0; i < y_true.shape()[0]; ++i) {
                if (y_true[i] == y_pred[i]) {
                    ++equal;
                }
            }

            return static_cast<np::float_>(equal) / static_cast<np::float_>(y_true.shape()[0]);
        }
    }// namespace metrics
}// namespace sklearn