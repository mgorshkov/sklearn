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
#include <pd/core/frame/DataFrame/DataFrame.hpp>
#include <pd/core/series/Series/Series.hpp>

namespace sklearn {
    namespace metrics {
        template<typename Array>
        inline Array abs(const Array &array) {
            return np::abs(array);
        }

        inline pd::Series abs(const pd::Series &series) {
            pd::Series result{series.shape(), series.name()};
            for (np::Size i = 0; i < series.size(); ++i) {
                result.at(i) = std::abs(static_cast<np::float_>(series.at(i)));
            }
            return result;
        }

        inline pd::DataFrame abs(const pd::DataFrame &dataFrame) {
            pd::DataFrame result;
            for (const auto &columnName: dataFrame.columns().getIndex()) {
                const auto &series = dataFrame[columnName];
                result.append(abs(series));
            }
            return result;
        }

        template<typename Array>
        inline np::float_ sum(const Array &array) {
            return np::sum(array);
        }

        inline np::float_ sum(const pd::Series &series) {
            np::float_ result{};
            for (np::Size i = 0; i < series.size(); ++i) {
                result += static_cast<np::float_>(series.at(i));
            }
            return result;
        }

        inline np::float_ sum(const pd::DataFrame &dataFrame) {
            np::float_ result{};
            for (const auto &columnName: dataFrame.columns().getIndex()) {
                const auto &series = dataFrame[columnName];
                result += sum(series);
            }
            return result;
        }

        template<typename Array>
        inline np::float_ max(const Array &array) {
            return np::max(array);
        }

        inline np::float_ max(const pd::Series &series) {
            auto result = static_cast<np::float_>(series.at(0));
            for (np::Size i = 1; i < series.size(); ++i) {
                if (static_cast<np::float_>(series.at(i)) > result) {
                    result = static_cast<np::float_>(series.at(i));
                }
            }
            return result;
        }

        inline np::float_ max(const pd::DataFrame &dataFrame) {
            np::float_ result = max(dataFrame[pd::internal::Value{0}]);
            for (const auto &columnName: dataFrame.columns().getIndex()) {
                const auto &series = dataFrame[columnName];
                if (max(series) > result) {
                    result = max(series);
                }
            }
            return result;
        }
    }// namespace metrics
}// namespace sklearn
