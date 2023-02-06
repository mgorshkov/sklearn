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

#include <sklearn/metrics/confusion_matrix.hpp>

namespace sklearn {
    namespace metrics {
        np::Array<np::Size> confusion_matrix(const ConfusionMatrixParameters<pd::DataFrame> &params) {
            if (params.y_true.empty() && params.y_pred.empty()) {
                return np::Array<np::Size>{};
            }
            if (params.y_true.ndim() != 1 || params.y_pred.ndim() != 1) {
                throw std::runtime_error("Arrays must be 1-dimensional");
            }
            if (params.y_true.size() != params.y_pred.size()) {
                throw std::runtime_error("Arrays must be of equal sizes");
            }
            // multiclass targets, sorted order
            std::set<pd::internal::Value> elements;
            for (np::Size i = 0; i < params.y_true.shape()[0]; ++i) {
                elements.insert(params.y_true.iloc(i, 0));
                elements.insert(params.y_pred.iloc(i, 0));
            }
            std::unordered_map<pd::internal::Value, np::Size> indices;
            np::Size size = 0;
            for (const auto &element: elements) {
                indices[element] = size++;
            }
            np::Array<np::Size> confusionMatrix{np::Shape{size, size}};
            for (np::Size i = 0; i < params.y_true.shape()[0]; ++i) {
                ++confusionMatrix[indices[params.y_true.iloc(i, 0)]][indices[params.y_pred.iloc(i, 0)]].get(0);
            }
            return confusionMatrix;
        }

        np::Array<np::Size> confusion_matrix(const ConfusionMatrixParameters<pd::Series> &params) {
            if (params.y_true.empty() && params.y_pred.empty()) {
                return np::Array<np::Size>{};
            }
            if (params.y_true.ndim() != 1 || params.y_pred.ndim() != 1) {
                throw std::runtime_error("Arrays must be 1-dimensional");
            }
            if (params.y_true.size() != params.y_pred.size()) {
                throw std::runtime_error("Arrays must be of equal sizes");
            }
            // multiclass targets, sorted order
            std::set<pd::internal::Value> elements;
            for (np::Size i = 0; i < params.y_true.shape()[0]; ++i) {
                elements.insert(params.y_true.iloc(i));
                elements.insert(params.y_pred.iloc(i));
            }
            std::unordered_map<pd::internal::Value, np::Size> indices;
            np::Size size = 0;
            for (const auto &element: elements) {
                indices[element] = size++;
            }
            np::Array<np::Size> confusionMatrix{np::Shape{size, size}};
            for (np::Size i = 0; i < params.y_true.shape()[0]; ++i) {
                ++confusionMatrix[indices[params.y_true.iloc(i)]][indices[params.y_pred.iloc(i)]].get(0);
            }
            return confusionMatrix;
        }
    }// namespace metrics
}// namespace sklearn