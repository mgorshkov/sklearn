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

#include <set>
#include <unordered_map>

#include <np/Array.hpp>
#include <pd/core/frame/DataFrame/DataFrame.hpp>

namespace sklearn {
    namespace metrics {
        /*
        Compute confusion matrix to evaluate the accuracy of a classification.
        By definition a confusion matrix C is such that C_ij is equal to the number of observations known to be in group i
        and predicted to be in group j.
        Thus in binary classification, the count of true negatives is C_0,0, false negatives is C_1,0, true positives is C_1,1,
        and false positives is C_0,1

        Parameters:
        y_true - array-like of shape (n_samples,) - Ground truth (correct) target values.
        y_pred - array-like of shape (n_samples,) - Estimated targets as returned by a classifier.
        */
        template<typename Array>
        struct ConfusionMatrixParameters {
            Array y_true;
            Array y_pred;
        };

        template<typename DType, np::Size Size = np::SIZE_DEFAULT>
        np::Array<np::Size> confusion_matrix(const ConfusionMatrixParameters<np::Array<DType, Size>> &params = {}) {
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
            std::set<DType> elements;
            for (np::Size i = 0; i < params.y_true.shape()[0]; ++i) {
                const auto &y_true = params.y_true[i];
                elements.insert(y_true.get(0));
                const auto &y_pred = params.y_pred[i];
                elements.insert(y_pred.get(0));
            }
            std::unordered_map<DType, np::Size> indices;
            np::Size size = 0;
            for (const auto &element: elements) {
                indices[element] = size++;
            }
            np::Array<np::Size> confusionMatrix{np::Shape{size, size}};
            for (np::Size i = 0; i < params.y_true.shape()[0]; ++i) {
                const auto &y_true = params.y_true[i];
                const auto &y_pred = params.y_pred[i];
                ++confusionMatrix[indices[y_true.get(0)]][indices[y_pred.get(0)]].get(0);
            }
            return confusionMatrix;
        }

        np::Array<np::Size> confusion_matrix(const ConfusionMatrixParameters<pd::DataFrame> &params = {});
        np::Array<np::Size> confusion_matrix(const ConfusionMatrixParameters<pd::Series> &params = {});

    }// namespace metrics
}// namespace sklearn