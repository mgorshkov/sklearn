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
#include <np/Constants.hpp>
#include <np/DType.hpp>

#include <pd/core/frame/DataFrame/DataFrame.hpp>
#include <pd/core/frame/DataFrame/DataFrameStreamIo.hpp>

#include <sklearn/model_selection/train_test_split.hpp>

#include <optional>
#include <vector>

namespace sklearn {
    namespace linear_model {
        /* Linear model fitted by minimizing a regularized empirical loss with SGD.
        SGD stands for Stochastic Gradient Descent: the gradient of the loss is estimated each sample at a time and the model
         is updated along the way with a decreasing strength schedule (aka learning rate).
        The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using
         either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net).
         If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow
         for learning sparse models and achieve online feature selection.
        This implementation works with data represented as dense numpy arrays of floating point values for the features.
        */

        struct SGDRegressorParameters {
        };

        template<typename ArrayDataType = np::Array<np::float_>, typename ArrayTargetType = ArrayDataType>
        class SGDRegressor {
        public:
            explicit SGDRegressor(SGDRegressorParameters parameters = {})
                : m_parameters{parameters}, m_intercept{0.0} {
            }

            SGDRegressor(const SGDRegressor &) = default;

            SGDRegressor(SGDRegressor &&) noexcept = default;

            SGDRegressor &operator=(const SGDRegressor &) = default;

            SGDRegressor &operator=(SGDRegressor &&) noexcept = default;

            // Fit linear model with Stochastic Gradient Descent.
            // X - training data
            // y - target values
            void fit(const ArrayDataType &X, const ArrayTargetType &y) {
                // initialize m_coeff and m_intercept randomly
                m_coeff = np::zeros(np::Shape{X.shape()[1]});
                m_intercept = 0.0;

                for (std::size_t i = 0; i < m_iterations; ++i) {
                    auto pred = forwardStep(X);
                    auto derivatives = backPropagation(X, y, pred);
                    if (!updateCoeff(derivatives))
                        break;
                }

                m_fitted = true;
            }

            // Predict using the linear model.
            // X - test samples.
            auto predict(const auto &X) {
                if (!m_fitted) {
                    throw std::runtime_error(
                            "This LinearRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.");
                }
                if (X.ndim() != 2) {
                    throw std::runtime_error("Expected 2D array.");
                }
                return forwardStep(X);
            }

            [[nodiscard]] np::Array<np::float_> coef_() const {
                return m_coeff;
            }

            [[nodiscard]] np::float_ intercept_() const {
                return m_intercept;
            }

        private:
            struct Derivatives {
                np::Array<np::float_> coeff;
                np::float_ intercept{0.0};
            };

            auto forwardStep(const ArrayDataType &X) {
                auto d = X.dot(m_coeff);
                return d.add(m_intercept);
            }

            Derivatives backPropagation(const ArrayDataType &trainInput, const ArrayTargetType &trainOutput, const auto &pred) {
                auto df = pred.subtract(trainOutput);
                return Derivatives{trainInput.transpose().dot(df).divide(trainInput.shape()[0]), df.sum() / trainInput.shape()[0]};
            }

            bool updateCoeff(const Derivatives &derivatives) {
                auto delta = derivatives.coeff.multiply(m_learningRate);
                m_coeff = m_coeff.subtract(delta);
                m_intercept -= derivatives.intercept * m_learningRate;
                return true;
            }

            SGDRegressorParameters m_parameters;
            bool m_fitted{false};
            np::Array<np::float_> m_coeff;
            np::float_ m_intercept;

            constexpr static const std::size_t m_iterations = 1000;
            constexpr static const np::float_ m_learningRate = 0.001;
        };

    }// namespace linear_model
}// namespace sklearn
