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
#include <np/linalg/Inv.hpp>

#include <pd/core/frame/DataFrame/DataFrame.hpp>
#include <pd/core/frame/DataFrame/DataFrameStreamIo.hpp>

#include <sklearn/model_selection/train_test_split.hpp>

#include <optional>
#include <vector>

namespace sklearn {
    namespace linear_model {
        /* Ordinary least squares Linear Regression.

        LinearRegression fits a linear model with coefficients w = (w1, ..., wp) to minimize the residual sum of squares
         between the observed targets in the dataset, and the targets predicted by the linear approximation.

         http://web.vu.lt/mif/a.buteikis/wp-content/uploads/PE_Book/3-2-OLS.html
        */

        struct LinearRegressionParameters {
        };

        class LinearRegression {
        public:
            explicit LinearRegression(LinearRegressionParameters parameters = {})
                : m_parameters{parameters}, m_intercept{0.0} {
            }

            LinearRegression(const LinearRegression &) = default;

            LinearRegression(LinearRegression &&) noexcept = default;

            LinearRegression &operator=(const LinearRegression &) = default;

            LinearRegression &operator=(LinearRegression &&) noexcept = default;

            // Fit linear model.
            // X - training data of shape (n_samples, n_features)
            // y - target values of shape (n_samples,)
            // sample_weight array of shape (n_samples,), default=None
            template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2, typename DType3, typename Derived3, typename Storage3>
            void fit(const np::ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &X, const np::ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &y,
                     const np::ndarray::internal::NDArrayBase<DType3, Derived3, Storage3> &sample_weight) {
                if (X.ndim() != 2) {
                    throw std::runtime_error("2D array expected as X");
                }
                if (y.ndim() != 1) {
                    throw std::runtime_error("1D array expected as y");
                }
                if (X.shape()[0] != y.shape()[0]) {
                    throw std::runtime_error("Found input variables with inconsistent numbers of samples");
                }
                if (!sample_weight.empty()) {
                    if (sample_weight.ndim() != 1) {
                        throw std::runtime_error("Sample weight is not 1D array");
                    }
                    if (sample_weight.shape()[0] != y.shape()[0]) {
                        throw std::runtime_error("Sample weight has inconsistent number of samples");
                    }
                }
                auto ones = np::ones(np::Shape{X.shape()[0]});
                auto x = np::column_stack(ones, X);
                auto xT = x.transpose();
                if (sample_weight.empty()) {
                    m_coeffs = np::linalg::inv(xT.dot(x)).dot(xT).dot(y);
                } else {
                    auto W = np::diag1(sample_weight);
                    m_coeffs = np::linalg::inv(xT.dot(W).dot(x)).dot(xT).dot(W).dot(y);
                }
                m_coeff = m_coeffs["1:"];
                m_intercept = m_coeffs.get(0);

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

            [[nodiscard]] auto coef_() const {
                return m_coeff;
            }

            [[nodiscard]] auto coeffs_() const {
                return m_coeffs;
            }

            [[nodiscard]] auto intercept_() const {
                return m_intercept;
            }

        private:
            auto forwardStep(const auto &X) {
                return X.dot(m_coeff).add(m_intercept);
            }

            LinearRegressionParameters m_parameters;
            bool m_fitted{false};
            np::ndarray::array_dynamic::NDArrayDynamicIndexKeyType<np::float_> m_coeff;
            np::Array<np::float_> m_coeffs;
            np::float_ m_intercept;
        };

    }// namespace linear_model
}// namespace sklearn
