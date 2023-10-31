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

#include <np/Array.hpp>
#include <np/Constants.hpp>
#include <np/Copy.hpp>
#include <np/DType.hpp>
#include <np/Manip.hpp>
#include <np/Math.hpp>
#include <np/linalg/Cholesky.hpp>
#include <np/linalg/Inv.hpp>
#include <scipy/linalg/lstsq.hpp>

#include <pd/core/frame/DataFrame/DataFrame.hpp>
#include <pd/core/frame/DataFrame/DataFrameStreamIo.hpp>

#include <sklearn/Exception.hpp>
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
            // @params fit_intercept : bool, default=true
            // Whether to calculate the intercept for this model. If set
            // to false, no intercept will be used in calculations (i.e. data is expected to be centered).
            bool fit_intercept = true;
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
            void fit(np::ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &X, np::ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &y,
                     const np::ndarray::internal::NDArrayBase<DType3, Derived3, Storage3> &sample_weight) {
                if (X.ndim() != 2) {
                    throw sklearn::RuntimeError("2D array expected as X");
                }
                if (y.ndim() != 1) {
                    throw sklearn::RuntimeError("1D array expected as y");
                }
                if (X.shape()[0] != y.shape()[0]) {
                    throw sklearn::RuntimeError("Found input variables with inconsistent numbers of samples");
                }
                if (!sample_weight.empty()) {
                    if (sample_weight.ndim() != 1) {
                        throw sklearn::RuntimeError("Sample weight is not 1D array");
                    }
                    if (sample_weight.shape()[0] != y.shape()[0]) {
                        throw sklearn::RuntimeError("Sample weight has inconsistent number of samples");
                    }
                }
                // Add column of ones if intercept is needed
                np::Array<np::float_> X_aug;
                auto n_samples = X.shape()[0];
                if (m_parameters.fit_intercept) {
                    auto n_features = X.shape()[1];
                    auto n_cols = n_features + 1;
                    // Manual construction of X_aug = [ones, X]
                    // Storage is row-major
                    X_aug = np::Array<np::float_>{np::Shape{n_samples, n_cols}};
                    auto *X_aug_data = X_aug.data();
                    if constexpr (Storage1::is_contiguous) {
                        const auto *X_data = X.data();
                        // Use memcpy for each row to leverage libc's optimized copy
                        const auto row_bytes = n_features * sizeof(np::float_);
                        const auto aug_row_stride = n_cols;
                        for (np::Size i = 0; i < n_samples; ++i) {
                            auto *row = X_aug_data + i * aug_row_stride;
                            row[0] = 1.0;
                            std::memcpy(row + 1, X_data + i * n_features, row_bytes);
                        }
                    } else {
                        for (np::Size i = 0; i < n_samples; ++i) {
                            auto *row = X_aug_data + i * n_cols;
                            row[0] = 1.0;
                            for (np::Size j = 0; j < n_features; ++j) {
                                row[j + 1] = X.get(i * n_features + j);
                            }
                        }
                    }
                } else {
                    X_aug = X.copy();
                }
                np::Array<np::float_> y_work = y.copy();

                // Solve using least squares
                // When weights are provided, use weighted Cholesky directly
                // to avoid an extra pass over the data (scaling X and y by sqrt(w))
                np::Array<np::float_> coeff_aug;
                if (!sample_weight.empty()) {
                    // Pass weights directly to the weighted Cholesky solver
                    // This avoids scaling the full 100k×n matrix by sqrt(w)
                    coeff_aug = np::linalg::lstsq_weighted_cholesky(X_aug, sample_weight, y_work);
                } else {
                    coeff_aug = scipy::linalg::lstsq(X_aug, y_work);
                }

                // Store augmented coefficients
                m_coeffs = coeff_aug;
                // Split intercept and coefficients
                if (m_parameters.fit_intercept) {
                    m_intercept = coeff_aug.get(0);
                    m_coeff = m_coeffs["1:"];
                } else {
                    m_intercept = 0.0;
                    m_coeff = m_coeffs["0:"];
                }
                m_fitted = true;
            }

            // Predict using the linear model.
            // X - test samples.
            auto predict(const auto &X) {
                if (!m_fitted) {
                    throw sklearn::NotFittedError(
                            "This LinearRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.");
                }
                if (X.ndim() != 2) {
                    throw sklearn::RuntimeError("Expected 2D array.");
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
