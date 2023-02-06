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

#include <sklearn/model_selection/train_test_split.hpp>

#include <optional>
#include <vector>

namespace sklearn {
    namespace preprocessing {
        struct StandardScalerParameters {
            /// If false, try to avoid a copy and do inplace scaling instead.
            /// This is not guaranteed to always work inplace;
            /// e.g. if the data is not a NumPy array or scipy.sparse CSR matrix, a copy may still be returned.
            bool copy{true};
            /// If true, center the data before scaling. This does not work (and will raise an exception) when
            /// attempted on sparse matrices, because centering them entails building a dense matrix which in common
            /// use cases is likely to be too large to fit in memory.
            bool with_mean{true};
            /// if true, scale the data to unit variance (or equivalently, unit standard deviation).
            bool with_std{true};
        };

        /* Standardize features by removing the mean and scaling to unit variance.
        The standard score of a sample x is calculated as:
        z = (x - u) / s
        where u is the mean of the training samples or zero if with_mean=false, and s is the standard deviation of the training samples
         or one if with_std=false.
        Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set.
         Mean and standard deviation are then stored to be used on later data using transform.
        Standardization of a dataset is a common requirement for many machine learning estimators:
         they might behave badly if the individual features do not more or less look like standard normally distributed data
         (e.g. Gaussian with 0 mean and unit variance).

        For instance many elements used in the objective function of a learning algorithm
         (such as the RBF kernel of Support Vector Machines or the L1 and L2 regularizers of linear models)
         assume that all features are centered around 0 and have variance in the same order.
         If a feature has a variance that is orders of magnitude larger than others,
         it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.
        This scaler can also be applied to sparse CSR or CSC matrices by passing with_mean=false to avoid breaking the sparsity structure of the data.
         */
        template<typename DType, np::Size SizeT>
        class StandardScaler;

        template<typename DType = np::DTypeDefault, np::Size SizeT = np::SIZE_DEFAULT>
        class StandardScaler {
        public:
            explicit StandardScaler(StandardScalerParameters parameters = StandardScalerParameters{})
                : m_parameters{parameters} {
            }

            StandardScaler &fit(const np::Array<DType> &array) {
                if (array.shape().size() != 2) {
                    throw std::runtime_error("Array must be 2-dimensional");
                }
                np::Size size = array.shape()[1];
                if (m_parameters.with_mean) {
                    m_mean = np::Array<np::float_>{np::Shape{size}};
                    for (np::Size i = 0; i < size; ++i) {
                        m_mean.set(i, array[":," + std::to_string(i)].mean());
                    }
                }
                if (m_parameters.with_std) {
                    m_var = np::Array<np::float_>{np::Shape{size}};
                    for (np::Size i = 0; i < size; ++i) {
                        m_var.set(i, array[":," + std::to_string(i)].var());
                    }
                    m_scale = m_var.sqrt();
                    if (std::all_of(m_scale.cbegin(), m_scale.cend(), [](auto element) { return element == 0; })) {
                        for (np::Size i = 0; i < m_scale.size(); ++i) {
                            m_scale.set(i, 1);
                        }
                    }
                }
                return *this;
            }

            StandardScaler &fit(const pd::DataFrame &dataFrame) {
                if (dataFrame.shape().size() != 2) {
                    throw std::runtime_error("DataFrame must be 2-dimensional");
                }
                np::Size size = dataFrame.shape()[1];
                if (m_parameters.with_mean) {
                    m_mean = np::Array<np::float_>{np::Shape{size}};
                    for (np::Size i = 0; i < size; ++i) {
                        m_mean.set(i, dataFrame[pd::internal::Value{i}].mean());
                    }
                }
                if (m_parameters.with_std) {
                    m_var = np::Array<np::float_>{np::Shape{size}};
                    for (np::Size i = 0; i < size; ++i) {
                        m_var.set(i, dataFrame[pd::internal::Value{i}].var());
                    }
                    m_scale = m_var.sqrt();
                    if (std::all_of(m_scale.cbegin(), m_scale.cend(), [](auto element) { return element == 0; })) {
                        for (np::Size i = 0; i < m_scale.size(); ++i) {
                            m_scale.set(i, 1);
                        }
                    }
                }
                return *this;
            }

            np::Array<DType> transform(const np::Array<DType> &array) {
                return (array - m_mean) / m_scale;
            }

            pd::DataFrame transform(const pd::DataFrame &dataFrame) {
                return dataFrame.subtractVector<np::float_>(m_mean).divideVector<np::float_>(m_scale);
            }

            np::Array<DType> fit_transform(const np::Array<DType> &array) {
                fit(array);
                return transform(array);
            }

            pd::DataFrame fit_transform(const pd::DataFrame &dataFrame) {
                fit(dataFrame);
                return transform(dataFrame);
            }

            const np::Array<DType> &mean_() const {
                return m_mean;
            }

            const np::Array<DType> &var_() const {
                return m_var;
            }

            const np::Array<DType> &scale_() const {
                return m_scale;
            }

        private:
            StandardScalerParameters m_parameters;
            np::Array<np::float_> m_mean;
            np::Array<np::float_> m_var;
            np::Array<np::float_> m_scale;
        };

    }// namespace preprocessing
}// namespace sklearn
