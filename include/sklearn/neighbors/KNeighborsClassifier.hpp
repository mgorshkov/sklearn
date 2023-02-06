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

#include <numeric>

#include <np/Array.hpp>
#include <scipy/stats/mode.hpp>
#include <sklearn/metrics/Distance.hpp>
#include <sklearn/metrics/DistanceMetric.hpp>
#include <sklearn/metrics/DistanceMetricType.hpp>
#include <sklearn/neighbors/AlgorithmType.hpp>
#include <sklearn/neighbors/WeightsType.hpp>

namespace sklearn {
    namespace neighbors {
        using Size = np::Size;

        template<typename DType = np::DTypeDefault, Size SizeT = np::SIZE_DEFAULT>
        using Array = np::Array<DType, SizeT>;

        using namespace scipy::stats;

        /// Classifier implementing the k-nearest neighbors vote.
        struct KNeighborsClassifierParameters {
            np::Size n_neighbors{5};
            WeightsType weights{WeightsType::kUniform};
            AlgorithmType algorithm{AlgorithmType::kAuto};
            int leaf_size{30};
            int p{2};
            metrics::DistanceMetricType metric{metrics::DistanceMetricType::kMinkowski};
        };

        template<typename DataType, typename TargetType = DataType>
        class KNeighborsClassifier;

        template<typename DataType, typename TargetType>
        class KNeighborsClassifier {
        public:
            explicit KNeighborsClassifier(KNeighborsClassifierParameters parameters = {})
                : m_parameters{parameters} {
                if (m_parameters.algorithm != AlgorithmType::kAuto && m_parameters.algorithm != AlgorithmType::kBruteForce) {
                    throw std::runtime_error("Only BruteForce algorithm is currently implemented");
                }
                if (m_parameters.weights != WeightsType::kUniform) {
                    throw std::runtime_error("Only Uniform weights are currently implemented");
                }
            }

            KNeighborsClassifier(const KNeighborsClassifier &) = default;
            KNeighborsClassifier(KNeighborsClassifier &&) noexcept = default;

            KNeighborsClassifier &operator=(const KNeighborsClassifier &) = default;
            KNeighborsClassifier &operator=(KNeighborsClassifier &&) noexcept = default;

            // Fit the k-nearest neighbors classifier from the training dataset.
            // X - training data
            // y - target values
            template<typename ArrayDataType, typename ArrayTargetType>
            void fit(const ArrayDataType &X, const ArrayTargetType &y) {
                m_X = X.copy();
                m_y = y.copy();
                m_fitted = true;
            }

            // Predict the class labels for the provided data.
            // X - test samples.
            template<typename ArrayPredictType>
            Array<TargetType> predict(const ArrayPredictType &X) {
                if (!m_fitted) {
                    throw std::runtime_error("This KNeighborsClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.");
                }
                auto metric = metrics::DistanceMetric<ArrayPredictType, Array<DataType>>::get_metric(m_parameters.metric, m_parameters.p);
                auto distances = metric->pairwise(X, m_X);
                np::Size totalSamples = X.shape()[0];
                Array<TargetType> pred{np::Shape{totalSamples}};

                for (np::Size sample = 0; sample < totalSamples; ++sample) {
                    pred.set(sample, predictSample(distances[sample]));
                }
                return pred;
            }

        private:
            template<typename DerivedX_fit, typename StorageD>
            TargetType predictSample(const np::ndarray::internal::NDArrayBase<DataType, DerivedX_fit, StorageD> &distance) const {
                std::priority_queue<std::pair<DataType, TargetType>, std::vector<std::pair<DataType, TargetType>>, std::greater<std::pair<DataType, TargetType>>> distances;
                for (np::Size j = 0; j < m_X.shape()[0]; ++j) {
                    auto p = std::make_pair(distance.get(j), m_y.get(j));
                    distances.push(p);
                }
                std::vector<TargetType> v;
                for (np::Size i = 0; i < m_parameters.n_neighbors; ++i) {
                    v.push_back(distances.top().second);
                    distances.pop();
                }
                np::Shape shape{m_parameters.n_neighbors};
                Array<TargetType> neighbors{std::move(v), shape};
                return mode<TargetType>(neighbors).first.get(0);
            }

            KNeighborsClassifierParameters m_parameters;
            Array<DataType> m_X;
            Array<TargetType> m_y;
            bool m_fitted{false};
        };

        template<>
        class KNeighborsClassifier<pd::DataFrame> {
        public:
            explicit KNeighborsClassifier(KNeighborsClassifierParameters parameters = {})
                : m_parameters{parameters} {
                if (m_parameters.algorithm != AlgorithmType::kAuto && m_parameters.algorithm != AlgorithmType::kBruteForce) {
                    throw std::runtime_error("Only BruteForce algorithm is currently implemented");
                }
                if (m_parameters.weights != WeightsType::kUniform) {
                    throw std::runtime_error("Only Uniform weights are currently implemented");
                }
            }

            KNeighborsClassifier(const KNeighborsClassifier &) = default;
            KNeighborsClassifier(KNeighborsClassifier &&) = default;

            KNeighborsClassifier &operator=(const KNeighborsClassifier &) = default;
            KNeighborsClassifier &operator=(KNeighborsClassifier &&) noexcept = default;

            // Fit the k-nearest neighbors classifier from the training dataset.
            // X - training data
            // y - target values
            void fit(const pd::DataFrame &X, const pd::DataFrame &y) {
                m_X = X;
                m_y = y;
                m_fitted = true;
            }

            // Predict the class labels for the provided data.
            // X - test samples.
            pd::DataFrame predict(const pd::DataFrame &X) {
                if (!m_fitted) {
                    throw std::runtime_error("This KNeighborsClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.");
                }
                auto metric = metrics::DistanceMetric<pd::DataFrame, pd::DataFrame>::get_metric(m_parameters.metric, m_parameters.p);
                auto distances = metric->pairwise(X, m_X);

                np::Size totalSamples = X.shape()[0];
                np::Array<pd::internal::Value> array{np::Shape{totalSamples}};
                for (np::Size sample = 0; sample < totalSamples; ++sample) {
                    array.set(sample, predictSample(distances[sample]));
                }
                return pd::DataFrame{array};
            }

        private:
            template<typename DerivedX_fit, typename StorageD>
            [[nodiscard]] pd::internal::Value predictSample(const np::ndarray::internal::NDArrayBase<np::float_, DerivedX_fit, StorageD> &distance) const {
                std::priority_queue<std::pair<np::float_, pd::internal::Value>, std::vector<std::pair<np::float_, pd::internal::Value>>, std::greater<std::pair<np::float_, pd::internal::Value>>> distances;
                for (np::Size j = 0; j < m_X.shape()[0]; ++j) {
                    auto p = std::make_pair(distance.get(j), m_y.at(j, 0));
                    distances.push(p);
                }

                std::vector<pd::internal::Value> v;
                for (np::Size i = 0; i < m_parameters.n_neighbors; ++i) {
                    v.push_back(distances.top().second);
                    distances.pop();
                }

                np::Shape shape{m_parameters.n_neighbors};
                Array<pd::internal::Value> neighbors{v, shape};
                return mode<pd::internal::Value>(neighbors).first.get(0);
            }

            KNeighborsClassifierParameters m_parameters;
            pd::DataFrame m_X;
            pd::DataFrame m_y;
            bool m_fitted{false};
        };

    }// namespace neighbors
}// namespace sklearn
