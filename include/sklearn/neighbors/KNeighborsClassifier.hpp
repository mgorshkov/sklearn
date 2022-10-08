/*
ML Methods from scikit-learn library

Copyright (c) 2022 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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

#include <sklearn/neighbors/AlgorithmType.hpp>
#include <sklearn/neighbors/WeightsType.hpp>
#include <np/Array.hpp>
#include <scipy/stats/mode.hpp>
#include <sklearn/metrics/Distance.hpp>
#include <sklearn/metrics/DistanceMetric.hpp>
#include <sklearn/metrics/DistanceMetricType.hpp>

namespace sklearn {
    namespace neighbors {
        using Size = np::Size;

        template<typename DType = np::DTypeDefault, Size SizeT = np::SIZE_DEFAULT, Size... Sizes>
        using Array = np::Array<DType, SizeT, Sizes...>;

        using namespace scipy::stats;

        /// Classifier implementing the k-nearest neighbors vote.
        template <typename DataType, typename TargetType, np::Size... Sizes>
        class KNeighborsClassifier;

        template <typename DataType, typename TargetType, np::Size Count, np::Size FeatureCount>
        class KNeighborsClassifier<DataType, TargetType, Count, FeatureCount> {
        public:
            KNeighborsClassifier(int n_neighbors = 5,
                                 WeightsType weights = WeightsType::kUniform,
                                 AlgorithmType algorithm = AlgorithmType::kAuto,
                                 int leaf_size = 30,
                                 int p = 2,
                                 metrics::DistanceMetricType metricType = metrics::DistanceMetricType::kMinkowski)
                : m_neighbors{n_neighbors}
                , m_metric{metrics::DistanceMetric<DataType>::get_metric(metricType)}{
                if (algorithm != AlgorithmType::kAuto && algorithm != AlgorithmType::kBruteForce) {
                    throw std::runtime_error("Only BruteForce algorithm is currently implemented");
                }
                if (weights != WeightsType::kUniform) {
                    throw std::runtime_error("Only Uniform weights are currently implemented");
                }
                (void)leaf_size;
                (void)p;
            }

            // Fit the k-nearest neighbors classifier from the training dataset.
            // X - training data
            // y - target values
            void fit(const Array<DataType, Count, FeatureCount>& X, const Array<TargetType, Count>& y) {
                m_X = X.copy();
                m_y = y.copy();
            }

            // Predict the class labels for the provided data.
            // X - test samples.
            Array<TargetType, Count> predict(const Array<DataType, Count, FeatureCount>& X) {
                Array<TargetType, Count> y_pred;

                auto distances = m_metric->pairwise(X, m_X);

                for (np::Size i = 0; i < X.shape()[0]; ++i) {
                    auto pred = predictSample(distances[i]);
                    y_pred.push_back(pred);
                }
                return y_pred;
            }

        private:
            TargetType predictSample(const Array<DataType>& distance) const {
                std::vector<std::pair<DataType, TargetType>> distances;  // distance
                // Calculating distances
                for (np::Size j = 0; j < m_X.shape()[0]; ++j) {
                    auto p = std::make_pair(distance.get(j), m_y.get(j));
                    distances.push_back(p);
                }

                // initialize original index locations
                std::vector<std::size_t> idx(distances.size());
                iota(idx.begin(), idx.end(), 0);

                // sort indexes based on comparing values in v
                // using std::stable_sort instead of std::sort
                // to avoid unnecessary index re-orderings
                // when v contains elements of equal values
                std::stable_sort(idx.begin(), idx.end(), [&distances](size_t i1, size_t i2) {
                    return distances[i1].first < distances[i2].first;
                });
                // Getting k nearest neighbors
                std::vector<TargetType> v;
                for (auto item = 0; item < m_neighbors; ++item) {
                    v.push_back(m_y.get(idx[item]));  // appending K nearest neighbors
                }
                // Making predictions
                np::Shape shape{m_neighbors};
                Array<TargetType> neighbors{v, shape};
                auto pred = mode<TargetType>(neighbors).first;
                return pred.get(0);
            }

            int m_neighbors;
            metrics::DistancePtr<DataType> m_metric;
            const Array<DataType, Count, FeatureCount> m_X;
            const Array<TargetType, Count> m_y;
        };

        template <typename DataType, typename TargetType>
        class KNeighborsClassifier<DataType, TargetType> {
        public:
            KNeighborsClassifier(int n_neighbors = 5,
                                 WeightsType weights = WeightsType::kUniform,
                                 AlgorithmType algorithm = AlgorithmType::kAuto,
                                 int leaf_size = 30,
                                 int p = 2,
                                 metrics::DistanceMetricType metricType = metrics::DistanceMetricType::kMinkowski)
                    : m_neighbors{n_neighbors}
                    , m_metric{metrics::DistanceMetric<DataType>::get_metric(metricType, p)}{
                if (algorithm != AlgorithmType::kAuto && algorithm != AlgorithmType::kBruteForce) {
                    throw std::runtime_error("Only BruteForce algorithm is currently implemented");
                }
                if (weights != WeightsType::kUniform) {
                    throw std::runtime_error("Only Uniform weights are currently implemented");
                }
                (void)leaf_size;
            }

            // Fit the k-nearest neighbors classifier from the training dataset.
            // X - training data
            // y - target values
            void fit(const Array<DataType>& X, const Array<TargetType>& y) {
                m_X = X.copy();
                m_y = y.copy();
            }

            // Predict the class labels for the provided data.
            // X - test samples.
            Array<TargetType> predict(const Array<DataType>& X) {
                std::vector<TargetType> y_pred;
                y_pred.resize(X.shape()[0]);

                auto distances = m_metric->pairwise(X, m_X);

                for (np::Size i = 0; i < X.shape()[0]; ++i) {
                    auto pred = predictSample(distances[i]);
                    y_pred.push_back(pred);
                }
                return Array<TargetType>{y_pred};
            }

        private:
            TargetType predictSample(const Array<DataType>& distance) const {
                std::vector<std::pair<DataType, TargetType>> distances;  // distance
                // Calculating distances
                for (np::Size j = 0; j < m_X.shape()[0]; ++j) {
                    auto p = std::make_pair(distance.get(j), m_y.get(j));
                    distances.push_back(p);
                }

                // initialize original index locations
                std::vector<std::size_t> idx(distances.size());
                iota(idx.begin(), idx.end(), 0);

                // sort indexes based on comparing values in v
                // using std::stable_sort instead of std::sort
                // to avoid unnecessary index re-orderings
                // when v contains elements of equal values
                std::stable_sort(idx.begin(), idx.end(), [&distances](size_t i1, size_t i2) {
                    return distances[i1].first < distances[i2].first;
                });
                // Getting k nearest neighbors
                std::vector<TargetType> v;
                for (auto item = 0; item < m_neighbors; ++item) {
                    v.push_back(m_y.get(idx[item]));  // appending K nearest neighbors
                }
                // Making predictions
                np::Shape shape{m_neighbors};
                Array<TargetType> neighbors{v, shape};
                auto pred = mode<TargetType>(neighbors).first;
                return pred.get(0);
            }

            int m_neighbors;
            metrics::DistancePtr<DataType> m_metric;
            Array<DataType> m_X;
            Array<TargetType> m_y;
        };
    }
}
