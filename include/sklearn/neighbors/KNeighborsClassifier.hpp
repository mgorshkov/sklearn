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
        template<typename DataType, typename TargetType>
        class KNeighborsClassifier {
        public:
            KNeighborsClassifier(np::Size n_neighbors = 5,
                                 WeightsType weights = WeightsType::kUniform,
                                 AlgorithmType algorithm = AlgorithmType::kAuto,
                                 int leaf_size = 30,
                                 int p = 2,
                                 metrics::DistanceMetricType metricType = metrics::DistanceMetricType::kMinkowski)
                : m_neighbors{n_neighbors}, m_p{p}, m_metricType{metricType} {
                if (algorithm != AlgorithmType::kAuto && algorithm != AlgorithmType::kBruteForce) {
                    throw std::runtime_error("Only BruteForce algorithm is currently implemented");
                }
                if (weights != WeightsType::kUniform) {
                    throw std::runtime_error("Only Uniform weights are currently implemented");
                }
                (void) leaf_size;
            }

            // Fit the k-nearest neighbors classifier from the training dataset.
            // Predict the class labels for the provided data.
            // X_fit - training data
            // y_fit - target values
            // X_predict - test samples.
            template<typename DerivedX_fit, typename StorageX_fit, typename DerivedY_fit, typename StorageY_fit, typename DerivedX_predict, typename StorageX_predict>
            Array<TargetType> fit_predict(const np::ndarray::internal::NDArrayBase<DataType, DerivedX_fit, StorageX_fit> &X_fit,
                                          const np::ndarray::internal::NDArrayBase<TargetType, DerivedY_fit, StorageY_fit> &y_fit,
                                          const np::ndarray::internal::NDArrayBase<DataType, DerivedX_predict, StorageX_predict> &X_predict) {
                auto metric = metrics::DistanceMetric<np::ndarray::internal::NDArrayBase<DataType, DerivedX_predict, StorageX_predict>,
                                                      np::ndarray::internal::NDArrayBase<DataType, DerivedX_fit, StorageX_fit>>::get_metric(m_metricType, m_p);
                auto distances = metric->pairwise(X_predict, X_fit);

                auto X_predict_shape{X_predict.shape()};
                np::Size totalSamples = X_predict_shape[0];
                np::Shape targetShape{totalSamples};
                Array<TargetType> pred{targetShape};

                for (np::Size sampleNumber = 0; sampleNumber < totalSamples; ++sampleNumber) {
                    pred.set(sampleNumber, predictSample(X_fit, y_fit, distances[sampleNumber]));
                }
                return pred;
            }

        private:
            template<typename DerivedX_fit, typename StorageX_fit, typename DerivedY_fit, typename StorageY_fit, typename StorageD>
            TargetType predictSample(const np::ndarray::internal::NDArrayBase<DataType, DerivedX_fit, StorageX_fit> &X_fit,
                                     const np::ndarray::internal::NDArrayBase<TargetType, DerivedY_fit, StorageY_fit> &y_fit,
                                     const np::ndarray::internal::NDArrayBase<DataType, DerivedX_fit, StorageD> &distance) const {
                std::vector<std::pair<DataType, TargetType>> distances;// distance
                // Calculating distances
                for (np::Size j = 0; j < X_fit.shape()[0]; ++j) {
                    auto p = std::make_pair(distance.get(j), y_fit.get(j));
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
                for (np::Size item = 0; item < m_neighbors; ++item) {
                    v.push_back(y_fit.get(idx[item]));// appending K nearest neighbors
                }
                // Making predictions
                np::Shape shape{m_neighbors};
                Array<TargetType> neighbors{v, shape};
                auto pred = mode<TargetType>(neighbors).first;
                return pred.get(0);
            }

            np::Size m_neighbors;
            int m_p;
            metrics::DistanceMetricType m_metricType;
        };
    }// namespace neighbors
}// namespace sklearn
