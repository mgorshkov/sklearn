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

#include <sklearn/metrics/ChebyshevDistance.hpp>
#include <sklearn/metrics/DistanceMetricType.hpp>
#include <sklearn/metrics/MinkowskiDistance.hpp>
#include <sklearn/neighbors/BallTree.hpp>
#include <sklearn/neighbors/KdTree.hpp>

namespace sklearn {
    namespace neighbors {
        template<typename DType>
        AlgorithmPtr<DType> get_algorithm(AlgorithmType type) {
            switch (type) {
                case AlgorithmType::kAuto:
                    return EuclideanDistancePtr<DType>{};
                case AlgorithmType::kBallTree:
                    return ManhattanDistancePtr<DType>{};
                case AlgorithmType::kKdTree:
                    return ChebyshevDistancePtr<DType>{};
                case AlgorithmType::kBrute:
                    return MinkowskiDistancePtr<DType>{};
                default:
                    throw std::runtime_error("Unknown algorithm type");
                    return nullptr;
            }
        }
    }// namespace neighbors
}// namespace sklearn
