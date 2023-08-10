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

#include <np/Comp.hpp>

#include <sklearn/metrics/DistanceMetric.hpp>
#include <sklearn/metrics/EuclideanDistance.hpp>

#include <SklearnTest.hpp>

using namespace sklearn::metrics;

class MetricsTest : public SklearnTest {
protected:
};

TEST_F(MetricsTest, chebyshevDistancePairwise1ParamTest) {
    /*
>>> from sklearn.metrics import DistanceMetric
>>> dist = DistanceMetric.get_metric('chebyshev')
>>> X = [[0, 1, 2],
       [3, 4, 5]]
>>> dist.pairwise(X)
array([[ 0.,  3.],
      [ 3.,  0.]])
*/
    auto dist = DistanceMetric<np::Array<np::float_>>::get_metric(DistanceMetricType::kChebyshev);
    double array_c[2][3] = {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}};
    np::Array<np::float_> array{array_c};
    auto result = dist->pairwise(array);
    double result_array_c[2][2] = {{0., 3.}, {3., 0.}};
    np::Array<np::float_> result_sample{result_array_c};
    compare(result, result_sample);
}

TEST_F(MetricsTest, chebyshevDistancePairwise2ParamsTest) {
    /*
>>> from sklearn.metrics import DistanceMetric
>>> dist = DistanceMetric.get_metric('chebyshev')
>>> X = [[0, 1, 2],
        [3, 4, 5]]
>>> Y = [[6, 7, 8],
        [9, 10, 11],
        [12, 13, 14]]
>>> dist.pairwise(X, Y)
array([[6., 9., 12.],
       [3., 6., 9.]])
*/
    auto dist = DistanceMetric<np::Array<np::float_>>::get_metric(DistanceMetricType::kChebyshev);
    np::float_ array_1_c[2][3] = {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}};
    np::Array<np::float_> array_1{array_1_c};
    np::float_ array_2_c[3][3] = {{6.0, 7.0, 8.0}, {9.0, 10.0, 11.0}, {12.0, 13.0, 14.0}};
    np::Array<np::float_> array_2{array_2_c};
    auto result = dist->pairwise(array_1, array_2);
    np::float_ result_array_c[2][3] = {{6., 9., 12.}, {3., 6., 9.}};
    np::Array<np::float_> result_sample{result_array_c};
    compare(result, result_sample);
}

TEST_F(MetricsTest, euclideanDistancePairwise1ParamTest) {
    /*
>>> from sklearn.metrics.pairwise import euclidean_distances
>>> dist = DistanceMetric.get_metric('euclidean')
>>> X = [[0, 1, 2],
         [3, 4, 5]]
    euclidean_distances(X)
array([[ 0.        ,  5.19615242],
      [ 5.19615242,  0.        ]])
*/
    auto dist = DistanceMetric<np::Array<np::float_>>::get_metric(DistanceMetricType::kEuclidean);
    double array_c[2][3] = {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}};
    np::Array<np::float_> array{array_c};
    auto result = dist->pairwise(array);
    double result_array_c[2][2] = {{0., 5.196152422706632}, {5.196152422706632, 0.}};
    np::Array<np::float_> result_sample{result_array_c};
    compare(result, result_sample);
}

TEST_F(MetricsTest, euclideanDistancePairwise2ParamsTest) {
    /*
>>> from sklearn.metrics import DistanceMetric
>>> dist = DistanceMetric.get_metric('euclidean')
>>> X = [[0, 1, 2],
        [3, 4, 5]]
>>> Y = [[6, 7, 8],
        [9, 10, 11],
        [12, 13, 14]]
>>> dist.pairwise(X, Y)
array([[10.39230485 15.58845727 20.78460969]
      [ 5.19615242 10.39230485 15.58845727]])
*/
    auto dist = DistanceMetric<np::Array<np::float_>>::get_metric(DistanceMetricType::kEuclidean);
    double array_1_c[2][3] = {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}};
    np::Array<np::float_> array_1{array_1_c};
    double array_2_c[3][3] = {{6.0, 7.0, 8.0}, {9.0, 10.0, 11.0}, {12.0, 13.0, 14.0}};
    np::Array<np::float_> array_2{array_2_c};
    auto result = dist->pairwise(array_1, array_2);
    np::float_ result_array_c[2][3] = {{10.392304845413264, 15.588457268119896, 20.784609690826528},
                                       {5.196152422706632, 10.392304845413264, 15.588457268119896}};
    np::Array<np::float_> result_sample{result_array_c};
    compare(result, result_sample);
}

TEST_F(MetricsTest, manhattanDistancePairwise1ParamTest) {
    /*
>>> from sklearn.metrics import DistanceMetric
>>> dist = DistanceMetric.get_metric('manhattan')
>>> X = [[0, 1, 2],
       [3, 4, 5]]
>>> dist.pairwise(X)
array([[ 0.,  9.],
      [ 9.,  0.]])
*/
    auto dist = DistanceMetric<np::Array<np::float_>>::get_metric(DistanceMetricType::kManhattan);
    double array_c[2][3] = {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}};
    np::Array<np::float_> array{array_c};
    auto result = dist->pairwise(array);
    double result_array_c[2][2] = {{0., 9.}, {9., 0.}};
    np::Array<np::float_> result_sample{result_array_c};
    compare(result, result_sample);
}

TEST_F(MetricsTest, manhattanDistancePairwise2ParamsTest) {
    /*
>>> from sklearn.metrics import DistanceMetric
>>> dist = DistanceMetric.get_metric('manhattan')
>>> X = [[0, 1, 2],
        [3, 4, 5]]
>>> Y = [[6, 7, 8],
        [9, 10, 11],
        [12, 13, 14]]
>>> dist.pairwise(X, Y)
array([[18., 27., 36.],
       [9., 18., 27.]])
*/
    auto dist = DistanceMetric<np::Array<np::float_>>::get_metric(DistanceMetricType::kManhattan);
    np::float_ array_1_c[2][3] = {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}};
    np::Array<np::float_> array_1{array_1_c};
    np::float_ array_2_c[3][3] = {{6.0, 7.0, 8.0}, {9.0, 10.0, 11.0}, {12.0, 13.0, 14.0}};
    np::Array<np::float_> array_2{array_2_c};
    auto result = dist->pairwise(array_1, array_2);
    np::float_ result_array_c[2][3] = {{18., 27., 36.}, {9., 18., 27.}};
    np::Array<np::float_> result_sample{result_array_c};
    compare(result, result_sample);
}

TEST_F(MetricsTest, minkowskiDistancePairwise1Test) {
    /*
>>> from sklearn.metrics import DistanceMetric
>>> dist = DistanceMetric.get_metric('minkowski')
>>> X = [[0, 1, 2],
        [3, 4, 5]]
>>> dist.pairwise(X)
array([[ 0.        ,  5.19615242],
      [ 5.19615242,  0.        ]])
*/
    auto dist = DistanceMetric<np::Array<np::float_>>::get_metric(DistanceMetricType::kMinkowski);
    double array_c[2][3] = {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}};
    np::Array<np::float_> array{array_c};
    auto result = dist->pairwise(array);
    double result_array_c[2][2] = {{0., 5.196152422706632}, {5.196152422706632, 0.}};
    np::Array<np::float_> result_sample{result_array_c};
    compare(result, result_sample);
}

TEST_F(MetricsTest, minkowskiDistancePairwise2Test) {
    /*
>>> from sklearn.metrics import DistanceMetric
>>> dist = DistanceMetric.get_metric('minkowski')
>>> X = [[0, 1, 2],
        [3, 4, 5]]
>>> Y = [[6, 7, 8],
        [9, 10, 11],
        [12, 13, 14]]
>>> dist.pairwise(X, Y)
array([[10.39230485, 15.58845727, 20.78460969],
       [ 5.19615242, 10.39230485, 15.58845727]])
*/
    auto dist = DistanceMetric<np::Array<np::float_>>::get_metric(DistanceMetricType::kMinkowski);
    np::float_ array_1_c[2][3] = {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}};
    np::Array<np::float_> array_1{array_1_c};
    np::float_ array_2_c[3][3] = {{6.0, 7.0, 8.0}, {9.0, 10.0, 11.0}, {12.0, 13.0, 14.0}};
    np::Array<np::float_> array_2{array_2_c};
    auto result = dist->pairwise(array_1, array_2);
    np::float_ result_array_c[2][3] = {{10.392304845413264, 15.588457268119896, 20.784609690826528},
                                       {5.196152422706632, 10.392304845413264, 15.588457268119896}};
    np::Array<np::float_> result_sample{result_array_c};
    compare(result, result_sample);
}
