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
#include <sklearn/metrics/accuracy_score.hpp>
#include <sklearn/metrics/f1_score.hpp>

#include "sklearn/metrics/confusion_matrix.hpp"
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
>>> from sklearn.metrics import DistanceMetric
>>> dist = DistanceMetric.get_metric('euclidean')
>>> X = [2.7810836, 2.550537003]
>>> dist.pairwise(X)
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

TEST_F(MetricsTest, accuracyScore1DArrayTest) {
    np::Array<np::intc> y_true{0, 1, 2, 3};
    np::Array<np::intc> y_pred{0, 2, 1, 3};

    np::float_ score = accuracy_score<np::intc>(y_true, y_pred);
    EXPECT_DOUBLE_EQ(score, 0.5);
}

TEST_F(MetricsTest, accuracyScore2DArrayTest) {
    np::intc array[2][2] = {{0, 1}, {1, 1}};
    np::Array<np::intc> y_pred{array};
    auto y_true = np::ones<np::intc>({2, 2});
    np::float_ score = accuracy_score<np::intc>(y_true, y_pred);
    EXPECT_DOUBLE_EQ(score, 0.5);
}

TEST_F(MetricsTest, accuracyScore1DDataFrameTest) {
    np::Array<np::intc> y_pred{0, 2, 1, 3};
    np::Array<np::intc> y_true{0, 1, 2, 3};

    np::float_ score = accuracy_score(pd::DataFrame{y_true}, pd::DataFrame{y_pred});
    EXPECT_DOUBLE_EQ(score, 0.5);
}

TEST_F(MetricsTest, accuracyScore2DDataFrameTest) {
    np::intc array[2][2] = {{0, 1}, {1, 1}};
    np::Array<np::intc> y_pred{array};
    auto y_true = np::ones<np::intc>({2, 2});
    np::float_ score = accuracy_score(pd::DataFrame{y_true}, pd::DataFrame{y_pred});
    EXPECT_DOUBLE_EQ(score, 0.5);
}

TEST_F(MetricsTest, accuracyScore1DSeriesTest) {
    np::Array<np::intc> y_pred{0, 2, 1, 3};
    np::Array<np::intc> y_true{0, 1, 2, 3};

    np::float_ score = accuracy_score(pd::Series{y_true}, pd::Series{y_pred});
    EXPECT_DOUBLE_EQ(score, 0.5);
}

TEST_F(MetricsTest, f1ScoreBinary1DArrayTest) {
    np::Array<np::bool_> y_true{false, true, false, false};
    np::Array<np::bool_> y_pred{false, true, false, true};

    np::float_ score = f1_score<np::bool_>({.y_true = std::move(y_true), .y_pred = std::move(y_pred)});
    EXPECT_DOUBLE_EQ(score, 0.6666666666666666);
}

TEST_F(MetricsTest, f1ScoreBinary1DDataFrameTest) {
    np::Array<np::bool_> y_true{false, true, false, false};
    np::Array<np::bool_> y_pred{false, true, false, true};

    np::float_ score = f1_score({.y_true = pd::DataFrame{y_true}, .y_pred = pd::DataFrame{y_pred}});
    EXPECT_DOUBLE_EQ(score, 0.6666666666666666);
}

TEST_F(MetricsTest, f1ScoreMacro1DArrayTest) {
    np::Array<np::intc> y_true{0, 1, 2, 0, 1, 2};
    np::Array<np::intc> y_pred{0, 2, 1, 0, 0, 1};

    np::float_ score = f1_score<np::intc>({.y_true = std::move(y_true), .y_pred = std::move(y_pred), .average = Average::avMacro});
    EXPECT_DOUBLE_EQ(score, 0.26666666666666666);
}

TEST_F(MetricsTest, f1ScoreMacro1DStringArrayTest) {
    np::Array<np::string_> y_true{"airplane", "car", "car", "car", "car", "airplane", "boat", "car", "airplane", "car"};
    np::Array<np::string_> y_pred{"airplane", "boat", "car", "car", "boat", "boat", "boat", "airplane", "airplane", "car"};

    np::float_ score = f1_score<np::string_>({.y_true = std::move(y_true), .y_pred = std::move(y_pred), .average = Average::avMacro});
    EXPECT_DOUBLE_EQ(score, 0.57777777777777783);
}

TEST_F(MetricsTest, f1ScoreMicro1DArrayTest) {
    np::Array<np::intc> y_true{0, 1, 2, 0, 1, 2};
    np::Array<np::intc> y_pred{0, 2, 1, 0, 0, 1};

    np::float_ score = f1_score<np::intc>({.y_true = std::move(y_true), .y_pred = std::move(y_pred), .average = Average::avMicro});
    EXPECT_DOUBLE_EQ(score, 0.3333333333333333);
}

TEST_F(MetricsTest, f1ScoreMicro1DStringArrayTest) {
    np::Array<np::string_> y_true{"airplane", "car", "car", "car", "car", "airplane", "boat", "car", "airplane", "car"};
    np::Array<np::string_> y_pred{"airplane", "boat", "car", "car", "boat", "boat", "boat", "airplane", "airplane", "car"};

    np::float_ score = f1_score<np::string_>({.y_true = std::move(y_true), .y_pred = std::move(y_pred), .average = Average::avMicro});
    EXPECT_DOUBLE_EQ(score, 0.60);
}

TEST_F(MetricsTest, f1ScoreWeighed1DStringArrayTest) {
    np::Array<np::string_> y_true{"airplane", "car", "car", "car", "car", "airplane", "boat", "car", "airplane", "car"};
    np::Array<np::string_> y_pred{"airplane", "boat", "car", "car", "boat", "boat", "boat", "airplane", "airplane", "car"};

    np::float_ score = f1_score<np::string_>({.y_true = std::move(y_true), .y_pred = std::move(y_pred), .average = Average::avWeighted});
    EXPECT_DOUBLE_EQ(score, 0.64);
}

TEST_F(MetricsTest, f1ScoreBinary2DDataFrameTest) {
    np::bool_ array_true[2][2] = {{true, false}, {false, true}};
    np::Array<np::bool_> y_true(array_true);
    np::bool_ array_pred[2][2] = {{false, true}, {true, true}};
    np::Array<np::bool_> y_pred(array_pred);
    EXPECT_THROW(f1_score({.y_true = pd::DataFrame{y_true}, .y_pred = pd::DataFrame{y_pred}}), std::runtime_error);
}

TEST_F(MetricsTest, f1Score1DCategorialSeriesTest) {
    np::Array<np::intc> y_true{0, 1, 2, 3};
    np::Array<np::intc> y_pred{0, 2, 1, 3};

    EXPECT_THROW(f1_score({.y_true = pd::Series{y_true}, .y_pred = pd::Series{y_pred}}), std::runtime_error);
}

TEST_F(MetricsTest, confusionMatrixNumbersArrayTest) {
    np::Array<np::intc> y_true{0, 0, 2, 2, 0, 2};
    np::Array<np::intc> y_pred{2, 0, 2, 2, 0, 1};

    auto matrix = confusion_matrix<np::intc>({.y_true = std::move(y_true), .y_pred = std::move(y_pred)});
    compare(matrix, np::Array<np::Size>({{2, 0, 1},
                                         {0, 0, 0},
                                         {0, 1, 2}}));
}

TEST_F(MetricsTest, confusionMatrixStringArrayTest) {
    np::Array<np::string_> y_true{"airplane", "car", "car", "car", "car", "airplane", "boat", "car", "airplane", "car"};
    np::Array<np::string_> y_pred{"airplane", "boat", "car", "car", "boat", "boat", "boat", "airplane", "airplane", "car"};

    auto matrix = confusion_matrix<np::string_>({.y_true = std::move(y_true), .y_pred = std::move(y_pred)});
    compare(matrix, np::Array<np::Size>({{2, 1, 0},
                                         {0, 1, 0},
                                         {1, 2, 3}}));
}