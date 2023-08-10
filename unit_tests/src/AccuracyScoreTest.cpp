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

#include <sklearn/metrics/accuracy_score.hpp>

#include <SklearnTest.hpp>

using namespace sklearn::metrics;

class AccuracyScoreTest : public SklearnTest {
protected:
};

TEST_F(AccuracyScoreTest, 1DArrayTest) {
    np::Array<np::intc> y_true{0, 1, 2, 3};
    np::Array<np::intc> y_pred{0, 2, 1, 3};

    np::float_ score = accuracy_score<np::intc>(y_true, y_pred);
    EXPECT_DOUBLE_EQ(score, 0.5);
}

TEST_F(AccuracyScoreTest, 2DArrayTest) {
    np::intc array[2][2] = {{0, 1}, {1, 1}};
    np::Array<np::intc> y_pred{array};
    auto y_true = np::ones<np::intc>({2, 2});
    np::float_ score = accuracy_score<np::intc>(y_true, y_pred);
    EXPECT_DOUBLE_EQ(score, 0.5);
}

TEST_F(AccuracyScoreTest, 1DDataFrameTest) {
    np::Array<np::intc> y_pred{0, 2, 1, 3};
    np::Array<np::intc> y_true{0, 1, 2, 3};

    np::float_ score = accuracy_score(pd::DataFrame{y_true}, pd::DataFrame{y_pred});
    EXPECT_DOUBLE_EQ(score, 0.5);
}

TEST_F(AccuracyScoreTest, 2DDataFrameTest) {
    np::intc array[2][2] = {{0, 1}, {1, 1}};
    np::Array<np::intc> y_pred{array};
    auto y_true = np::ones<np::intc>({2, 2});
    np::float_ score = accuracy_score(pd::DataFrame{y_true}, pd::DataFrame{y_pred});
    EXPECT_DOUBLE_EQ(score, 0.5);
}

TEST_F(AccuracyScoreTest, 1DSeriesTest) {
    np::Array<np::intc> y_pred{0, 2, 1, 3};
    np::Array<np::intc> y_true{0, 1, 2, 3};

    np::float_ score = accuracy_score(pd::Series{y_true}, pd::Series{y_pred});
    EXPECT_DOUBLE_EQ(score, 0.5);
}
