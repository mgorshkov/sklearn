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

#include <sklearn/metrics/f1_score.hpp>

#include <SklearnTest.hpp>

using namespace sklearn::metrics;

class F1ScoreTest : public SklearnTest {
protected:
};

TEST_F(F1ScoreTest, binary1DArrayTest) {
    np::Array<np::bool_> y_true{false, true, false, false};
    np::Array<np::bool_> y_pred{false, true, false, true};

    np::float_ score = f1_score<np::bool_>({.y_true = std::move(y_true), .y_pred = std::move(y_pred)});
    EXPECT_DOUBLE_EQ(score, 0.6666666666666666);
}

TEST_F(F1ScoreTest, binary1DDataFrameTest) {
    np::Array<np::bool_> y_true{false, true, false, false};
    np::Array<np::bool_> y_pred{false, true, false, true};

    np::float_ score = f1_score({.y_true = pd::DataFrame{y_true}, .y_pred = pd::DataFrame{y_pred}});
    EXPECT_DOUBLE_EQ(score, 0.6666666666666666);
}

TEST_F(F1ScoreTest, macro1DArrayTest) {
    np::Array<np::intc> y_true{0, 1, 2, 0, 1, 2};
    np::Array<np::intc> y_pred{0, 2, 1, 0, 0, 1};

    np::float_ score = f1_score<np::intc>({.y_true = std::move(y_true), .y_pred = std::move(y_pred), .average = Average::avMacro});
    EXPECT_DOUBLE_EQ(score, 0.26666666666666666);
}

TEST_F(F1ScoreTest, macro1DStringArrayTest) {
    np::Array<np::string_> y_true{"airplane", "car", "car", "car", "car", "airplane", "boat", "car", "airplane", "car"};
    np::Array<np::string_> y_pred{"airplane", "boat", "car", "car", "boat", "boat", "boat", "airplane", "airplane", "car"};

    np::float_ score = f1_score<np::string_>({.y_true = std::move(y_true), .y_pred = std::move(y_pred), .average = Average::avMacro});
    EXPECT_DOUBLE_EQ(score, 0.57777777777777783);
}

TEST_F(F1ScoreTest, micro1DArrayTest) {
    np::Array<np::intc> y_true{0, 1, 2, 0, 1, 2};
    np::Array<np::intc> y_pred{0, 2, 1, 0, 0, 1};

    np::float_ score = f1_score<np::intc>({.y_true = std::move(y_true), .y_pred = std::move(y_pred), .average = Average::avMicro});
    EXPECT_DOUBLE_EQ(score, 0.3333333333333333);
}

TEST_F(F1ScoreTest, micro1DStringArrayTest) {
    np::Array<np::string_> y_true{"airplane", "car", "car", "car", "car", "airplane", "boat", "car", "airplane", "car"};
    np::Array<np::string_> y_pred{"airplane", "boat", "car", "car", "boat", "boat", "boat", "airplane", "airplane", "car"};

    np::float_ score = f1_score<np::string_>({.y_true = std::move(y_true), .y_pred = std::move(y_pred), .average = Average::avMicro});
    EXPECT_DOUBLE_EQ(score, 0.60);
}

TEST_F(F1ScoreTest, weighed1DStringArrayTest) {
    np::Array<np::string_> y_true{"airplane", "car", "car", "car", "car", "airplane", "boat", "car", "airplane", "car"};
    np::Array<np::string_> y_pred{"airplane", "boat", "car", "car", "boat", "boat", "boat", "airplane", "airplane", "car"};

    np::float_ score = f1_score<np::string_>({.y_true = std::move(y_true), .y_pred = std::move(y_pred), .average = Average::avWeighted});
    EXPECT_DOUBLE_EQ(score, 0.64);
}

TEST_F(F1ScoreTest, binary2DDataFrameTest) {
    np::bool_ array_true[2][2] = {{true, false}, {false, true}};
    np::Array<np::bool_> y_true(array_true);
    np::bool_ array_pred[2][2] = {{false, true}, {true, true}};
    np::Array<np::bool_> y_pred(array_pred);
    EXPECT_THROW(f1_score({.y_true = pd::DataFrame{y_true}, .y_pred = pd::DataFrame{y_pred}}), std::runtime_error);
}

TEST_F(F1ScoreTest, categorial1DSeriesTest) {
    np::Array<np::intc> y_true{0, 1, 2, 3};
    np::Array<np::intc> y_pred{0, 2, 1, 3};

    EXPECT_THROW(f1_score({.y_true = pd::Series{y_true}, .y_pred = pd::Series{y_pred}}), std::runtime_error);
}
