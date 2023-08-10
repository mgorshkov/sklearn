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

#include <sklearn/metrics/r2_score.hpp>

#include <SklearnTest.hpp>

using namespace sklearn::metrics;

class R2ScoreTest : public SklearnTest {
protected:
};

TEST_F(R2ScoreTest, test1) {
    np::Array<np::float_> y_true{3.0, -0.5, 2.0, 7.0};
    np::Array<np::float_> y_pred{2.5, 0.0, 2.0, 8.0};

    R2ScoreParameters<np::Array<np::float_>> params{.y_true = std::move(y_true), .y_pred = std::move(y_pred)};
    np::float_ score = r2_score(params);
    EXPECT_DOUBLE_EQ(score, 0.9486081370449679);
}

TEST_F(R2ScoreTest, test2) {
    np::Array<np::float_> y_true{1, 2, 3};
    np::Array<np::float_> y_pred{1, 2, 3};

    R2ScoreParameters<np::Array<np::float_>> params{.y_true = std::move(y_true), .y_pred = std::move(y_pred)};
    np::float_ score = r2_score(params);
    EXPECT_DOUBLE_EQ(score, 1.0);
}

TEST_F(R2ScoreTest, test3) {
    np::Array<np::float_> y_true{1, 2, 3};
    np::Array<np::float_> y_pred{2, 2, 2};

    R2ScoreParameters<np::Array<np::float_>> params{.y_true = std::move(y_true), .y_pred = std::move(y_pred)};
    np::float_ score = r2_score(params);
    EXPECT_DOUBLE_EQ(score, 0.0);
}

TEST_F(R2ScoreTest, test4) {
    np::Array<np::float_> y_true{1, 2, 3};
    np::Array<np::float_> y_pred{3, 2, 1};

    R2ScoreParameters<np::Array<np::float_>> params{.y_true = std::move(y_true), .y_pred = std::move(y_pred)};
    np::float_ score = r2_score(params);
    EXPECT_DOUBLE_EQ(score, -3.0);
}

TEST_F(R2ScoreTest, test5) {
    np::Array<np::float_> y_true{-2, -2, -2};
    np::Array<np::float_> y_pred{-2, -2, -2};

    R2ScoreParameters<np::Array<np::float_>> params{.y_true = std::move(y_true), .y_pred = std::move(y_pred)};
    np::float_ score = r2_score(params);
    EXPECT_DOUBLE_EQ(score, 1.0);
}
