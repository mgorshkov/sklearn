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

#include <sklearn/metrics/mean_absolute_error.hpp>

#include <SklearnTest.hpp>

using namespace sklearn::metrics;

class MeanAbsoluteErrorTest : public SklearnTest {
protected:
};

TEST_F(MeanAbsoluteErrorTest, test1) {
    np::Array<np::float_> y_true{3, -0.5, 2, 7};
    np::Array<np::float_> y_pred{2.5, 0.0, 2, 8};

    MeanAbsoluteErrorParameters<np::Array<np::float_>> params{.y_true = std::move(y_true), .y_pred = std::move(y_pred)};
    np::float_ error = mean_absolute_error(params);
    EXPECT_DOUBLE_EQ(error, 0.5);
}

TEST_F(MeanAbsoluteErrorTest, test2) {
    np::float_ ar1[3][2] = {{0.5, 1.0}, {-1.0, 1.0}, {7.0, -6.0}};
    np::float_ ar2[3][2] = {{0.0, 2.0}, {-1.0, 2.0}, {8.0, -5.0}};
    np::Array<np::float_> y_true{ar1};
    np::Array<np::float_> y_pred{ar2};

    MeanAbsoluteErrorParameters<np::Array<np::float_>> params{.y_true = std::move(y_true), .y_pred = std::move(y_pred)};
    np::float_ error = mean_absolute_error(params);
    EXPECT_DOUBLE_EQ(error, 0.75);
}
