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

#include <sklearn/metrics/confusion_matrix.hpp>

#include <SklearnTest.hpp>

using namespace sklearn::metrics;

class ConfusionMatrixTest : public SklearnTest {
protected:
};

TEST_F(ConfusionMatrixTest, numbersArrayTest) {
    np::Array<np::intc> y_true{0, 0, 2, 2, 0, 2};
    np::Array<np::intc> y_pred{2, 0, 2, 2, 0, 1};

    auto matrix = confusion_matrix<np::intc>({.y_true = std::move(y_true), .y_pred = std::move(y_pred)});
    compare(matrix, np::Array<np::Size>({{2, 0, 1},
                                         {0, 0, 0},
                                         {0, 1, 2}}));
}

TEST_F(ConfusionMatrixTest, stringArrayTest) {
    np::Array<np::string_> y_true{"airplane", "car", "car", "car", "car", "airplane", "boat", "car", "airplane", "car"};
    np::Array<np::string_> y_pred{"airplane", "boat", "car", "car", "boat", "boat", "boat", "airplane", "airplane", "car"};

    auto matrix = confusion_matrix<np::string_>({.y_true = std::move(y_true), .y_pred = std::move(y_pred)});
    compare(matrix, np::Array<np::Size>({{2, 1, 0},
                                         {0, 1, 0},
                                         {1, 2, 3}}));
}
