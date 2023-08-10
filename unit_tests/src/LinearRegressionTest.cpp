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

#include <np/Array.hpp>
#include <np/Comp.hpp>

#include <sklearn/datasets/datasets.hpp>
#include <sklearn/linear_model/LinearRegression.hpp>
#include <sklearn/metrics/mean_squared_error.hpp>
#include <sklearn/metrics/r2_score.hpp>

#include <SklearnTest.hpp>

class LinearRegressionTest : public SklearnTest {
protected:
};

TEST_F(LinearRegressionTest, OLSTest) {
    using namespace sklearn::linear_model;
    auto reg = LinearRegression{};

    np::float_ X[4][2] = {{1.0, 1.0}, {1.0, 2.0}, {2.0, 2.0}, {3.0, 4.0}};
    np::float_ y[4] = {6.0, 8.0, 9.0, 11.0};
    reg.fit(np::Array<np::float_>{X}, np::Array<np::float_>{y});

    EXPECT_DOUBLE_EQ(reg.intercept_(), 4.8000000000000558);

    compare(reg.coef_(), np::Array<np::float_>{0.7, 1.1});

    np::float_ ar_pred[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    auto pred = reg.predict(np::Array<np::float_>{ar_pred});

    np::float_ pred_sample[3] = {7.7, 11.3, 14.9};
    compare(pred, np::Array<np::float_>{pred_sample});
}

TEST_F(LinearRegressionTest, weightedOLSTest) {
    // weighted OLS
    using namespace sklearn::linear_model;
    auto reg = LinearRegression{};

    np::float_ X[4][2] = {{1.0, 1.0}, {1.0, 2.0}, {2.0, 2.0}, {3.0, 4.0}};
    np::float_ y[4] = {6.0, 8.0, 9.0, 11.0};
    np::float_ sample_weight[4] = {4.0, 0.5, 2.0, 3.0};
    reg.fit(np::Array<np::float_>{X}, np::Array<np::float_>{y}, np::Array<np::float_>{sample_weight});

    EXPECT_DOUBLE_EQ(reg.intercept_(), 4.1250000000000187);

    compare(reg.coef_(), np::Array<np::float_>{1.5625, 0.59375});

    np::float_ ar_pred[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    auto pred = reg.predict(np::Array<np::float_>{ar_pred});

    np::float_ pred_sample[3] = {6.875, 11.1875, 15.5};
    compare(pred, np::Array<np::float_>{pred_sample});
}

TEST_F(LinearRegressionTest, diabetesTest) {
    using namespace sklearn::linear_model;
    using namespace sklearn::datasets;
    using namespace sklearn::metrics;

    // Load the diabetes dataset
    auto diabetes = load_diabetes();
    auto diabetes_X = diabetes.data();
    auto diabetes_y = diabetes.target();

    // Use only one feature
    auto diabetes_X_feature = diabetes_X[":, 2"].reshape(np::Shape{442, 1});

    // Split the data into training/testing sets
    auto diabetes_X_train = diabetes_X_feature[":-20,:"];
    auto diabetes_X_test = diabetes_X_feature["-20:,:"];

    // Split the targets into training/testing sets
    auto diabetes_y_train = diabetes_y[":-20"];
    auto diabetes_y_test = diabetes_y["-20:"];

    // Create linear regression object
    auto regr = LinearRegression{};

    // Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train);

    // Make predictions using the testing set
    auto diabetes_y_pred = regr.predict(diabetes_X_test);

    // The coefficients
    compare(regr.coef_(), np::Array<np::float_>{938.23786125});
    // The mean squared error
    MeanSquaredErrorParameters<decltype(diabetes_y_test), decltype(diabetes_y_pred)> mseParams{.y_true = diabetes_y_test, .y_pred = diabetes_y_pred};
    auto mse = mean_squared_error(mseParams);
    EXPECT_DOUBLE_EQ(mse, 2548.0723987259735);
    R2ScoreParameters<decltype(diabetes_y_test), decltype(diabetes_y_pred)> r2ScoreParams{.y_true = diabetes_y_test, .y_pred = diabetes_y_pred};
    // The coefficient of determination: 1 is perfect prediction
    auto r2 = r2_score(r2ScoreParams);
    EXPECT_DOUBLE_EQ(r2, 0.47257544798227069);
}