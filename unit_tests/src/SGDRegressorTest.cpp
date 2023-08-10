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
#include <sklearn/linear_model/SGDRegressor.hpp>
#include <sklearn/metrics/mean_squared_error.hpp>
#include <sklearn/metrics/r2_score.hpp>

#include <SklearnTest.hpp>

class SGDRegressorTest : public SklearnTest {
protected:
};

TEST_F(SGDRegressorTest, ordinaryLeastSquaresTest) {
    using namespace sklearn::linear_model;
    auto reg = SGDRegressor<np::Array<np::float_>>{};

    np::float_ ar1[3][2] = {{0.0, -0.5}, {1.4, 1.3}, {2.1, 2.2}};
    np::float_ ar2[3] = {0.8, 3.2, 9.0};
    reg.fit(np::Array<np::float_>{ar1}, np::Array<np::float_>{ar2});

    np::float_ coef[2] = {1.567695875262711, 1.5725461064903186};
    compare(reg.coef_(), np::Array<np::float_>{coef});

    EXPECT_DOUBLE_EQ(reg.intercept_(), 0.88612575198061738);

    np::float_ ar_pred[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    auto pred = reg.predict(np::Array<np::float_>{ar_pred});

    np::float_ pred_sample[3] = {5.5989138402239655, 11.879397803730026, 18.159881767236083};
    compare(pred, np::Array<np::float_>{pred_sample});
}

TEST_F(SGDRegressorTest, diabetesTest) {
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

    // Create SGD regressor object
    auto regr = SGDRegressor<decltype(diabetes_X_train), decltype(diabetes_y_train)>{};

    // Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train);

    // Make predictions using the testing set
    auto diabetes_y_pred = regr.predict(diabetes_X_test);

    // The coefficients
    compare(regr.coef_(), np::Array<np::float_>{2.1681799});
    // The mean squared error
    MeanSquaredErrorParameters<decltype(diabetes_y_test), decltype(diabetes_y_pred)> mseParams{.y_true = diabetes_y_test, .y_pred = diabetes_y_pred};
    auto mse = mean_squared_error(mseParams);
    EXPECT_DOUBLE_EQ(mse, 5676.474488286437);
    R2ScoreParameters<decltype(diabetes_y_test), decltype(diabetes_y_pred)> r2ScoreParams{.y_true = diabetes_y_test, .y_pred = diabetes_y_pred};
    // The coefficient of determination: 1 is perfect prediction
    auto r2 = r2_score(r2ScoreParams);
    EXPECT_DOUBLE_EQ(r2, -0.17497132951225702);
}