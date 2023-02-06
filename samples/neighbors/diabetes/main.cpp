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

#include <iostream>

#include <pd/core/frame/DataFrame/DataFrameStreamIo.hpp>
#include <pd/read_csv.hpp>

#include <sklearn/metrics/accuracy_score.hpp>
#include <sklearn/metrics/confusion_matrix.hpp>
#include <sklearn/metrics/f1_score.hpp>
#include <sklearn/model_selection/train_test_split.hpp>
#include <sklearn/neighbors/KNeighborsClassifier.hpp>
#include <sklearn/preprocessing/StandardScaler.hpp>

int main(int, char **) {
    using namespace pd;
    using namespace sklearn::model_selection;
    using namespace sklearn::neighbors;
    using namespace sklearn::preprocessing;
    using namespace sklearn::metrics;

    auto data = read_csv("https://raw.githubusercontent.com/adityakumar529/Coursera_Capstone/master/diabetes.csv");
    const char *non_zero[] = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"};
    for (const auto &column: non_zero) {
        data[column] = data[column].replace(0L, np::NaN);
        auto mean = data[column].mean(true);
        data[column] = data[column].replace(np::NaN, mean);
    }

    auto X = data.iloc(":", "0:8");
    auto y = data.iloc(":", "8");
    auto [X_train, X_test, y_train, y_test] = train_test_split({.X = X, .y = y, .test_size = 0.2, .random_state = 42});

    auto sc_X = StandardScaler{};
    X_train = sc_X.fit_transform(X_train);
    X_test = sc_X.transform(X_test);

    auto classifier = KNeighborsClassifier<pd::DataFrame>{{.n_neighbors = 13, .p = 2, .metric = sklearn::metrics::DistanceMetricType::kEuclidean}};
    classifier.fit(X_train, y_train);
    auto y_pred = classifier.predict(X_test);
    std::cout << "Prediction: " << y_pred << std::endl;
    auto cm = confusion_matrix({.y_true = y_test, .y_pred = y_pred});
    std::cout << cm << std::endl;
    std::cout << f1_score({.y_true = y_test, .y_pred = y_pred}) << std::endl;
    std::cout << accuracy_score(y_test, y_pred) << std::endl;

    return 0;
}
