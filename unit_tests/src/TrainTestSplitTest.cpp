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

#include <sklearn/datasets/datasets.hpp>
#include <sklearn/model_selection/train_test_split.hpp>

#include <SklearnTest.hpp>

using namespace sklearn;

class TrainTestSplitTest : public SklearnTest {
protected:
};

TEST_F(TrainTestSplitTest, trainTestSplitTestArray) {
    using namespace model_selection;

    auto X = np::arange(10).reshape(np::Shape{5, 2});
    auto y = np::arange(5);
    auto [X_train, X_test, y_train, y_test] =
            train_test_split<np::intc, np::intc>({.X = X, .y = y, .test_size = 0.33, .random_state = 42});

    np::Shape sh_X_train{3, 2};
    EXPECT_EQ(X_train.shape(), sh_X_train);

    np::Shape sh_y_train{3};
    EXPECT_EQ(y_train.shape(), sh_y_train);

    np::Shape sh_X_test{2, 2};
    EXPECT_EQ(X_test.shape(), sh_X_test);

    np::Shape sh_y_test{2};
    EXPECT_EQ(y_test.shape(), sh_y_test);

    auto X_sample = X_train.append(X_test).reshape(np::Shape{5, 2});
    std::sort(X_sample.begin(), X_sample.end());
    compare(X_sample, X);

    auto y_sample = y_train.append(y_test);
    std::sort(y_sample.begin(), y_sample.end());
    compare(y_sample, y);
}

TEST_F(TrainTestSplitTest, trainTestSplitTestDataFrame) {
    using namespace model_selection;

    auto X = np::arange(10).reshape(np::Shape{5, 2});
    auto y = np::arange(5);
    auto [X_train, X_test, y_train, y_test] =
            train_test_split({.X = pd::DataFrame{X}, .y = pd::DataFrame{y}, .test_size = 0.33, .random_state = 42});

    np::Shape sh_X_train{3, 2};
    EXPECT_EQ(X_train.shape(), sh_X_train);

    np::Shape sh_X_test{2, 2};
    EXPECT_EQ(X_test.shape(), sh_X_test);

    {
        auto X_train_series0 = X_train[pd::internal::Value{0}];
        auto X_train_series0_array = *static_cast<np::Array<np::intc> *>(X_train_series0.values());

        auto X_test_series0 = X_test[pd::internal::Value{0}];
        auto X_test_series0_array = *static_cast<np::Array<np::intc> *>(X_test_series0.values());

        auto X_sample_0 = X_train_series0_array.append(X_test_series0_array);
        std::sort(X_sample_0.begin(), X_sample_0.end());
        compare(X_sample_0, np::Array<np::intc>{0, 2, 4, 6, 8});
    }
    {
        auto X_train_series1 = X_train[pd::internal::Value{1}];
        auto X_train_series1_array = *static_cast<np::Array<np::intc> *>(X_train_series1.values());

        auto X_test_series1 = X_test[pd::internal::Value{1}];
        auto X_test_series1_array = *static_cast<np::Array<np::intc> *>(X_test_series1.values());

        auto X_sample_1 = X_train_series1_array.append(X_test_series1_array);
        std::sort(X_sample_1.begin(), X_sample_1.end());
        compare(X_sample_1, np::Array<np::intc>{1, 3, 5, 7, 9});
    }

    np::Shape sh_y_train{3};
    EXPECT_EQ(y_train.shape(), sh_y_train);

    np::Shape sh_y_test{2};
    EXPECT_EQ(y_test.shape(), sh_y_test);

    auto y_train_series = y_train[pd::internal::Value{0}];
    auto y_train_series_array = *static_cast<np::Array<np::intc> *>(y_train_series.values());

    auto y_test_series = y_test[pd::internal::Value{0}];
    auto y_test_series_array = *static_cast<np::Array<np::intc> *>(y_test_series.values());

    auto y_sample = y_train_series_array.append(y_test_series_array);
    std::sort(y_sample.begin(), y_sample.end());
    compare(y_sample, np::Array<np::intc>{0, 1, 2, 3, 4});
}

TEST_F(TrainTestSplitTest, stratifyTest) {
    using namespace datasets;
    using namespace model_selection;

    auto iris = datasets::load_iris();
    auto data = iris.data();
    auto target = iris.target();

    try {
        auto [X_train, X_test, y_train, y_test] =
                train_test_split<np::float_, np::int_, 600, 150>({.X = data, .y = target, .test_size = 0.8, .stratify = target});

        np::Array<np::short_> y_test_sample{0, 0, 0, 0, 2, 2, 1, 0, 1, 2, 2, 0, 0, 1, 0, 1, 1, 2, 1, 2, 0, 2, 2,
                                            1, 2, 1, 1, 0, 2, 1};
        compare(y_test, y_test_sample);
        EXPECT_TRUE(false);
    } catch (const std::runtime_error &e) {
        EXPECT_STREQ(e.what(), "This function is not implemented yet");
    }
}
