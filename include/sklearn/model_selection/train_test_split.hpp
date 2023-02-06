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

#pragma once

#include <np/Array.hpp>
#include <np/Constants.hpp>
#include <np/DType.hpp>

#include <pd/core/frame/DataFrame/DataFrame.hpp>
#include <pd/core/frame/DataFrame/DataFrameStreamIo.hpp>

#include <sklearn/model_selection/train_test_split.hpp>

#include <optional>
#include <vector>

namespace sklearn {
    namespace model_selection {
        template<typename TypeX, typename TypeY>
        using Split = std::tuple<TypeX, TypeX, TypeY, TypeY>;

        /* Parameters:
        X, y - np arrays or pd dataframes with same length / shape[0]
        test_size np::float_ or np::Size, default=None
        If np::float_, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If np::Size, represents the absolute number of test samples. If std::nullopt, the value is set to the complement of the train size.
        If train_size is also std::nullopt, it will be set to 0.25.

        // train_size np::float_ or np::Size, default=std::nullopt
        // If np::float_, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
        // If np::Size, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.

        random_state int, seed for random number generator, default=std::nullopt
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

        shuffle bool, default=true
        Whether or not to shuffle the data before splitting. If shuffle=false then stratify must be std::nullopt.

        stratify array, default=None
        If not std::nullopt, data is split in a stratified fashion, using this as the class labels.
        */

        template<typename ArrayX, typename ArrayY>
        struct train_test_split_params {
            ArrayX X;
            ArrayY y;
            std::optional<std::variant<np::Size, np::float_>> test_size{std::nullopt};
            std::optional<std::variant<np::Size, np::float_>> train_size{std::nullopt};
            std::optional<int> random_state{std::nullopt};
            bool shuffle{true};
            std::optional<ArrayY> stratify{std::nullopt};
        };

        template<typename DTypeX, typename DTypeY, np::Size SizeX = np::SIZE_DEFAULT, np::Size SizeY = np::SIZE_DEFAULT>
        inline Split<np::Array<DTypeX>, np::Array<DTypeY>> train_test_split(train_test_split_params<np::Array<DTypeX, SizeX>, np::Array<DTypeY, SizeY>> params = {}) {
            if (!params.test_size && !params.train_size) {
                params.test_size = 0.25;
                params.train_size = 1.0 - std::get<np::float_>(*params.test_size);
            } else if (!params.test_size) {
                params.test_size = 1.0 - std::get<np::float_>(*params.train_size);
            } else if (!params.train_size) {
                params.train_size = 1.0 - std::get<np::float_>(*params.test_size);
            }
            if (params.X.empty()) {
                throw std::runtime_error("X must not be empty");
            }
            if (params.y.empty()) {
                throw std::runtime_error("y must not be empty");
            }

            auto shape_X = params.X.shape();
            auto shape_y = params.y.shape();

            if (shape_X[0] != shape_y[0]) {
                throw std::runtime_error("X and y must have equal number or rows");
            }

            np::Size rows = shape_X[0];
            np::Size columns = shape_X.size() > 1 ? shape_X[1] : 1;

            np::Array<DTypeX> X_shuffled{shape_X};
            np::Array<DTypeY> y_shuffled{shape_y};
            if (params.shuffle) {
                if (params.stratify) {
                    throw std::runtime_error("This function is not implemented yet");
                }
                std::unique_ptr<std::default_random_engine> e;
                if (params.random_state) {
                    e = std::make_unique<std::default_random_engine>(*params.random_state);
                } else {
                    std::random_device rd;
                    e = std::make_unique<std::default_random_engine>(rd());
                }
                std::vector<np::Size> indices{};
                indices.resize(rows);
                std::iota(indices.begin(), indices.end(), 0L);
                std::shuffle(indices.begin(), indices.end(), *e);
                for (np::Size row = 0; row < rows; ++row) {
                    for (np::Size column = 0; column < columns; ++column) {
                        X_shuffled.set(column + row * columns, params.X.get(column + indices[row] * columns));
                    }
                    y_shuffled.set(row, params.y.get(indices[row]));
                }
            } else {
                if (params.stratify) {
                    throw std::runtime_error("Stratify must be null if shuffle = false");
                }
                X_shuffled = params.X.copy();
                y_shuffled = params.y.copy();
            }
            np::Shape shape_X_test;
            np::Shape shape_X_train;
            np::Shape shape_y_test;
            np::Shape shape_y_train;
            if (params.test_size) {
                if (np::float_ *pval = std::get_if<np::float_>(&params.test_size.value())) {
                    shape_X_test = np::Shape{static_cast<np::Size>(*pval * rows), columns};
                    shape_X_train = np::Shape{rows - shape_X_test[0], columns};
                    shape_y_test = np::Shape{static_cast<np::Size>(*pval * shape_y[0])};
                    shape_y_train = np::Shape{rows - shape_y_test[0]};
                }
            }
            if (params.train_size) {
                if (np::float_ *pval = std::get_if<np::float_>(&params.train_size.value())) {
                    shape_X_train = np::Shape{static_cast<np::Size>(*pval * rows), columns};
                    shape_X_test = np::Shape{rows - shape_X_train[0], columns};
                    shape_y_train = np::Shape{static_cast<np::Size>(*pval * shape_y[0])};
                    shape_y_test = np::Shape{rows - shape_y_train[0]};
                }
            }

            auto X_test = np::Array<DTypeX>{shape_X_test};
            std::copy(X_shuffled.cbegin(), X_shuffled.cbegin() + shape_X_test.calcSizeByShape(), X_test.begin());

            auto X_train = np::Array<DTypeX>{shape_X_train};
            std::copy(X_shuffled.cbegin() + shape_X_test.calcSizeByShape(),
                      X_shuffled.cbegin() + shape_X_test.calcSizeByShape() + shape_X_train.calcSizeByShape(),
                      X_train.begin());

            auto y_test = np::Array<DTypeY>{shape_y_test};
            std::copy(y_shuffled.cbegin(), y_shuffled.cbegin() + shape_y_test.calcSizeByShape(), y_test.begin());

            auto y_train = np::Array<DTypeY>{shape_y_train};
            std::copy(y_shuffled.cbegin() + shape_y_test.calcSizeByShape(),
                      y_shuffled.cbegin() + shape_y_test.calcSizeByShape() + shape_y_train.calcSizeByShape(),
                      y_train.begin());

            return {X_train, X_test, y_train, y_test};
        }

        inline Split<pd::DataFrame, pd::DataFrame> train_test_split(train_test_split_params<pd::DataFrame, pd::DataFrame> params) {
            if (!params.test_size && !params.train_size) {
                params.test_size = 0.25;
                params.train_size = 1.0 - std::get<np::float_>(*params.test_size);
            } else if (!params.test_size) {
                params.test_size = 1.0 - std::get<np::float_>(*params.train_size);
            } else {
                params.train_size = 1.0 - std::get<np::float_>(*params.test_size);
            }
            if (params.X.empty()) {
                throw std::runtime_error("X must not be empty");
            }
            if (params.y.empty()) {
                throw std::runtime_error("y must not be empty");
            }
            auto shape_X = params.X.shape();
            auto shape_y = params.y.shape();
            if (shape_X[0] != shape_y[0]) {
                throw std::runtime_error("X and y must have equal number or rows");
            }

            np::Size rows = shape_X[0];
            np::Size columns = shape_X.size() > 1 ? shape_X[1] : 1;

            pd::DataFrame X_shuffled;
            pd::DataFrame y_shuffled;
            if (params.shuffle) {
                if (params.stratify) {
                    throw std::runtime_error("This function is not implemented yet");
                }
                std::unique_ptr<std::default_random_engine> e;
                if (params.random_state) {
                    e = std::make_unique<std::default_random_engine>(*params.random_state);
                } else {
                    std::random_device rd;
                    e = std::make_unique<std::default_random_engine>(rd());
                }
                std::vector<np::Size> indices{};
                indices.resize(rows);
                std::iota(indices.begin(), indices.end(), 0L);
                std::shuffle(indices.begin(), indices.end(), *e);
                for (np::Size column = 0; column < columns; ++column) {
                    np::Array<pd::internal::Value> data{np::Shape{rows}};
                    for (np::Size row = 0; row < rows; ++row) {
                        auto value = params.X.at(indices[row], column);
                        data.set(row, value);
                    }
                    X_shuffled.append(pd::Series{data, column});
                }
                np::Array<pd::internal::Value> data{np::Shape{rows}};
                for (np::Size row = 0; row < rows; ++row) {
                    auto value = params.y.at(indices[row], 0);
                    data.set(row, value);
                }
                y_shuffled.append(pd::Series{data, 0});
            } else {
                if (params.stratify) {
                    throw std::runtime_error("Stratify must be null if shuffle = false");
                }
                X_shuffled = params.X;
                y_shuffled = params.y;
            }
            np::Shape shape_X_test;
            np::Shape shape_X_train;
            np::Shape shape_y_test;
            np::Shape shape_y_train;
            if (params.test_size) {
                if (np::float_ *pval = std::get_if<np::float_>(&params.test_size.value())) {
                    shape_X_test = np::Shape{static_cast<np::Size>(*pval * static_cast<np::float_>(rows)), shape_X[1]};
                    shape_X_train = np::Shape{rows - shape_X_test[0], columns};
                    shape_y_test = np::Shape{static_cast<np::Size>(*pval * static_cast<np::float_>(rows))};
                    shape_y_train = np::Shape{rows - shape_y_test[0]};
                }
            }
            if (params.train_size) {
                if (np::float_ *pval = std::get_if<np::float_>(&params.train_size.value())) {
                    shape_X_train = np::Shape{static_cast<np::Size>(*pval * rows), columns};
                    shape_X_test = np::Shape{rows - shape_X_train[0], columns};
                    shape_y_train = np::Shape{static_cast<np::Size>(*pval * rows)};
                    shape_y_test = np::Shape{rows - shape_y_train[0]};
                }
            }
            np::Size sizeTest = shape_X_test[0];
            std::string rowsTest = ":" + std::to_string(sizeTest);
            auto X_test = X_shuffled.iloc(rowsTest);
            std::string rowsTrain = std::to_string(sizeTest) + ":";
            auto X_train = X_shuffled.iloc(rowsTrain);
            auto y_test = y_shuffled.iloc(rowsTest);
            auto y_train = y_shuffled.iloc(rowsTrain);

            return {X_train, X_test, y_train, y_test};
        }

    }// namespace model_selection
}// namespace sklearn
