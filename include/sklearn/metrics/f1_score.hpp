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
#include <sklearn/metrics/confusion_matrix.hpp>
#include <unordered_map>

namespace sklearn {
    namespace metrics {
        enum class Average {
            avMicro,
            avMacro,
            avSamples,
            avWeighted,
            avBinary,
            avNone
        };

        enum class ZeroDivision {
            zdWarn,
            zdOff,
            zdOn
        };

        template<typename ArrayX, typename ArrayY = ArrayX>
        struct F1ScoreParameters {
            ArrayX y_true;
            ArrayY y_pred;
            bool pos_lavel{true};
            Average average{Average::avBinary};
            ZeroDivision zero_division{ZeroDivision::zdWarn};
        };

        struct F1Score {
            [[nodiscard]] np::float_ f1() const {
                if (precision + recall == 0.0) {
                    return 0.0;
                }
                np::float_ f1 = 2 * precision * recall / (precision + recall);
                return f1;
            }

            void calcPrecision() {
                precision = tp + fp == 0 ? 0 : static_cast<np::float_>(tp) / static_cast<np::float_>(tp + fp);
            }
            void calcRecall() {
                recall = tp + fn == 0 ? 0 : static_cast<np::float_>(tp) / static_cast<np::float_>(tp + fn);
            }

            np::Size tp{0};
            np::Size fp{0};
            np::Size fn{0};
            np::Size support{0};
            np::float_ precision{0};
            np::float_ recall{0};
        };

        template<typename DTypeX, typename DTypeY = DTypeX, np::Size SizeX = np::SIZE_DEFAULT, np::Size SizeY = np::SIZE_DEFAULT>
        np::float_ f1_score(const F1ScoreParameters<np::Array<DTypeX, SizeX>, np::Array<DTypeY, SizeY>> &params = {}) {
            if (params.y_true.empty() && params.y_pred.empty()) {
                return 1.0;
            }
            if (params.y_true.ndim() != params.y_pred.ndim()) {
                throw std::runtime_error("Arrays must be of equal dimensions");
            }
            if (params.y_true.size() != params.y_pred.size()) {
                throw std::runtime_error("Arrays must be of equal sizes");
            }

            if (params.average == Average::avBinary) {
                if (!std::is_same_v<DTypeX, bool>) {
                    throw std::runtime_error("Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].");
                } else if (params.y_true.shape().size() > 1) {
                    throw std::runtime_error("Target is multilabel-indicator but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted', 'samples']");
                }

                F1Score score{};
                for (np::Size i = 0; i < params.y_true.shape()[0]; ++i) {
                    auto y_true = params.y_true[i];
                    auto y_pred = params.y_pred[i];
                    bool equal = np::array_equal(y_true, y_pred);
                    bool allOnesInPred = std::all_of(y_pred.cbegin(), y_pred.cend(), [](auto i) {
                        bool result;
                        np::ndarray::internal::isTrue(i, result);
                        return result;
                    });
                    if (allOnesInPred) {
                        if (equal) {
                            ++score.tp;
                        } else {
                            ++score.fp;
                        }
                    } else if (!equal) {
                        ++score.fn;
                    }
                }
                score.calcPrecision();
                score.calcRecall();
                return score.f1();
            }

            if (params.average == Average::avMicro) {
                auto confusionMatrix = confusion_matrix<DTypeX>({.y_true = params.y_true, .y_pred = params.y_pred});
                std::unordered_map<np::Size, F1Score> scores;
                for (np::Size i = 0; i < confusionMatrix.len(); ++i) {
                    for (np::Size j = 0; j < confusionMatrix[i].len(); ++j) {
                        if (i == j) {
                            scores[i].tp = confusionMatrix[i][j].get(0);
                        } else {
                            scores[i].fn += confusionMatrix[i][j].get(0);
                            scores[i].fp += confusionMatrix[j][i].get(0);
                        }
                        scores[i].calcPrecision();
                        scores[i].calcRecall();
                    }
                }
                F1Score totalScore{};
                for (const auto &score: scores) {
                    totalScore.tp += score.second.tp;
                    totalScore.fp += score.second.fp;
                    totalScore.fn += score.second.fn;
                }
                totalScore.calcPrecision();
                totalScore.calcRecall();
                return totalScore.f1();
            }

            if (params.average == Average::avMacro) {
                auto confusionMatrix = confusion_matrix<DTypeX>({.y_true = params.y_true, .y_pred = params.y_pred});
                std::unordered_map<np::Size, F1Score> scores;
                for (np::Size i = 0; i < confusionMatrix.len(); ++i) {
                    for (np::Size j = 0; j < confusionMatrix[i].len(); ++j) {
                        if (i == j) {
                            scores[i].tp = confusionMatrix[i][j].get(0);
                        } else {
                            scores[i].fn += confusionMatrix[i][j].get(0);
                            scores[i].fp += confusionMatrix[j][i].get(0);
                        }
                        scores[i].calcPrecision();
                        scores[i].calcRecall();
                    }
                }
                np::float_ macro_f1 = 0;
                for (const auto &score: scores) {
                    macro_f1 += score.second.f1();
                }
                macro_f1 /= scores.empty() ? 0 : static_cast<np::float_>(scores.size());
                return macro_f1;
            }

            if (params.average == Average::avWeighted) {
                auto confusionMatrix = confusion_matrix<DTypeX>({.y_true = params.y_true, .y_pred = params.y_pred});
                std::unordered_map<np::Size, F1Score> scores;
                np::Size totalSupport{0};
                for (np::Size i = 0; i < confusionMatrix.len(); ++i) {
                    for (np::Size j = 0; j < confusionMatrix[i].len(); ++j) {
                        if (i == j) {
                            scores[i].tp = confusionMatrix[i][j].get(0);
                        } else {
                            scores[i].fn += confusionMatrix[i][j].get(0);
                            scores[i].fp += confusionMatrix[j][i].get(0);
                        }
                        scores[i].calcPrecision();
                        scores[i].calcRecall();

                        scores[i].support += confusionMatrix[i][j].get(0);
                        totalSupport += confusionMatrix[i][j].get(0);
                    }
                }

                np::float_ weighted_f1 = 0;
                for (const auto &score: scores) {
                    auto f1 = score.second.f1();
                    weighted_f1 += totalSupport == 0.0 ? 0.0 : f1 * score.second.support / static_cast<np::float_>(totalSupport);
                }
                return weighted_f1;
            }
            throw std::runtime_error("Invalid average param");

            return 0.0;
        }

        np::float_ f1_score(const F1ScoreParameters<pd::DataFrame> &params = {});
        np::float_ f1_score(const F1ScoreParameters<pd::Series> &params = {});

    }// namespace metrics
}// namespace sklearn