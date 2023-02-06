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

namespace sklearn {
    namespace metrics {
        np::float_ f1_score(const F1ScoreParameters<pd::DataFrame> &params) {
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
                F1Score score{};
                for (np::Size i = 0; i < params.y_true.shape()[0]; ++i) {
                    auto y_true = params.y_true.iloc(i);
                    auto y_pred = params.y_pred.iloc(i);
                    bool equal = y_true == y_pred;
                    bool allOnesInPred = true;
                    for (std::size_t j = 0; j < y_pred.shape()[0]; ++j) {
                        if (y_pred.dtype() != "bool" && y_pred.dtype() != "int32" && y_pred.dtype() != "int64" && y_pred.dtype() != "uint64") {
                            throw std::runtime_error("Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].");
                        } else if (params.y_pred.shape().size() > 1) {
                            throw std::runtime_error("Target is multilabel-indicator but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted', 'samples']");
                        }

                        if (!y_pred[j]) {
                            allOnesInPred = false;
                            break;
                        }
                    }
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
                auto confusionMatrix = confusion_matrix({.y_true = params.y_true, .y_pred = params.y_pred});
                std::unordered_map<pd::internal::Value, F1Score> scores;
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
                totalScore.calcRecall();
                totalScore.calcPrecision();
                return totalScore.f1();
            }

            if (params.average == Average::avMacro) {
                auto confusionMatrix = confusion_matrix({.y_true = params.y_true, .y_pred = params.y_pred});
                std::unordered_map<pd::internal::Value, F1Score> scores;
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
                return scores.empty() ? 0.0 : macro_f1 / static_cast<np::float_>(scores.size());
            }

            if (params.average == Average::avWeighted) {
                auto confusionMatrix = confusion_matrix({.y_true = params.y_true, .y_pred = params.y_pred});
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

        np::float_ f1_score(const F1ScoreParameters<pd::Series> &params) {
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
                F1Score score{};
                for (np::Size i = 0; i < params.y_true.shape()[0]; ++i) {
                    auto y_true = params.y_true[i];
                    auto y_pred = params.y_pred[i];
                    if (!y_pred.isBool()) {
                        throw std::runtime_error("Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].");
                    } else if (params.y_true.shape().size() > 1) {
                        throw std::runtime_error("Target is multilabel-indicator but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted', 'samples']");
                    }
                    bool equal = y_true == y_pred;
                    bool allOnesInPred = y_pred == true;
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
                auto confusionMatrix = confusion_matrix({.y_true = params.y_true, .y_pred = params.y_pred});
                std::unordered_map<pd::internal::Value, F1Score> scores;
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
                totalScore.calcRecall();
                totalScore.calcPrecision();
                return totalScore.f1();
            }

            if (params.average == Average::avMacro) {
                auto confusionMatrix = confusion_matrix({.y_true = params.y_true, .y_pred = params.y_pred});
                std::unordered_map<pd::internal::Value, F1Score> scores;
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
                return scores.empty() ? 0.0 : macro_f1 / static_cast<np::float_>(scores.size());
            }

            if (params.average == Average::avWeighted) {
                auto confusionMatrix = confusion_matrix({.y_true = params.y_true, .y_pred = params.y_pred});
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
    }// namespace metrics
}// namespace sklearn