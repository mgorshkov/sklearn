/*
⚡ ML methods in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU

Copyright (c) 2023-2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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

// Rewrite of https://github.com/mobigroup/gmtsar/blob/pygmtsar/todo/PRM.robust_trend2d.ipynb

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <np/Array.hpp>
#include <scipy/special/betainc.hpp>
#include <sklearn/linear_model/LinearRegression.hpp>
#include <sklearn/metrics/mean_squared_error.hpp>

using namespace np;
using namespace scipy;
using namespace sklearn;

auto GMT_trend2d(const Array<float_> &data, int rank) {
    // scale factor for normally distributed data is 1.4826
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html
    float_ MAD_NORMALIZE = 1.4826;
    // significance value
    float_ sig_threshold = 0.51;

    if (rank != 1 && rank != 2 && rank != 3) {
        throw sklearn::RuntimeError("Number of model parameters \"rank\" should be 1, 2, or 3");
    }

    // see gmt_stat.c
    auto gmtstat_f_q = [](float_ chisq1, float_ nu1, float_ chisq2, float_ nu2) {
        if (chisq1 == 0.0) {
            return 1.0;
        }
        if (chisq2 == 0.0) {
            return 0.0;
        }
        return scipy::special::betainc(0.5 * nu2, 0.5 * nu1, chisq2 / (chisq2 + chisq1));
    };

    Array<float_> x;
    if (rank == 2 || rank == 3) {
        auto x_ = data[":,0"];
        x = interp(x_, Array<float_>{x_.min(), x_.max()}, Array<float_>{-1, +1});
    }
    Array<float_> y;
    if (rank == 3) {
        auto y_ = data[":,1"];
        y = interp(y_, Array<float_>{y_.min(), y_.max()}, Array<float_>{-1, +1});
    }
    auto z = data[":, 2"].copy();
    Array<float_> w = ones(z.shape()).copy();

    Array<float_> xy;
    if (rank == 1) {
        xy = expand_dims(zeros(z.shape()), 1);
    } else if (rank == 2) {
        xy = expand_dims(x, 1);
    } else if (rank == 3) {
        xy = stack(x, y).transpose();
    }

    // create linear regression object
    auto mlr = linear_model::LinearRegression{};

    std::vector<float_> chisqs;
    Array<float_> coeffs;

    while (true) {
        // fit linear regression
        mlr.fit(xy, z, w);

        auto r = abs_sub(z, mlr.predict(xy));
        auto chisq = sum_sq_weighted(r, w) / static_cast<float_>(z.size() - 3);
        chisqs.push_back(chisq);

        auto k = 1.5 * MAD_NORMALIZE * median(r);
        w = where_tukey(r, k);
        auto sig = (chisqs.size() == 1 ? 1 : gmtstat_f_q(chisqs[chisqs.size() - 1], static_cast<float_>(z.size() - 3), chisqs[chisqs.size() - 2], static_cast<float_>(z.size() - 3)));
        // Go back to previous model only if previous chisq < current chisq
        if (chisqs.size() == 1 or chisqs[chisqs.size() - 2] > chisqs[chisqs.size() - 1]) {
            coeffs = mlr.coeffs_();
        }

        //std::cout << "chisq, " << chisq << ", significant," << sig << std::endl;
        if (sig < sig_threshold) {
            break;
        }
    }
    // Return first 'rank' coefficients
    auto result = Array<float_>{};
    for (int i = 0; i < rank; ++i) {
        result = append(result, Array<float_>{coeffs.get(i)});
    }
    return result;
}

auto generate_data(auto rank, auto num_points, auto noise_level) {
    random::seed(42);
    auto x = linspace(-10.0, 10.0, num_points);
    auto y = linspace(-10.0, 10.0, num_points);
    if (rank == 1) {
        auto z = 3 * x + 5 + noise_level * random::randn(num_points);
        return column_stack(x, y, z);
    }
    if (rank == 2) {
        auto z = 2 * x + 3 * y + 5 + noise_level * random::randn(num_points);
        return column_stack(x, y, z);
    }
    auto z = 2 * x * x + 3 * y * y + 5 + noise_level * random::randn(num_points);
    return column_stack(x, y, z);
}

auto calculate_mse(const Array<float_> &data, const Array<float_> &coeffs, int rank) {
    auto z_actual = data[":,2"].copy();
    Array<float_> z_predicted;

    if (rank == 1) {
        z_predicted = coeffs.get(0) * ones(z_actual.shape());
    } else if (rank == 2) {
        // Interpolate x the same way as in GMT_trend2d
        auto x_ = data[":,0"];
        auto x = interp(x_, Array<float_>{x_.min(), x_.max()}, Array<float_>{-1, +1});
        z_predicted = coeffs.get(0) + coeffs.get(1) * x;
    } else if (rank == 3) {
        // Interpolate x and y the same way as in GMT_trend2d
        auto x_ = data[":,0"];
        auto x = interp(x_, Array<float_>{x_.min(), x_.max()}, Array<float_>{-1, +1});
        auto y_ = data[":,1"];
        auto y = interp(y_, Array<float_>{y_.min(), y_.max()}, Array<float_>{-1, +1});
        z_predicted = coeffs.get(0) + coeffs.get(1) * x + coeffs.get(2) * y;
    }
    using sklearn::metrics::mean_squared_error;
    using sklearn::metrics::MeanSquaredErrorParameters;
    MeanSquaredErrorParameters<Array<float_>> params{.y_true = z_actual, .y_pred = z_predicted};
    return mean_squared_error(params);
}

void test_mse(int num_points = 100 * 1000, const std::vector<int> &ranks = {1, 2, 3}, const std::vector<int> &noise_levels = {0, 1, 10, 50}) {
    std::vector<std::tuple<int, int, float_>> results;
    for (auto rank: ranks) {
        for (auto noise_level: noise_levels) {
            auto data = generate_data(rank, num_points, noise_level);
            auto coeffs_gmt = GMT_trend2d(data, rank);
            // round coefficients to 8 decimal places
            auto coeffs_rounded = coeffs_gmt.copy();
            for (std::size_t i = 0; i < coeffs_rounded.size(); ++i) {
                coeffs_rounded.set(i, std::round(coeffs_rounded.get(i) * 1e8) / 1e8);
            }
            auto mse_gmt = calculate_mse(data, coeffs_rounded, rank);
            // round MSE to zero decimal places
            mse_gmt = std::round(mse_gmt);
            results.emplace_back(rank, noise_level, mse_gmt);
        }
    }
    // print table
    std::cout << "Rank\tNoise Level\tGMT_trend2d, MSE\n";
    for (const auto &[rank, noise_level, mse]: results) {
        std::cout << rank << "\t" << noise_level << "\t" << mse << "\n";
    }
}

auto measure_time(auto func, const auto &data, int rank, int n_runs) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n_runs; ++i) {
        func(data, rank);
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / n_runs;
}

struct Result {
    int rank;
    int noise_level;
    long long gmt_time;
};

void test_time(int num_points = 100 * 1000, int n_runs = 50, const std::vector<int> &ranks = {1, 2, 3}, const std::vector<int> &noise_levels = {0, 1, 10, 50}) {
    std::vector<Result> results;
    for (auto rank: ranks) {
        for (auto noise_level: noise_levels) {
            auto data = generate_data(rank, num_points, noise_level);

            auto gmt_time = measure_time(GMT_trend2d, data, rank, n_runs) / 1'000'000;// time in milliseconds
            results.push_back({rank, noise_level, gmt_time});
        }
    }

    auto headers = {"Rank", "Noise Level", "GMT_trend2d, [ms]"};
    for (const auto &header: headers) {
        std::cout << header << "\t";
    }
    std::cout << std::endl;
    for (const auto &result: results) {
        std::cout << result.rank << "\t" << result.noise_level << "\t" << result.gmt_time << std::endl;
    }
}

int main(int, char **) {
    test_mse();
    test_time();

    return 0;
}
