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

// Rewrite of https://github.com/mobigroup/gmtsar/blob/pygmtsar/todo/PRM.robust_trend2d.ipynb

#include <ctime>
#include <iostream>
#include <vector>

#include <np/Array.hpp>
#include <scipy/special/betainc.hpp>
#include <sklearn/linear_model/LinearRegression.hpp>

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
        throw std::runtime_error("Number of model parameters \"rank\" should be 1, 2, or 3");
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
    auto z = data[":, 2"];
    auto w = ones(z.shape());

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

        auto r = abs(z.subtract(mlr.predict(xy)));
        auto chisq = sum((r * r * w)) / static_cast<float_>(z.size() - 3);
        chisqs.push_back(chisq);
        auto k = 1.5 * MAD_NORMALIZE * median(r);
        w = where<float_>(
                r, [k](const auto &element) { return element <= k; }, [](const auto &element) { return 1.0; },
                [k](const auto &element) { return 2 * k / element - k * k / (element * element); });
        auto sig = (chisqs.size() == 1 ? 1 : gmtstat_f_q(chisqs[chisqs.size() - 1], static_cast<float_>(z.size() - 3), chisqs[chisqs.size() - 2], static_cast<float_>(z.size() - 3)));
        // Go back to previous model only if previous chisq < current chisq
        if (chisqs.size() == 1 or chisqs[chisqs.size() - 2] > chisqs[chisqs.size() - 1]) {
            coeffs = mlr.coeffs_();
        }

        //print ('chisq', chisq, 'significant', sig)
        if (sig < sig_threshold) {
            break;
        }
    }
    // get the slope and intercept of the line best fit
    return (coeffs[":" + std::to_string(rank)]);
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

auto measure_time(auto func, auto data, int rank, int n_runs) {
    timespec start_time{};
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    for (int i = 0; i < n_runs; ++i) {
        func(data, rank);
    }
    timespec end_time{};
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    return 1000000000 * (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec);
}

struct Result {
    int rank;
    int noise_level;
    int gmt_time;
};

void test_time(int num_points = 100 * 1000, int n_runs = 50, const std::vector<int> &ranks = {1, 2, 3}, const std::vector<int> &noise_levels = {0, 1, 10, 50}) {
    std::vector<Result> results;
    for (auto rank: ranks) {
        for (auto noise_level: noise_levels) {
            auto data = generate_data(rank, num_points, noise_level);

            int gmt_time = 1000 * measure_time(GMT_trend2d, data, rank, n_runs);
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
    test_time();

    return 0;
}
