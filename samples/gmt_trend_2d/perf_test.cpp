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

// Detailed performance profiling of GMT_trend2d components

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <np/Array.hpp>
#include <scipy/special/betainc.hpp>
#include <sklearn/linear_model/LinearRegression.hpp>
#include <sklearn/metrics/mean_squared_error.hpp>

using namespace np;
using namespace scipy;
using namespace sklearn;

class Timer {
public:
    void start() { m_start = std::chrono::steady_clock::now(); }
    long long elapsed_ns() const {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - m_start).count();
    }
    double elapsed_ms() const { return elapsed_ns() / 1'000'000.0; }

private:
    std::chrono::time_point<std::chrono::steady_clock> m_start;
};

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

auto GMT_trend2d(const Array<float_> &data, int rank) {
    float_ MAD_NORMALIZE = 1.4826;
    float_ sig_threshold = 0.51;

    if (rank != 1 && rank != 2 && rank != 3) {
        throw sklearn::RuntimeError("Number of model parameters \"rank\" should be 1, 2, or 3");
    }

    auto gmtstat_f_q = [](float_ chisq1, float_ nu1, float_ chisq2, float_ nu2) {
        if (chisq1 == 0.0) return 1.0;
        if (chisq2 == 0.0) return 0.0;
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

    auto mlr = linear_model::LinearRegression{};

    std::vector<float_> chisqs;
    Array<float_> coeffs;

    while (true) {
        mlr.fit(xy, z, w);

        auto r = abs_sub(z, mlr.predict(xy));
        auto chisq = sum_sq_weighted(r, w) / static_cast<float_>(z.size() - 3);
        chisqs.push_back(chisq);

        auto k = 1.5 * MAD_NORMALIZE * median(r);
        w = where_tukey(r, k);
        auto sig = (chisqs.size() == 1 ? 1 : gmtstat_f_q(chisqs[chisqs.size() - 1], static_cast<float_>(z.size() - 3), chisqs[chisqs.size() - 2], static_cast<float_>(z.size() - 3)));
        if (chisqs.size() == 1 or chisqs[chisqs.size() - 2] > chisqs[chisqs.size() - 1]) {
            coeffs = mlr.coeffs_();
        }

        if (sig < sig_threshold) {
            break;
        }
    }
    auto result = Array<float_>{};
    for (int i = 0; i < rank; ++i) {
        result = append(result, Array<float_>{coeffs.get(i)});
    }
    return result;
}

// Ultra-granular profiling of data preparation
void profile_data_prep(int num_points = 100 * 1000, int rank = 2, int n_runs = 50) {
    std::cout << "\n========== ULTRA-GRANULAR DATA PREP PROFILING ==========\n";
    std::cout << "num_points=" << num_points << ", rank=" << rank << "\n\n";

    auto data = generate_data(rank, num_points, 0);
    Timer timer;
    std::vector<long long> times;

    // 1a. Just the slice: data[":,0"]
    times.clear();
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        auto x_ = data[":,0"];
        (void) x_;
        times.push_back(timer.elapsed_ns());
    }
    auto avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "1a. Slice data[:,0]: " << avg_ns / 1'000'000.0 << " ms\n";

    // 1b. x_.min() and x_.max()
    times.clear();
    auto x_ = data[":,0"];
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        auto mn = x_.min();
        auto mx = x_.max();
        (void) mn;
        (void) mx;
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "1b. min/max: " << avg_ns / 1'000'000.0 << " ms\n";

    // 1c. Array construction {min, max} and {-1, +1}
    times.clear();
    auto mn = x_.min();
    auto mx = x_.max();
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        auto xp = Array<float_>{mn, mx};
        auto fp = Array<float_>{-1.0, +1.0};
        (void) xp;
        (void) fp;
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "1c. Array<float_>{min,max} and {-1,+1}: " << avg_ns / 1'000'000.0 << " ms\n";

    // 1d. Just interp() alone
    times.clear();
    auto xp = Array<float_>{mn, mx};
    auto fp = Array<float_>{-1.0, +1.0};
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        auto x = interp(x_, xp, fp);
        (void) x;
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "1d. interp() alone: " << avg_ns / 1'000'000.0 << " ms\n";

    // 1e. data[:,2].copy()
    times.clear();
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        auto z = data[":, 2"].copy();
        (void) z;
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "1e. data[:,2].copy(): " << avg_ns / 1'000'000.0 << " ms\n";

    // 1f. ones(z.shape()).copy()
    times.clear();
    auto z = data[":, 2"].copy();
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        auto w = ones(z.shape()).copy();
        (void) w;
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "1f. ones(z.shape()).copy(): " << avg_ns / 1'000'000.0 << " ms\n";

    // 1g. Full data prep (all steps together)
    times.clear();
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        auto x_slice = data[":,0"];
        auto x_interp = interp(x_slice, Array<float_>{x_slice.min(), x_slice.max()}, Array<float_>{-1, +1});
        auto z_copy = data[":, 2"].copy();
        auto w_copy = ones(z_copy.shape()).copy();
        (void) x_interp;
        (void) z_copy;
        (void) w_copy;
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "1g. Full data prep (all steps): " << avg_ns / 1'000'000.0 << " ms\n";
}

// Profile individual components of GMT_trend2d
void profile_gmt_components(int num_points = 100 * 1000, int rank = 2, int noise_level = 0, int n_runs = 10) {
    std::cout << "\n========== DETAILED COMPONENT PROFILING ==========\n";
    std::cout << "num_points=" << num_points << ", rank=" << rank << ", noise_level=" << noise_level << "\n\n";

    auto data = generate_data(rank, num_points, noise_level);

    Timer timer;
    std::vector<long long> times;

    // Build data prep arrays once
    Array<float_> x, z, w;
    {
        auto x_ = data[":,0"];
        x = interp(x_, Array<float_>{x_.min(), x_.max()}, Array<float_>{-1, +1});
        z = data[":, 2"].copy();
        w = ones(z.shape()).copy();
    }

    // Feature matrix construction
    times.clear();
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        Array<float_> xy;
        if (rank == 1) {
            xy = expand_dims(zeros(z.shape()), 1);
        } else if (rank == 2) {
            xy = expand_dims(x, 1);
        } else if (rank == 3) {
            xy = stack(x, data[":,1"]).transpose();
        }
        times.push_back(timer.elapsed_ns());
    }
    auto avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "2. Feature matrix construction: " << avg_ns / 1'000'000.0 << " ms\n";

    Array<float_> xy;
    if (rank == 1) {
        xy = expand_dims(zeros(z.shape()), 1);
    } else if (rank == 2) {
        xy = expand_dims(x, 1);
    } else if (rank == 3) {
        xy = stack(x, data[":,1"]).transpose();
    }

    // 3. LinearRegression::fit (weighted)
    times.clear();
    for (int run = 0; run < n_runs; ++run) {
        auto mlr = linear_model::LinearRegression{};
        timer.start();
        mlr.fit(xy, z, w);
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "3. LinearRegression::fit (weighted): " << avg_ns / 1'000'000.0 << " ms\n";

    // 4. LinearRegression::fit (unweighted)
    times.clear();
    Array<float_> empty_w;
    for (int run = 0; run < n_runs; ++run) {
        auto mlr = linear_model::LinearRegression{};
        timer.start();
        mlr.fit(xy, z, empty_w);
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "4. LinearRegression::fit (unweighted): " << avg_ns / 1'000'000.0 << " ms\n";

    // 5. predict
    auto mlr = linear_model::LinearRegression{};
    mlr.fit(xy, z, w);
    times.clear();
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        auto pred = mlr.predict(xy);
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "5. predict: " << avg_ns / 1'000'000.0 << " ms\n";

    // 6. Residual computation: abs(z - predict) - using fused abs_sub
    auto pred = mlr.predict(xy);
    times.clear();
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        auto r = abs_sub(z, pred);
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "6. abs(z - predict) [fused]: " << avg_ns / 1'000'000.0 << " ms\n";

    // 7. Chi-squared computation: sum((r*r*w)) / (n-3) - using fused sum_sq_weighted
    auto r = abs_sub(z, pred);
    times.clear();
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        auto chisq = sum_sq_weighted(r, w) / static_cast<float_>(z.size() - 3);
        (void) chisq;
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "7. Chi-squared [fused]: " << avg_ns / 1'000'000.0 << " ms\n";

    // 8. Median computation
    times.clear();
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        auto med = median(r);
        (void) med;
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "8. median(r): " << avg_ns / 1'000'000.0 << " ms\n";

    // 9. Weight update: where(r <= k, 1, 2k/r - k²/r²)
    auto k = 1.5 * 1.4826 * median(r);
    times.clear();
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        auto w1 = where_tukey(r, k);
        (void) w1;
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "9. Weight update (where): " << avg_ns / 1'000'000.0 << " ms\n";

    // 10. betainc (significance test)
    times.clear();
    for (int run = 0; run < n_runs; ++run) {
        timer.start();
        auto sig = scipy::special::betainc(0.5 * (z.size() - 3), 0.5 * (z.size() - 3), 0.5);
        (void) sig;
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "10. betainc: " << avg_ns / 1'000'000.0 << " ms\n";

    // 11. Full single iteration of the IRLS loop (using fused operations)
    times.clear();
    for (int run = 0; run < n_runs; ++run) {
        auto mlr_local = linear_model::LinearRegression{};
        auto w_local = w.copy();
        timer.start();
        mlr_local.fit(xy, z, w_local);
        auto r_local = abs_sub(z, mlr_local.predict(xy));
        auto chisq_local = sum_sq_weighted(r_local, w_local) / static_cast<float_>(z.size() - 3);
        (void) chisq_local;
        auto k_local = 1.5 * 1.4826 * median(r_local);
        auto w1_local = where_tukey(r_local, k_local);
        (void) w1_local;
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "\n11. Full single IRLS iteration [fused]: " << avg_ns / 1'000'000.0 << " ms\n";

    // 12. Full GMT_trend2d call
    times.clear();
    for (int run = 0; run < n_runs; ++run) {
        auto data_copy = data.copy();
        timer.start();
        auto result = GMT_trend2d(data_copy, rank);
        (void) result;
        times.push_back(timer.elapsed_ns());
    }
    avg_ns = std::accumulate(times.begin(), times.end(), 0LL) / n_runs;
    std::cout << "12. Full GMT_trend2d: " << avg_ns / 1'000'000.0 << " ms\n";
}

// Profile across all ranks and noise levels
void profile_all(int num_points = 100 * 1000, int n_runs = 10) {
    std::cout << "\n========== CROSS-PROFILE: ALL RANKS & NOISE LEVELS ==========\n";
    std::cout << "num_points=" << num_points << ", n_runs=" << n_runs << "\n\n";

    for (int rank: {1, 2, 3}) {
        for (int noise: {0, 1, 10, 50}) {
            auto data = generate_data(rank, num_points, noise);

            Timer timer;
            timer.start();
            for (int run = 0; run < n_runs; ++run) {
                auto data_copy = data.copy();
                GMT_trend2d(data_copy, rank);
            }
            auto total_ms = timer.elapsed_ms() / n_runs;

            // Count iterations
            int iter_count = 0;
            {
                auto data_copy = data.copy();
                auto x_ = data_copy[":,0"];
                auto x = interp(x_, Array<float_>{x_.min(), x_.max()}, Array<float_>{-1, +1});
                auto z = data_copy[":, 2"].copy();
                Array<float_> w = ones(z.shape()).copy();
                Array<float_> xy;
                if (rank == 1) xy = expand_dims(zeros(z.shape()), 1);
                else if (rank == 2)
                    xy = expand_dims(x, 1);
                else
                    xy = stack(x, data_copy[":,1"]).transpose();

                auto mlr = linear_model::LinearRegression{};
                float_ sig_threshold = 0.51;
                std::vector<float_> chisqs;
                while (true) {
                    mlr.fit(xy, z, w);
                    auto r = abs(z - mlr.predict(xy));
                    auto chisq = sum((r * r * w)) / static_cast<float_>(z.size() - 3);
                    chisqs.push_back(chisq);
                    auto k = 1.5 * 1.4826 * median(r);
                    w = where<float_>(
                            r, [k](const auto &element) { return element <= k; }, [](const auto &) { return 1.0; },
                            [k](const auto &element) { return 2 * k / element - k * k / (element * element); });
                    auto sig = (chisqs.size() == 1 ? 1 : scipy::special::betainc(0.5 * (z.size() - 3), 0.5 * (z.size() - 3), chisqs.back() / (chisqs.back() + chisqs[chisqs.size() - 2])));
                    iter_count++;
                    if (sig < sig_threshold) break;
                }
            }

            std::cout << "Rank=" << rank << " Noise=" << noise
                      << " | Total=" << std::setw(6) << std::fixed << std::setprecision(1) << total_ms << " ms"
                      << " | Iterations=" << iter_count
                      << " | Per-iter=" << std::setw(5) << (total_ms / iter_count) << " ms\n";
        }
    }
}

int main(int, char **) {
    // Ultra-granular data prep profiling
    profile_data_prep(100'000, 2, 50);

    // Profile components for rank=2 (most common case)
    profile_gmt_components(100'000, 2, 0, 20);

    // Profile across all combinations
    profile_all(100'000, 20);

    return 0;
}
