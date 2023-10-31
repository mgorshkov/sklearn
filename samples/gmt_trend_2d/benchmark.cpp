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

// Cross-language benchmark: C++ vs Python GMT_trend2d performance comparison.
//
// This program:
//   1. Calls GMT_trend2d in C++ directly and measures execution time.
//   2. Spawns a Python subprocess running the identical algorithm and measures
//      its execution time.
//   3. Computes and prints the percentage performance difference.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <np/Array.hpp>
#include <scipy/special/betainc.hpp>
#include <sklearn/linear_model/LinearRegression.hpp>
#include <sklearn/metrics/mean_squared_error.hpp>

using namespace np;
using namespace scipy;
using namespace sklearn;

// ---------------------------------------------------------------------------
// GMT_trend2d — C++ implementation (identical to main.cpp)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Data generation (identical to main.cpp)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Utility: run a shell command and capture its stdout
// ---------------------------------------------------------------------------
std::string exec_cmd(const std::string &cmd) {
    std::array<char, 4096> buffer{};
    std::string result;
    // Use popen to capture stdout
    auto deleter = [](FILE *f) { if (f) pclose(f); };
    std::unique_ptr<FILE, decltype(deleter)> pipe(popen(cmd.c_str(), "r"), deleter);
    if (!pipe) {
        return "";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

// ---------------------------------------------------------------------------
// Measure Python GMT_trend2d time via subprocess
// ---------------------------------------------------------------------------
double measure_python_time(int rank, int num_points, int noise_level, int n_runs) {
    // Build the command to invoke the Python benchmark helper
    std::string script_path = "./benchmark_python.py";
    // Also try the source directory
    std::string cmd = "python3 " + script_path + " " +
                      std::to_string(rank) + " " +
                      std::to_string(num_points) + " " +
                      std::to_string(noise_level) + " " +
                      std::to_string(n_runs) + " 2>/dev/null";

    std::string output = exec_cmd(cmd);

    // If not found in cwd, try the source tree path
    if (output.empty()) {
        cmd = "python3 samples/gmt_trend_2d/benchmark_python.py " +
              std::to_string(rank) + " " +
              std::to_string(num_points) + " " +
              std::to_string(noise_level) + " " +
              std::to_string(n_runs) + " 2>/dev/null";
        output = exec_cmd(cmd);
    }

    // Parse the "PYTHON_TIME <value>" line
    auto pos = output.find("PYTHON_TIME");
    if (pos == std::string::npos) {
        std::cerr << "ERROR: Could not parse Python output. Got:\n"
                  << output << std::endl;
        return -1.0;
    }
    auto val_start = pos + 11;// skip "PYTHON_TIME"
    // Skip whitespace
    while (val_start < output.size() && (output[val_start] == ' ' || output[val_start] == '\t'))
        ++val_start;
    auto val_end = val_start;
    while (val_end < output.size() && output[val_end] != '\n' && output[val_end] != '\r')
        ++val_end;
    std::string val_str = output.substr(val_start, val_end - val_start);
    return std::stod(val_str);
}

// ---------------------------------------------------------------------------
// Measure C++ GMT_trend2d time
// ---------------------------------------------------------------------------
double measure_cpp_time(const Array<float_> &data, int rank, int n_runs) {
    // Warm-up run
    GMT_trend2d(data, rank);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n_runs; ++i) {
        GMT_trend2d(data, rank);
    }
    auto end = std::chrono::steady_clock::now();
    auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return static_cast<double>(total_ns) / (n_runs * 1'000'000.0);// ms
}

// ---------------------------------------------------------------------------
// Main benchmark driver
// ---------------------------------------------------------------------------
int main(int, char **) {
    const int num_points = 100'000;
    const int n_runs = 20;
    const std::vector<int> ranks = {1, 2, 3};
    const std::vector<int> noise_levels = {0, 1, 10, 50};

    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "  GMT_trend2d Cross-Language Benchmark: C++ vs Python\n";
    std::cout << "  num_points = " << num_points << ", n_runs = " << n_runs << "\n";
    std::cout << "================================================================================\n";
    std::cout << "\n";

    // Table header
    std::cout << std::left
              << std::setw(6) << "Rank"
              << std::setw(14) << "Noise Level"
              << std::setw(18) << "C++ Time [ms]"
              << std::setw(18) << "Python Time [ms]"
              << std::setw(22) << "Speedup (C++ vs Py)"
              << std::setw(14) << "Result"
              << "\n";
    std::cout << std::string(92, '-') << "\n";

    for (auto rank: ranks) {
        for (auto noise_level: noise_levels) {
            // Generate data once (same seed for both)
            auto data = generate_data(rank, num_points, noise_level);

            // C++ time
            double cpp_ms = measure_cpp_time(data, rank, n_runs);

            // Python time
            double py_ms = measure_python_time(rank, num_points, noise_level, n_runs);

            // Compute speedup
            std::string speedup_str;
            std::string result_str;
            if (py_ms > 0.0 && cpp_ms > 0.0) {
                double ratio = py_ms / cpp_ms;
                double pct = (ratio - 1.0) * 100.0;
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(1) << ratio << "x";
                if (pct > 0) {
                    ss << " (+" << std::fixed << std::setprecision(0) << pct << "%)";
                } else {
                    ss << " (" << std::fixed << std::setprecision(0) << pct << "%)";
                }
                speedup_str = ss.str();

                if (ratio > 1.0) {
                    result_str = "C++ FASTER";
                } else if (ratio < 1.0) {
                    result_str = "Python FASTER";
                } else {
                    result_str = "EQUAL";
                }
            } else {
                speedup_str = "N/A";
                result_str = "ERROR";
            }

            std::cout << std::left
                      << std::setw(6) << rank
                      << std::setw(14) << noise_level
                      << std::setw(18) << std::fixed << std::setprecision(3) << cpp_ms
                      << std::setw(18) << std::fixed << std::setprecision(3) << py_ms
                      << std::setw(22) << speedup_str
                      << std::setw(14) << result_str
                      << "\n";
        }
    }

    std::cout << std::string(92, '-') << "\n";
    std::cout << "\n";
    std::cout << "NOTE: Speedup > 1.0x means C++ is faster than Python.\n";
    std::cout << "      Percentage shows how much faster C++ is relative to Python.\n";
    std::cout << "\n";

    return 0;
}
