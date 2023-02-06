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
#include <sklearn/preprocessing/StandardScaler.hpp>

#include <SklearnTest.hpp>

using namespace sklearn;

class StandardScalerTest : public SklearnTest {
protected:
};

TEST_F(StandardScalerTest, standardScalerTest2x4) {
    using namespace preprocessing;
    using namespace np;
    float_ X_train_arr[4][2] = {{0., 0.},
                                {0., 0.},
                                {1., 1.},
                                {1., 1.}};
    Array<float_> X_train{X_train_arr};
    auto scaler = StandardScaler();
    scaler.fit(X_train);

    Array<float_> scaler_mean_sample{0.5, 0.5};
    compare(scaler.mean_(), scaler_mean_sample);

    Array<float_> scaler_var_sample{0.25, 0.25};
    compare(scaler.var_(), scaler_var_sample);

    Array<float_> scaler_scale_sample{0.5, 0.5};
    compare(scaler.scale_(), scaler_scale_sample);

    auto X_scaled = scaler.transform(X_train);
    float_ X_scaled_arr[4][2] = {{-1.0, -1.0},
                                 {-1.0, -1.0},
                                 {1.0, 1.0},
                                 {1.0, 1.0}};
    Array<float_> X_scaled_sample{X_scaled_arr};
    compare(X_scaled, X_scaled_sample);

    X_scaled = scaler.transform(Array<np::float_>{2., 2.});
    compare(X_scaled, Array<np::float_>{3, 3});
}

TEST_F(StandardScalerTest, standardScalerTest3x3) {
    using namespace preprocessing;
    using namespace np;
    float_ X_train_arr[3][3] = {{1., -1., 2.},
                                {2., 0., 0.},
                                {0., 1., -1.}};
    Array<float_> X_train{X_train_arr};
    auto scaler = StandardScaler();
    scaler.fit(X_train);
    Array<float_> scaler_mean_sample{1., 0., 0.33333333333333331};
    compare(scaler.mean_(), scaler_mean_sample);
    auto A = scaler.var_()[2];
    Array<float_> scaler_var_sample{0.66666666666666663, 0.66666666666666663, 1.5555555555555556};
    compare(scaler.var_(), scaler_var_sample);

    Array<float_> scaler_scale_sample{0.81649658092772603, 0.81649658092772603, 1.247219128924647};
    compare(scaler.scale_(), scaler_scale_sample);

    auto X_scaled = scaler.transform(X_train);
    float_ X_scaled_arr[3][3] = {{0., -1.2247448713915889, 1.3363062095621221},
                                 {1.2247448713915889, 0., -0.2672612419124244},
                                 {-1.2247448713915889, 1.2247448713915889, -1.0690449676496976}};
    Array<float_> X_scaled_sample{X_scaled_arr};
    compare(X_scaled, X_scaled_sample);
}
