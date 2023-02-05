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

#include <sklearn/datasets/datasets.hpp>
#include <sklearn/neighbors/KNeighborsClassifier.hpp>

int main(int, char **) {
    using namespace sklearn::datasets;
    using namespace sklearn::neighbors;

    auto iris = load_iris();
    auto data = iris.data();
    auto target = iris.target();

    auto kn = sklearn::neighbors::KNeighborsClassifier<np::float_, np::short_>{};
    auto d_slice = data[0];
    np::Shape sh_d_slice{1, d_slice.shape()[0]};
    auto d_slice_reshaped = d_slice.reshape(sh_d_slice);
    std::cout << "Prediction: " << kn.fit_predict(data["1:"], target["1:"], d_slice_reshaped) << std::endl;
    std::cout << "target: " << target[0] << std::endl;// 0
    return 0;
}
