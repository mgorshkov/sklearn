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

#include <sklearn/datasets/Iris.hpp>

namespace sklearn {
    namespace datasets {

        using namespace np;

        float_ Iris::kData[Iris::kSamples][Iris::kDims] =
                {{5.1, 3.5, 1.4, 0.2},
                 {4.9, 3., 1.4, 0.2},
                 {4.7, 3.2, 1.3, 0.2},
                 {4.6, 3.1, 1.5, 0.2},
                 {5., 3.6, 1.4, 0.2},
                 {5.4, 3.9, 1.7, 0.4},
                 {4.6, 3.4, 1.4, 0.3},
                 {5., 3.4, 1.5, 0.2},
                 {4.4, 2.9, 1.4, 0.2},
                 {4.9, 3.1, 1.5, 0.1},
                 {5.4, 3.7, 1.5, 0.2},
                 {4.8, 3.4, 1.6, 0.2},
                 {4.8, 3., 1.4, 0.1},
                 {4.3, 3., 1.1, 0.1},
                 {5.8, 4., 1.2, 0.2},
                 {5.7, 4.4, 1.5, 0.4},
                 {5.4, 3.9, 1.3, 0.4},
                 {5.1, 3.5, 1.4, 0.3},
                 {5.7, 3.8, 1.7, 0.3},
                 {5.1, 3.8, 1.5, 0.3},
                 {5.4, 3.4, 1.7, 0.2},
                 {5.1, 3.7, 1.5, 0.4},
                 {4.6, 3.6, 1., 0.2},
                 {5.1, 3.3, 1.7, 0.5},
                 {4.8, 3.4, 1.9, 0.2},
                 {5., 3., 1.6, 0.2},
                 {5., 3.4, 1.6, 0.4},
                 {5.2, 3.5, 1.5, 0.2},
                 {5.2, 3.4, 1.4, 0.2},
                 {4.7, 3.2, 1.6, 0.2},
                 {4.8, 3.1, 1.6, 0.2},
                 {5.4, 3.4, 1.5, 0.4},
                 {5.2, 4.1, 1.5, 0.1},
                 {5.5, 4.2, 1.4, 0.2},
                 {4.9, 3.1, 1.5, 0.2},
                 {5., 3.2, 1.2, 0.2},
                 {5.5, 3.5, 1.3, 0.2},
                 {4.9, 3.6, 1.4, 0.1},
                 {4.4, 3., 1.3, 0.2},
                 {5.1, 3.4, 1.5, 0.2},
                 {5., 3.5, 1.3, 0.3},
                 {4.5, 2.3, 1.3, 0.3},
                 {4.4, 3.2, 1.3, 0.2},
                 {5., 3.5, 1.6, 0.6},
                 {5.1, 3.8, 1.9, 0.4},
                 {4.8, 3., 1.4, 0.3},
                 {5.1, 3.8, 1.6, 0.2},
                 {4.6, 3.2, 1.4, 0.2},
                 {5.3, 3.7, 1.5, 0.2},
                 {5., 3.3, 1.4, 0.2},
                 {7., 3.2, 4.7, 1.4},
                 {6.4, 3.2, 4.5, 1.5},
                 {6.9, 3.1, 4.9, 1.5},
                 {5.5, 2.3, 4., 1.3},
                 {6.5, 2.8, 4.6, 1.5},
                 {5.7, 2.8, 4.5, 1.3},
                 {6.3, 3.3, 4.7, 1.6},
                 {4.9, 2.4, 3.3, 1.},
                 {6.6, 2.9, 4.6, 1.3},
                 {5.2, 2.7, 3.9, 1.4},
                 {5., 2., 3.5, 1.},
                 {5.9, 3., 4.2, 1.5},
                 {6., 2.2, 4., 1.},
                 {6.1, 2.9, 4.7, 1.4},
                 {5.6, 2.9, 3.6, 1.3},
                 {6.7, 3.1, 4.4, 1.4},
                 {5.6, 3., 4.5, 1.5},
                 {5.8, 2.7, 4.1, 1.},
                 {6.2, 2.2, 4.5, 1.5},
                 {5.6, 2.5, 3.9, 1.1},
                 {5.9, 3.2, 4.8, 1.8},
                 {6.1, 2.8, 4., 1.3},
                 {6.3, 2.5, 4.9, 1.5},
                 {6.1, 2.8, 4.7, 1.2},
                 {6.4, 2.9, 4.3, 1.3},
                 {6.6, 3., 4.4, 1.4},
                 {6.8, 2.8, 4.8, 1.4},
                 {6.7, 3., 5., 1.7},
                 {6., 2.9, 4.5, 1.5},
                 {5.7, 2.6, 3.5, 1.},
                 {5.5, 2.4, 3.8, 1.1},
                 {5.5, 2.4, 3.7, 1.},
                 {5.8, 2.7, 3.9, 1.2},
                 {6., 2.7, 5.1, 1.6},
                 {5.4, 3., 4.5, 1.5},
                 {6., 3.4, 4.5, 1.6},
                 {6.7, 3.1, 4.7, 1.5},
                 {6.3, 2.3, 4.4, 1.3},
                 {5.6, 3., 4.1, 1.3},
                 {5.5, 2.5, 4., 1.3},
                 {5.5, 2.6, 4.4, 1.2},
                 {6.1, 3., 4.6, 1.4},
                 {5.8, 2.6, 4., 1.2},
                 {5., 2.3, 3.3, 1.},
                 {5.6, 2.7, 4.2, 1.3},
                 {5.7, 3., 4.2, 1.2},
                 {5.7, 2.9, 4.2, 1.3},
                 {6.2, 2.9, 4.3, 1.3},
                 {5.1, 2.5, 3., 1.1},
                 {5.7, 2.8, 4.1, 1.3},
                 {6.3, 3.3, 6., 2.5},
                 {5.8, 2.7, 5.1, 1.9},
                 {7.1, 3., 5.9, 2.1},
                 {6.3, 2.9, 5.6, 1.8},
                 {6.5, 3., 5.8, 2.2},
                 {7.6, 3., 6.6, 2.1},
                 {4.9, 2.5, 4.5, 1.7},
                 {7.3, 2.9, 6.3, 1.8},
                 {6.7, 2.5, 5.8, 1.8},
                 {7.2, 3.6, 6.1, 2.5},
                 {6.5, 3.2, 5.1, 2.},
                 {6.4, 2.7, 5.3, 1.9},
                 {6.8, 3., 5.5, 2.1},
                 {5.7, 2.5, 5., 2.},
                 {5.8, 2.8, 5.1, 2.4},
                 {6.4, 3.2, 5.3, 2.3},
                 {6.5, 3., 5.5, 1.8},
                 {7.7, 3.8, 6.7, 2.2},
                 {7.7, 2.6, 6.9, 2.3},
                 {6., 2.2, 5., 1.5},
                 {6.9, 3.2, 5.7, 2.3},
                 {5.6, 2.8, 4.9, 2.},
                 {7.7, 2.8, 6.7, 2.},
                 {6.3, 2.7, 4.9, 1.8},
                 {6.7, 3.3, 5.7, 2.1},
                 {7.2, 3.2, 6., 1.8},
                 {6.2, 2.8, 4.8, 1.8},
                 {6.1, 3., 4.9, 1.8},
                 {6.4, 2.8, 5.6, 2.1},
                 {7.2, 3., 5.8, 1.6},
                 {7.4, 2.8, 6.1, 1.9},
                 {7.9, 3.8, 6.4, 2.},
                 {6.4, 2.8, 5.6, 2.2},
                 {6.3, 2.8, 5.1, 1.5},
                 {6.1, 2.6, 5.6, 1.4},
                 {7.7, 3., 6.1, 2.3},
                 {6.3, 3.4, 5.6, 2.4},
                 {6.4, 3.1, 5.5, 1.8},
                 {6., 3., 4.8, 1.8},
                 {6.9, 3.1, 5.4, 2.1},
                 {6.7, 3.1, 5.6, 2.4},
                 {6.9, 3.1, 5.1, 2.3},
                 {5.8, 2.7, 5.1, 1.9},
                 {6.8, 3.2, 5.9, 2.3},
                 {6.7, 3.3, 5.7, 2.5},
                 {6.7, 3., 5.2, 2.3},
                 {6.3, 2.5, 5., 1.9},
                 {6.5, 3., 5.2, 2.},
                 {6.2, 3.4, 5.4, 2.3},
                 {5.9, 3., 5.1, 1.8}};

        int_ Iris::kTarget[Iris::kSamples] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                              2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                              2, 2};

        void Iris::load() {
            m_data = np::Array<np::float_, kSamples * kDims>{kData};
            m_target = np::Array<np::int_, kSamples>{kTarget};
        }
    }// namespace datasets
}// namespace sklearn