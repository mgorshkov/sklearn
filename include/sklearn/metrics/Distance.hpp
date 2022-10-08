/*
ML Methods from scikit-learn library

Copyright (c) 2022 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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

namespace sklearn {
    namespace metrics {
        template <typename DType, np::Size... Sizes>
        class Distance;

        template <typename DType, np::Size SizeT, np::Size... SizeTs>
        class Distance<DType, SizeT, SizeTs...> {
        public:
            virtual ~Distance() = default;
            virtual np::Array<DType, SizeT, SizeTs...> pairwise(const np::Array<DType, SizeT, SizeTs...>& X) = 0;
            virtual np::Array<DType, SizeT, SizeTs...> pairwise(const np::Array<DType, SizeT, SizeTs...>& X, const np::Array<DType, SizeT, SizeTs...>& Y) = 0;
        };

        template <typename DType>
        class Distance<DType> {
        public:
            virtual ~Distance() = default;
            virtual np::Array<DType> pairwise(const np::Array<DType>& X) = 0;
            virtual np::Array<DType> pairwise(const np::Array<DType>& X, const np::Array<DType>& Y) = 0;
        };

        template <typename DType, np::Size... Sizes>
        using DistancePtr = std::shared_ptr<Distance<DType, Sizes...>>;
    }
}