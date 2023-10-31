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

#pragma once

#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include <np/Exception.hpp>
#include <sklearn/Exception.hpp>

#define SKLEARN_THROW_UNLESS(cond, message) \
    if (!(cond)) throw sklearn::RuntimeError(message);

#define SKLEARN_THROW_UNLESS_WITH_ARG(cond, message, arg) \
    if (!(cond)) throw sklearn::Exception(message, arg);

#define SKLEARN_THROW_CONSTEXPR_UNLESS(cond, message) \
    if constexpr (!(cond)) throw sklearn::RuntimeError(message);

#define SKLEARN_THROW_CONSTEXPR_UNLESS_WITH_ARG(cond, message, arg) \
    if constexpr (!(cond)) throw sklearn::Exception(message, arg);

// Macro to throw any standard exception with stack trace appended
#define SKLEARN_THROW_WITH_STACKTRACE(exception_type, message) \
    throw exception_type(sklearn::internal::addStackTrace(message))

namespace sklearn {
    namespace internal {
        // Helper function to add stack trace to a message
        inline std::string addStackTrace(const std::string &message) {
            return message + "\nStack trace:\n" + np::getStackTrace();
        }

        inline std::string getLastError() {
#ifdef WIN32
            DWORD lastError = ::GetLastError();
            if (lastError == 0)
                return std::string();

            LPSTR messageBuffer = nullptr;
            size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                         NULL, lastError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR) &messageBuffer, 0, NULL);

            std::string message(messageBuffer, size);

            LocalFree(messageBuffer);

            return message;
#else
            return std::strerror(errno);
#endif
        }
    }// namespace internal

    class Exception : public std::runtime_error {
    public:
        inline explicit Exception(const std::string &message)
            : std::runtime_error(internal::addStackTrace(message)) {
        }

        inline Exception(const std::string &message, const std::string &arg)
            : std::runtime_error(internal::addStackTrace(message + arg + ", Error: " + internal::getLastError())) {
        }
    };

    class RuntimeError : public Exception {
    public:
        explicit RuntimeError(const std::string &message) : Exception(message) {}
    };

    class InvalidArgumentError : public Exception {
    public:
        explicit InvalidArgumentError(const std::string &message) : Exception(message) {}
    };

    class NotFittedError : public Exception {
    public:
        explicit NotFittedError(const std::string &message) : Exception(message) {}
    };

    class NotImplementedError : public Exception {
    public:
        explicit NotImplementedError(const std::string &message) : Exception(message) {}
    };

}// namespace sklearn
