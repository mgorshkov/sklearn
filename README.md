# About
ML Methods from scikit-learn library.

# Description
Implements some ML Methods from scikit-learn library.

# Requirements
Any C++17-compatible compiler:
* gcc 8 or higher
* clang 6 or higher
* Visual Studio 2017 or higher

# Repo
```
git clone https://github.com/mgorshkov/sklearn.git
```

# Build unit tests and sample
```
mkdir build && cd build
cmake ..
cmake --build .
```

# Build docs
```
cmake --build . --target doc
```

Open sklearn/build/doc/html/index.html in your browser.

# Install
```
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=~/sklearn_install
cmake --build . --target install
```

# Usage example (samples/neighbors)
```
#include <iostream>
#include <sklearn/datasets/datasets.hpp>
#include <sklearn/neighbors/KNeighborsClassifier.hpp>

int main(int, char **) {
    using namespace sklearn::datasets;
    using namespace sklearn::neighbors;

    auto iris = load_iris();
    auto data = iris.data();
    auto target = iris.target();

    auto kn = KNeighborsClassifier<np::float_, np::short_>{};
    kn.fit(data["1:"], target["1:"]);
    std::cout << "Prediction: " << kn.predict(data[0]) << std::endl;
    std::cout << "target: " << target[0] << std::endl; // 0
    return 0;
}
```
# How to build the sample

1. Clone the repo
```
git clone https://github.com/mgorshkov/sklearn.git
```
2. cd samples/neighbors
```
cd samples/neighbors
```
3. Make build dir
```
mkdir -p build-release && cd build-release
```
4. Configure cmake
```
cmake -DCMAKE_BUILD_TYPE=Release ..
```
5. Build
```
cmake --build .
```
6. Run the app
```
$./neighbors
```
