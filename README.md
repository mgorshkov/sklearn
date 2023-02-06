[![Build status](https://ci.appveyor.com/api/projects/status/2pl7od2nosslyqay/branch/main?svg=true)](https://ci.appveyor.com/project/mgorshkov/sklearn/branch/main)

# About
ML Methods from scikit-learn library.

# Description
Implements some ML Methods from scikit-learn library.

# Requirements
Any C++20-compatible compiler:
* gcc 10 or higher
* clang 6 or higher
* Visual Studio 2019 or higher

# Repo
```
git clone https://github.com/mgorshkov/sklearn.git
```

# Build unit tests and sample
## Linux/MacOS
```
mkdir build && cd build
cmake ..
cmake --build .
```
## Windows
```
mkdir build && cd build
cmake ..
cmake --build . --config Release
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

# Usage example (samples/neighbors/iris)
```
#include <iostream>

#include <sklearn/datasets/datasets.hpp>
#include <sklearn/metrics/accuracy_score.hpp>
#include <sklearn/model_selection/train_test_split.hpp>
#include <sklearn/neighbors/KNeighborsClassifier.hpp>
#include <sklearn/preprocessing/StandardScaler.hpp>

int main(int, char **) {
    using namespace sklearn::metrics;
    using namespace sklearn::datasets;
    using namespace sklearn::model_selection;
    using namespace sklearn::neighbors;
    using namespace sklearn::preprocessing;

    auto iris = load_iris();
    auto data = iris.data();
    auto target = iris.target();

    auto [X_train, X_test, y_train, y_test] =
            train_test_split<np::float_, np::int_, 600, 150>({.X = data, .y = target, .test_size = 0.2, .random_state = 42});
    auto sc_X = StandardScaler{};
    X_train = sc_X.fit_transform(X_train);
    X_test = sc_X.transform(X_test);

    auto kn = KNeighborsClassifier<np::float_, np::int_>{{.n_neighbors = 13,
                                                          .p = 2,
                                                          .metric = sklearn::metrics::DistanceMetricType::kEuclidean}};
    kn.fit(X_train, y_train);

    auto y_pred = kn.predict(X_test);
    std::cout << "Prediction: " << y_pred << std::endl;
    std::cout << "Target: " << y_test << std::endl;

    auto score = accuracy_score<np::int_>(y_test, y_pred);
    std::cout << "Score: " << score << std::endl;

    return 0;
}
```
# How to build the sample

1. Clone the repo
```
git clone https://github.com/mgorshkov/sklearn.git
```
2. cd samples/neighbors/iris
```
cd samples/neighbors/iris
```
3. Make build dir
```
mkdir -p build-release && cd build-release
```
4. Configure cmake
```
cmake ..
```
5. Build
## Linux/MacOS
```
cmake --build .
```
## Windows
```
cmake --build . --config Release
```
6. Run the app
```
$ ./neighbors_iris
Prediction: [1 2 1 0 2 0 2 0 0 2 0 1 0 2 1 1 0 0 0 2 0 2 2 2 0 1 2 1 2 1]
Target: [1 2 1 0 2 0 2 0 0 2 0 1 0 2 1 1 0 0 0 2 0 2 2 2 0 1 2 1 2 1]
Score: 1
```

# Usage example (samples/neighbors/diabetes)
```
#include <iostream>

#include <pd/core/frame/DataFrame/DataFrameStreamIo.hpp>
#include <pd/read_csv.hpp>

#include <sklearn/metrics/accuracy_score.hpp>
#include <sklearn/metrics/confusion_matrix.hpp>
#include <sklearn/metrics/f1_score.hpp>
#include <sklearn/model_selection/train_test_split.hpp>
#include <sklearn/neighbors/KNeighborsClassifier.hpp>
#include <sklearn/preprocessing/StandardScaler.hpp>

int main(int, char **) {
    using namespace pd;
    using namespace sklearn::model_selection;
    using namespace sklearn::neighbors;
    using namespace sklearn::preprocessing;
    using namespace sklearn::metrics;

    auto data = read_csv("https://raw.githubusercontent.com/adityakumar529/Coursera_Capstone/master/diabetes.csv");
    const char *non_zero[] = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"};
    for (const auto &column: non_zero) {
        data[column] = data[column].replace(0L, np::NaN);
        auto mean = data[column].mean(true);
        data[column] = data[column].replace(np::NaN, mean);
    }

    auto X = data.iloc(":", "0:8");
    auto y = data.iloc(":", "8");
    auto [X_train, X_test, y_train, y_test] = train_test_split({.X = X, .y = y, .test_size = 0.2, .random_state = 42});

    auto sc_X = StandardScaler{};
    X_train = sc_X.fit_transform(X_train);
    X_test = sc_X.transform(X_test);

    auto classifier = KNeighborsClassifier<pd::DataFrame>{{.n_neighbors = 13,
                                                           .p = 2,
                                                           .metric = sklearn::metrics::DistanceMetricType::kEuclidean}};
    classifier.fit(X_train, y_train);
    auto y_pred = classifier.predict(X_test);
    std::cout << "Prediction: " << y_pred << std::endl;
    auto cm = confusion_matrix({.y_true = y_test, .y_pred = y_pred});
    std::cout << cm << std::endl;
    std::cout << f1_score({.y_true = y_test, .y_pred = y_pred}) << std::endl;
    std::cout << accuracy_score(y_test, y_pred) << std::endl;

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
cd samples/neighbors/iris
```
3. Make build dir
```
mkdir -p build-release && cd build-release
```
4. Configure cmake
```
cmake ..
```
5. Build
## Linux/MacOS
```
cmake --build .
```
## Windows
```
cmake --build . --config Release
```
6. Run the app
```
$ ./neighbors_diabetes
Prediction: 	0
0	0
1	0
2	0
3	0
4	1
...
149	1
150	0
151	0
152	0
153	1
154 rows x 1 columns

[[85 15]
 [19 35]]
0.673077
0.779221
```

# Links
* C++ numpy-like template-based array implementation: https://github.com/mgorshkov/np
* Methods from pandas library on top of NP library: https://github.com/mgorshkov/pd
* Scientific methods on top of NP library: https://github.com/mgorshkov/scipy
