#!/bin/bash
#
# build.sh — Build, package, and archive the sklearn C++ library.
#
# This script performs three steps in sequence:
#   1. Compile the project with CMake in Release mode.
#   2. Assemble a distribution package containing headers, sources,
#      samples, tests, and the compiled shared library.
#   3. Compress the package into a .tgz archive.
#
# Usage:
#   ./scripts/build.sh [additional CMake arguments...]
#
# Environment:
#   Reads version_major, version_minor, version_patch, and package_name
#   from the build.properties file at the project root.
#
# Output:
#   package/<package_name>-<version>.tgz   — the distributable archive
#   package/<package_name>                  — symlink to the unpacked directory

# Resolve the project root directory (parent of the scripts/ folder).
ROOT_DIR="$(readlink -f $(dirname $BASH_SOURCE)/..)"

# Load version and package-name metadata.
source ${ROOT_DIR}/build.properties

# ---------------------------------------------------------------------------
# Paths derived from build.properties
# ---------------------------------------------------------------------------
PACKAGE_ROOT=${ROOT_DIR}/package
PACKAGE_VERSION=${version_major}.${version_minor}.${version_patch}
PACKAGE_NAME=${package_name}
PACKAGE_FULLNAME=${PACKAGE_NAME}-${PACKAGE_VERSION}
PACKAGE_PATH=${PACKAGE_ROOT}/${PACKAGE_FULLNAME}
PACKAGE_TAR=${PACKAGE_PATH}.tgz

echo "PACKAGE_TAR: $PACKAGE_TAR"

# ---------------------------------------------------------------------------
# Step 1 — Build the project with CMake
# ---------------------------------------------------------------------------
function build_package() {
    rm -rf ${ROOT_DIR}/build || return 1
    mkdir -p ${ROOT_DIR}/build || return 1
    cd ${ROOT_DIR}/build || return 1
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PACKAGE_PATH} "$@" || return 1
    cmake --build . --config Release -j
}

# ---------------------------------------------------------------------------
# Copy source/header files from a project folder into the package directory.
#
# Arguments:
#   $1 — source path relative to ROOT_DIR
#   $2 — destination path (defaults to PACKAGE_PATH/$1)
#
# Only files matching the extensions .hpp, .cpp, .md, .csv, .npy, .sh, .txt
# are copied (one level deep).
# ---------------------------------------------------------------------------
function copy() {
    local path="$1"
    local dest="${2:-$PACKAGE_PATH/$path}"

    mkdir -p $dest

    find $path -maxdepth 1 -type f -regex ".*\.\(hpp\|cpp\|md\|csv\|npy\|sh\|txt\)$" -exec cp {} $dest \;
}

# ---------------------------------------------------------------------------
# Step 2 — Assemble the distribution package
# ---------------------------------------------------------------------------
function create_package() {
    rm -rf ${PACKAGE_PATH}
    rm -f ${PACKAGE_TAR}
    mkdir -p ${PACKAGE_PATH}
    cd ${PACKAGE_ROOT} || return 1
    rm -f ${PACKAGE_NAME}
    ln -s ${PACKAGE_FULLNAME} ${PACKAGE_NAME}
    cd ${ROOT_DIR} || return 1

    # List of project sub-directories whose source/header files will be
    # included in the package.
    FOLDERS=(
        .
        include
        include/sklearn/datasets
        include/sklearn/metrics
        include/sklearn/model_selection
        include/sklearn/neighbors
        samples
        samples/neighbors
        samples/neighbors/diabetes
        samples/neighbors/iris
        scripts
        src
        src/datasets
        src/model_selection
        unit_tests
        unit_tests/include
        unit_tests/src
    )
    for folder in "${FOLDERS[@]}"; do
        copy $folder
    done
    mkdir -p $PACKAGE_PATH/lib
    # Copy the compiled shared library into the package.
    copy build/src/libsklearn.* $PACKAGE_PATH/lib

    return 0
}

# ---------------------------------------------------------------------------
# Step 3 — Compress the package directory into a .tgz archive
# ---------------------------------------------------------------------------
function zip_package() {
    rm -f ${PACKAGE_TAR} || return 1
    tar zcf ${PACKAGE_TAR} -C "$(dirname ${PACKAGE_PATH})" "$(basename ${PACKAGE_PATH})"

    return 0
}

# ---------------------------------------------------------------------------
# Entry point — run all three steps in order
# ---------------------------------------------------------------------------
function main() {
    build_package || return 1
    create_package || return 1
    zip_package
}

main
