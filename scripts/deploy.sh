#!/bin/bash
#
# deploy.sh — Upload the pre-built package archive to a remote Artifactory
#             repository.
#
# This script expects that build.sh has already been run successfully so that
# the .tgz archive exists under the package/ directory.  It then uploads that
# archive to the JFrog Artifactory instance using HTTP PUT (curl -T).
#
# Usage:
#   ./scripts/deploy.sh
#
# Prerequisites:
#   - build.sh must have been executed first.
#   - The environment variables USERNAME and PASSWORD must be set to valid
#     Artifactory credentials (e.g., exported in the shell or sourced from a
#     secure file).
#
# Environment:
#   Reads version_major, version_minor, version_patch, and package_name
#   from the build.properties file at the project root.
#
#   USERNAME  — Artifactory user name (must be set externally)
#   PASSWORD  — Artifactory password or API token (must be set externally)

# Resolve the project root directory (parent of the scripts/ folder).
ROOT_DIR="$(readlink -f $(dirname $BASH_SOURCE)/..)"

# Load version and package-name metadata.
source ${ROOT_DIR}/build.properties

# ---------------------------------------------------------------------------
# Paths and remote URL
# ---------------------------------------------------------------------------
PACKAGE_ROOT=${ROOT_DIR}/package
PACKAGE_VERSION=${version_major}.${version_minor}.${version_patch}
PACKAGE_NAME=${package_name}
PACKAGE_FULLNAME=${PACKAGE_NAME}-${PACKAGE_VERSION}.tgz
PACKAGE_PATH=${PACKAGE_ROOT}/${PACKAGE_FULLNAME}
URL=https://mgorshkov.jfrog.io/artifactory/default-generic-local/$PACKAGE_NAME/$PACKAGE_FULLNAME

# Upload the archive using HTTP PUT with basic authentication.
curl -T $PACKAGE_PATH -u$USERNAME:$PASSWORD "$URL"
