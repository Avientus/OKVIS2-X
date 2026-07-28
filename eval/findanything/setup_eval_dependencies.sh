#!/bin/bash

# Concept-Graphs Repository
REPO_1_URL="https://github.com/concept-graphs/concept-graphs.git"
REPO_1_HASH="93277a02bd89171f8121e84203121cf7af9ebb5d"
REPO_1_NAME="concept-graphs"
REPO_1_DIR="conceptgraphs"


# Chamferdist Repository
REPO_2_URL="https://github.com/krrish94/chamferdist.git"
REPO_2_HASH="ee753899b583bdc06bd2505dc162fd54ba0c6c5b"
REPO_2_NAME="chamferdist"
REPO_2_DIR="chamferdist"

# ==========================================

# Function to download a specific repo at a specific hash
download_git_at_hash() {
    local URL=$1
    local HASH=$2
    local NAME=$3
    local DIR=$4

    echo "--- Preparing to download into: $DIR ---"

    # clone repo
    git clone "$URL"
    # checkout commit
    mv "$NAME" "$DIR"
    cd "$DIR"
    git checkout "$HASH"
    cd ..
    echo ""
}

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed."
    exit 1
fi

# Execute the downloads
download_git_at_hash "$REPO_1_URL" "$REPO_1_HASH" "$REPO_1_NAME" "$REPO_1_DIR"
download_git_at_hash "$REPO_2_URL" "$REPO_2_HASH" "$REPO_2_NAME" "$REPO_2_DIR"

echo "Done!"