#!/usr/bin/env bash
# Fetch the scikit-learn user-guide .rst sources at a pinned commit (BSD-3 licensed).
set -euo pipefail
COMMIT=d685424bec8f8585d8017c1aab1db7535d3fa03d
DEST="$(dirname "$0")/../data/sklearn_docs"
[ -d "$DEST/.git" ] && { echo "already downloaded"; exit 0; }
git clone --filter=blob:none --no-checkout https://github.com/scikit-learn/scikit-learn.git "$DEST"
git -C "$DEST" sparse-checkout set doc/modules
git -C "$DEST" checkout "$COMMIT"
