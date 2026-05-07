#!/bin/bash
# Pre-clone all FetchContent dependencies using blobless clones (much faster than full clones).
# Then run `make build` with FETCHCONTENT_SOURCE_DIR overrides so CMake skips cloning.

set -e

DEPS_DIR="$(cd "$(dirname "$0")" && pwd)/.deps"
mkdir -p "$DEPS_DIR"

# Repo definitions: name|url|commit
REPOS=(
  "repo-cutlass|https://github.com/NVIDIA/cutlass|57e3cfb47a2d9e0d46eb6335c3dc411498efa198"
  "repo-deepgemm|https://github.com/sgl-project/DeepGEMM|35c4bc87713726d048f65275f6f1b551a4e7a6dc"
  "repo-fmt|https://github.com/fmtlib/fmt|553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
  "repo-triton|https://github.com/triton-lang/triton|8f9f695ea8fde23a0c7c88e4ab256634ca27789f"
  "repo-flashinfer|https://github.com/flashinfer-ai/flashinfer.git|bc29697ba20b7e6bdb728ded98f04788e16ee021"
  "repo-flash-attention|https://github.com/sgl-project/sgl-attn|f9af0c2a1d82ab1812e6987e9338363cc2bf0f8d"
  "repo-flash-attention-origin|https://github.com/Dao-AILab/flash-attention.git|203b9b3dba39d5d08dffb49c09aa622984dff07d"
  "repo-mscclpp|https://github.com/microsoft/mscclpp.git|51eca89d20f0cfb3764ccd764338d7b22cd486a6"
  "repo-fast-hadamard-transform|https://github.com/sgl-project/fast-hadamard-transform.git|48f3c13764dc2ec662ade842a4696a90a137f1bc"
)

clone_repo() {
  local name url commit
  IFS='|' read -r name url commit <<< "$1"
  local dir="$DEPS_DIR/$name"

  if [ -d "$dir/.git" ]; then
    echo "[skip] $name already cloned, checking out $commit"
    cd "$dir"
    git fetch --filter=blob:none origin "$commit" 2>/dev/null || true
    git checkout "$commit" --quiet 2>/dev/null || {
      echo "[fetch] $name: fetching commit $commit"
      git fetch origin
      git checkout "$commit" --quiet
    }
  else
    echo "[clone] $name from $url"
    git clone --filter=blob:none --no-checkout "$url" "$dir"
    cd "$dir"
    git checkout "$commit" --quiet
  fi
  echo "[done] $name @ $commit"
}

export -f clone_repo
export DEPS_DIR

echo "=== Pre-cloning ${#REPOS[@]} repos into $DEPS_DIR ==="
echo ""

# Clone in parallel (up to 4 at a time)
printf '%s\n' "${REPOS[@]}" | xargs -P 4 -I {} bash -c 'clone_repo "$@"' _ {}

echo ""
echo "=== All repos cloned ==="
echo ""

# Build cmake define args
CMAKE_DEFINES=""
for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url commit <<< "$entry"
  upper_name=$(echo "$name" | tr '[:lower:]' '[:upper:]')
  CMAKE_DEFINES="$CMAKE_DEFINES -Ccmake.define.FETCHCONTENT_SOURCE_DIR_${upper_name}=$DEPS_DIR/$name"
done

echo "CMAKE_DEFINES: $CMAKE_DEFINES"
echo ""
echo "To build manually, run:"
echo "  cd $(dirname "$0")"
echo "  CMAKE_POLICY_VERSION_MINIMUM=3.5 MAX_JOBS=\$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=\$(nproc) uv build --wheel $CMAKE_DEFINES -Cbuild-dir=build . --verbose --color=always --no-build-isolation"
echo ""
echo "Or run this script with --build to clone + build in one step"

if [ "${1:-}" = "--build" ]; then
  echo "=== Starting build ==="
  cd "$(dirname "$0")"
  rm -rf dist/* 2>/dev/null || true
  CMAKE_POLICY_VERSION_MINIMUM=3.5 MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
    uv build --wheel $CMAKE_DEFINES -Cbuild-dir=build . --verbose --color=always --no-build-isolation
  echo "=== Build complete ==="
  pip3 install dist/*whl --force-reinstall --no-deps
fi
