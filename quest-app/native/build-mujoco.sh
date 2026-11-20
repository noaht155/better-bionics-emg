#!/usr/bin/env bash
# Builds MuJoCo (same version as emg-reading's) for the Quest 3 (Android arm64) with the NDK that Unity installs.
# Output: quest-app/native/build/mujoco/lib/libmujoco.so
# Usage: quest-app/native/build-mujoco.sh [path to Unity's AndroidPlayer folder]
set -euo pipefail

VERSION=4.0.1
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$HERE/build"

if [ $# -ge 1 ]; then
  PLAYER="$1"
else
  PLAYER="$(ls -d "/c/Program Files/Unity/Hub/Editor/"*/Editor/Data/PlaybackEngines/AndroidPlayer \
    /Applications/Unity/Hub/Editor/*/PlaybackEngines/AndroidPlayer 2>/dev/null | sort | tail -1 || true)"
fi
[ -d "$PLAYER/NDK" ] || { echo "Unity's Android player (with NDK) not found, pass its path"; exit 1; }
CMAKE_BIN="$(ls -d "$PLAYER"/SDK/cmake/*/bin | tail -1)"
export PATH="$CMAKE_BIN:$PATH"
NDK_VERSION="$(sed -n 's/^Pkg.Revision = \([0-9]*\.[0-9]*\).*/\1/p' "$PLAYER/NDK/source.properties")"

mkdir -p "$BUILD"
[ -d "$BUILD/mujoco-src" ] || git clone -q --depth 1 --branch "$VERSION" https://github.com/google-deepmind/mujoco.git "$BUILD/mujoco-src"

# CMAKE_ANDROID_NDK_VERSION: without it CMake 3.22 assumes an old NDK and links with gold when LTO is on, which
# NDK 27 no longer has. The C++ runtime is linked by hand because libmujoco is linked by the C driver.
cmake -S "$BUILD/mujoco-src" -B "$BUILD/mujoco" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$CMAKE_BIN/ninja" \
  -DCMAKE_TOOLCHAIN_FILE="$PLAYER/NDK/build/cmake/android.toolchain.cmake" \
  -DCMAKE_ANDROID_NDK_VERSION="$NDK_VERSION" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-32 -DANDROID_STL=c++_static \
  "-DCMAKE_SHARED_LINKER_FLAGS=-lc++_static -lc++abi" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_SIMULATE=OFF -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_TEST_PYTHON_UTIL=OFF
cmake --build "$BUILD/mujoco" --target mujoco
ls -l "$BUILD/mujoco/lib/libmujoco.so"
