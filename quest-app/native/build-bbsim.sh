#!/usr/bin/env bash
# Builds bbsim.c (the app's sim) for the Quest against build-mujoco.sh's libmujoco, building that first if it's missing.
# Output: quest-app/native/build/libbbsim.so
# Usage: quest-app/native/build-bbsim.sh [path to Unity's AndroidPlayer folder]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/build/mujoco/lib/libmujoco.so" ] || "$HERE/build-mujoco.sh" "$@"
BUILD="$HERE/build"
# The NDK's Windows tools need C:/ style paths, not Git Bash's /c/
if command -v cygpath > /dev/null; then
  HERE="$(cygpath -m "$HERE")"
  BUILD="$(cygpath -m "$BUILD")"
fi
if [ $# -ge 1 ]; then
  PLAYER="$1"
else
  PLAYER="$(ls -d "/c/Program Files/Unity/Hub/Editor/"*/Editor/Data/PlaybackEngines/AndroidPlayer \
    /Applications/Unity/Hub/Editor/*/PlaybackEngines/AndroidPlayer 2>/dev/null | sort | tail -1 || true)"
fi
CC="$(ls "$PLAYER"/NDK/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android32-clang* | head -1)"

"$CC" -O2 -shared -fPIC -fvisibility=hidden -I"$BUILD/mujoco-src/include" "$HERE/bbsim.c" \
  -L"$BUILD/mujoco/lib" -lmujoco -lm -o "$BUILD/libbbsim.so"
