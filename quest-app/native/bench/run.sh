#!/usr/bin/env bash
# Builds bench.c for the Quest, pushes it with bbsim, MuJoCo and record.py's files, and runs it. Needs build-bbsim.sh
# and record.py run first, and the Quest on USB with debugging allowed.
# Usage: quest-app/native/bench/run.sh [path to Unity's AndroidPlayer folder]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$(cd "$HERE/.." && pwd)/build"
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
ADB="$PLAYER/SDK/platform-tools/adb"
CC="$(ls "$PLAYER"/NDK/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android32-clang* | head -1)"
DEST=/data/local/tmp/mujoco-bench
export MSYS_NO_PATHCONV=1

"$CC" -O2 -I"$BUILD/mujoco-src/include" "$HERE/bench.c" -L"$BUILD" -L"$BUILD/mujoco/lib" -lbbsim -lmujoco -lm -o "$BUILD/bench/bench"
"$ADB" shell mkdir -p "$DEST"
"$ADB" push "$BUILD/mujoco/lib/libmujoco.so" "$BUILD/libbbsim.so" "$BUILD/bench/bench" "$BUILD/bench/scene.xml" "$BUILD/bench/rule.txt" \
  "$BUILD/bench/ticks.bin" "$DEST/" > /dev/null
"$ADB" shell "cd $DEST && chmod 755 bench && LD_LIBRARY_PATH=. ./bench 3"
