#!/usr/bin/env bash
# Builds the Quest app: bbsim and MuJoCo for the headset, sim.py's model, then the APK through the Unity CLI
# (Assets/Editor/QuestBuild.cs), and installs and starts it when the Quest is on USB.
# Usage: quest-app/build.sh [--no-install]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PLAYER="$(ls -d "/c/Program Files/Unity/Hub/Editor/"*/Editor/Data/PlaybackEngines/AndroidPlayer \
  /Applications/Unity/Hub/Editor/*/PlaybackEngines/AndroidPlayer 2>/dev/null | sort | tail -1 || true)"
UNITY="$(ls "/c/Program Files/Unity/Hub/Editor/"*/Editor/Unity.exe \
  /Applications/Unity/Hub/Editor/*/Unity.app/Contents/MacOS/Unity 2>/dev/null | sort | tail -1 || true)"
[ -n "$UNITY" ] && [ -d "$PLAYER" ] || { echo "Unity with Android support not found"; exit 1; }
PY="$ROOT/emg-reading/.venv/Scripts/python"
[ -x "$PY" ] || PY="$ROOT/emg-reading/.venv/bin/python"
ADB="$PLAYER/SDK/platform-tools/adb"

"$HERE/native/build-bbsim.sh" "$PLAYER"
"$PY" "$HERE/native/export_model.py"
PLUGINS="$HERE/Assets/Plugins/Android/arm64-v8a"
mkdir -p "$PLUGINS"
cp "$HERE/native/build/mujoco/lib/libmujoco.so" "$HERE/native/build/libbbsim.so" "$PLUGINS/"

mkdir -p "$HERE/Logs"
LOG="$HERE/Logs/build.log"
if ! "$UNITY" -batchmode -quit -nographics -projectPath "$HERE" -executeMethod QuestBuild.Build -logFile "$LOG"; then
  grep -E "error|Error|failed" "$LOG" | grep -v "^\s*$" | tail -20
  echo "build failed, full log in $LOG"
  exit 1
fi
APK="$HERE/Build/bb-contact.apk"
echo "built $APK"

[ "${1:-}" = "--no-install" ] && exit 0
if "$ADB" get-state > /dev/null 2>&1; then
  "$ADB" install -r "$APK"
  "$ADB" shell am start -n com.betterbionics.contact/com.unity3d.player.UnityPlayerActivity
else
  echo "no Quest on USB, not installed"
fi
