#!/bin/bash
set -euo pipefail
SWIPE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SWIPE_BUILD="${SWIPE_BUILD_DIR:-$SWIPE_ROOT/macos/.build}"
SWIPE_APP="$SWIPE_ROOT/dist/Swipe.app"
swift build --package-path "$SWIPE_ROOT/macos" -c release --scratch-path "$SWIPE_BUILD"
SWIPE_BIN="$(swift build --package-path "$SWIPE_ROOT/macos" -c release --scratch-path "$SWIPE_BUILD" --show-bin-path)"
mkdir -p "$SWIPE_APP/Contents/MacOS" "$SWIPE_APP/Contents/Resources"
# Replace the inode instead of overwriting executable pages in a running app.
cp "$SWIPE_BIN/Swipe" "$SWIPE_APP/Contents/MacOS/.Swipe.next"
mv -f "$SWIPE_APP/Contents/MacOS/.Swipe.next" "$SWIPE_APP/Contents/MacOS/Swipe"
python3 - "$SWIPE_APP" "$SWIPE_ROOT" <<'PY'
import json,plistlib,sys
from pathlib import Path
app,root=map(Path,sys.argv[1:])
info={'CFBundleIdentifier':'io.github.twis06.swipe','CFBundleName':'Swipe','CFBundleDisplayName':'Swipe','CFBundleExecutable':'Swipe','CFBundlePackageType':'APPL','CFBundleShortVersionString':'0.2.0','CFBundleVersion':'2','LSMinimumSystemVersion':'14.0','LSUIElement':True,'NSHighResolutionCapable':True,'NSMicrophoneUsageDescription':'Swipe listens to finger movements on your MacBook speaker grille to trigger your chosen actions. Audio is processed locally and is not saved.'}
with (app/'Contents/Info.plist').open('wb') as f:plistlib.dump(info,f)
(app/'Contents/Resources/runtime.json').write_text(json.dumps({'project':str(root)},indent=2))
PY
codesign --force --sign - --identifier io.github.twis06.swipe "$SWIPE_APP"
codesign --verify --strict "$SWIPE_APP"
echo "Built $SWIPE_APP"
echo 'Open with: open dist/Swipe.app'
