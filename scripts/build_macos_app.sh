#!/bin/bash

set -euo pipefail

# Build a simple macOS .app bundle without code signing
# This bundles the project and a launcher that sets up a venv and runs main.py

APP_NAME="LLMind"
VERSION="${1:-1.0.0}"
PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
DIST_DIR="$PROJECT_ROOT/dist"
APP_DIR="$DIST_DIR/${APP_NAME}.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

echo "Building $APP_NAME.app (version $VERSION) in $DIST_DIR"
rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

# Info.plist
cat > "$CONTENTS_DIR/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>local.${APP_NAME}.app</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
    <key>CFBundleExecutable</key>
    <string>launch</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

# Copy icon if available
if [ -f "$PROJECT_ROOT/favicon/favicon.icns" ]; then
  cp "$PROJECT_ROOT/favicon/favicon.icns" "$RESOURCES_DIR/${APP_NAME}.icns"
fi

# Bundle the project (excluding large/user dirs)
APP_PAYLOAD_DIR="$RESOURCES_DIR/payload"
mkdir -p "$APP_PAYLOAD_DIR"

echo "Copying project files..."
rsync -a --delete \
  --exclude 'venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'dist' \
  --exclude '*.app' \
  --exclude 'data/models' \
  --exclude 'logs' \
  "$PROJECT_ROOT/" "$APP_PAYLOAD_DIR/"

# Launcher script
cat > "$MACOS_DIR/launch" <<'LAUNCH'
#!/bin/bash
set -e

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESOURCES_DIR="$APP_DIR/Resources"
PAYLOAD_DIR="$RESOURCES_DIR/payload"

cd "$PAYLOAD_DIR"

# Set up Python venv inside app sandbox on first run
VENV_DIR="$RESOURCES_DIR/venv"
PYTHON_BIN="$(/usr/bin/python3 -c 'import sys; print(sys.executable)')"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source "$VENV_DIR/bin/activate"
fi

# Start server
export PYTHONUNBUFFERED=1
python main.py
LAUNCH
chmod +x "$MACOS_DIR/launch"

echo "Done. Open $APP_DIR to run ${APP_NAME}.app"


