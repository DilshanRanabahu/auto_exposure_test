#!/bin/bash
# Auto Exposure Launcher for Daheng Camera on Linux (panther-thullex)
# Target user: thullex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Allow root / sudo to use the current user's X11 display (for cv2.imshow GUI)
if [ -n "$DISPLAY" ]; then
    xhost +local:root >/dev/null 2>&1 || true
    xhost +local:thullex >/dev/null 2>&1 || true
fi

# Stop rms-app if running so the Daheng camera is free
if systemctl is-active --quiet rms-app 2>/dev/null; then
    echo "Stopping rms-app service to release camera..."
    sudo systemctl stop rms-app 2>/dev/null || true
fi

# Run auto_exposure.py with sudo (passing current user's DISPLAY and XAUTHORITY)
echo "Starting Auto Exposure as root..."
sudo env DISPLAY="$DISPLAY" XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" python3 "$DIR/auto_exposure.py" "$@"

EXIT_CODE=$?

echo ""
echo "Auto exposure finished with exit code: $EXIT_CODE"
exit $EXIT_CODE
