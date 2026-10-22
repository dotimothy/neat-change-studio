#!/usr/bin/env bash
# The desktop icon's launcher (setup.sh makes the icon): start the web demo in the
# background unless it already runs, then open it in the board's browser.
#   webapp/desktop-launch.sh          # port 8098, or CYWS_PORT
# The demo keeps running after this window closes; stop it with webapp/run.sh --stop.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${CYWS_PORT:-8098}"
URL="https://localhost:${PORT}/"

pause() { [[ -t 0 ]] && read -r -p "Press Enter to close this window. " _; }

if curl -fsS --max-time 1 "http://localhost:${PORT}/api/status" 2> /dev/null | grep -q '"cyws-demo"'; then
  echo "Neat Change Studio is already running on port ${PORT}."
else
  echo "Starting Neat Change Studio…"
  if ! "$HERE/run.sh" --daemon --port "$PORT"; then
    echo
    echo "The demo did not start (see above)."
    pause
    exit 1
  fi
fi

echo "Opening ${URL}"
echo "The browser warns about the self-signed certificate once: Advanced, then continue."
if command -v xdg-open > /dev/null 2>&1; then
  # In its own session, so the browser it starts outlives this terminal window (no SIGHUP).
  if command -v setsid > /dev/null 2>&1; then
    setsid xdg-open "$URL" > /dev/null 2>&1 < /dev/null &
  else
    nohup xdg-open "$URL" > /dev/null 2>&1 < /dev/null &
    disown 2> /dev/null || true
  fi
  sleep 3
else
  echo "xdg-open not found; open ${URL} in a browser."
  pause
fi
