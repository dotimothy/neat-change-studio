# Terminal styling for webapp/run.sh (sourced): the app logo and name, Neat-coloured status lines.
# Like Neat GenAI Studio's run.sh. Everything falls back to plain text when stdout is not a terminal,
# TERM is "dumb" or NO_COLOR is set, so logs stay clean. The logo is webapp/term_logo.txt
# (tools/make_term_logo.py draws it from the app icon), in truecolor; NEAT_NO_LOGO=1 leaves it out.
TERM_LOGO="$(dirname "${BASH_SOURCE[0]}")/term_logo.txt"
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  # the Neat logo palette: teal / green up, blue and lime sides, orange down, ink centre
  P_TEAL=$'\033[38;2;61;179;138m'; P_GREEN=$'\033[38;2;74;168;54m'; P_LIME=$'\033[38;2;154;190;30m'
  P_BLUE=$'\033[38;2;58;125;216m'; P_ORANGE=$'\033[38;2;223;108;30m'; P_INK=$'\033[38;2;60;66;74m'
  C_ACCENT="$P_TEAL"; C_MUTED=$'\033[38;2;140;150;160m'
  C_OK=$'\033[38;2;53;196;137m'; C_WARN=$'\033[38;2;224;173;74m'; C_ERR=$'\033[38;2;239;91;98m'
  TERM_FANCY=1
else
  C_RESET='' C_BOLD='' C_DIM='' P_TEAL='' P_GREEN='' P_LIME='' P_BLUE='' P_ORANGE='' P_INK=''
  C_ACCENT='' C_MUTED='' C_OK='' C_WARN='' C_ERR=''
  TERM_FANCY=0
fi

step()  { printf '\n%s\n' "${C_ACCENT}${C_BOLD}▸${C_RESET} ${C_BOLD}$*${C_RESET}"; }
info()  { printf '%s\n' "  ${C_MUTED}·${C_RESET} $*"; }
ok()    { printf '%s\n' "  ${C_OK}✔${C_RESET} $*"; }
warn()  { printf '%s\n' "  ${C_WARN}⚠${C_RESET} $*" >&2; }
errln() { printf '%s\n' "  ${C_ERR}✘${C_RESET} $*" >&2; }
kv()    { printf '  %s%-10s%s %s\n' "${C_MUTED}" "$1" "${C_RESET}" "$2"; }
link()  { printf '%s' "${C_ACCENT}${C_BOLD}$1${C_RESET}"; }

# A row of Neat sparkles in the logo colours.
sparkles() {
  local palette=("$P_TEAL" "$P_GREEN" "$P_LIME" "$P_BLUE" "$P_ORANGE") line="" i
  for ((i = 0; i < ${1:-12}; i++)); do line+="${palette[i % 5]}✦${C_RESET} "; done
  printf '%s' "$line"
}

# A titled break: sparkles, then a bold heading.
section() { printf '\n  %s\n  %s%s%s\n' "$(sparkles 12)" "${C_BOLD}" "$1" "${C_RESET}"; }

# The logo with the name, a tagline and a few facts beside it (pass the facts as arguments).
# Without colour: the name and tagline only.
banner() {
  local -a text=("" "" "${C_BOLD}NEAT${C_RESET} ${C_ACCENT}${C_BOLD}Change Studio${C_RESET}"
                 "${C_MUTED}Before/after change detection on the SiMa.ai Modalix MLA${C_RESET}" ""
                 "$(sparkles 10)" "" "$@")
  if ((TERM_FANCY)) && [[ -f "$TERM_LOGO" && -z "${NEAT_NO_LOGO:-}" ]]; then
    local -a logo; mapfile -t logo < "$TERM_LOGO"
    local n=$(( ${#logo[@]} > ${#text[@]} ? ${#logo[@]} : ${#text[@]} )) i pad
    pad="$(printf '%*s' 20 '')"
    printf '\n'
    for ((i = 0; i < n; i++)); do printf '  %s   %s\n' "${logo[i]:-$pad}" "${text[i]:-}"; done
  else
    printf '\n'
    local line; for line in "${text[@]:2}"; do [[ -n "$line" ]] && printf '  %s\n' "$line"; done
  fi
}
