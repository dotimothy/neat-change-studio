#!/usr/bin/env bash
# Compile every probe ONNX in build/probes/ (bf16) and report MLA ELF / A65 counts.
# Usage: compile/probes/run_probes.sh [probe ...]   (default: all)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"
probes=("$@")
[[ ${#probes[@]} -gt 0 ]] || probes=(coam32 coam8 scse16 dec256 stem)
for p in "${probes[@]}"; do
  compile/compile.sh --model "build/probes/$p.onnx" --precision bf16 --name "probe_$p" \
    --out "build/probes/$p" > /dev/null 2>&1 &
done
wait
printf '%-8s %-6s %-5s %s\n' probe elfs a65 plugins
for p in "${probes[@]}"; do
  s="build/probes/$p/compile_summary.json"
  if [[ -f "$s" ]]; then
    python3 -c "import json,sys; d=json.load(open('$s')); print(f\"{'$p':8} {len(d.get('mla_elfs',[])):<6} {len(d.get('a65',[])):<5} {d.get('plugins')}\")"
  else
    echo "$p: no summary (see build/probes/$p/compile.log)"; grep -m3 -i "error\|Cannot assign" "build/probes/$p/compile.log"
  fi
done
