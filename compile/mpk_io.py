#!/usr/bin/env python3
"""Check the input order of a compiled CYWS MPK.

The SiMa compiler may order a multi-input MPK differently from the ONNX graph,
and pyneat matches tensors to inputs by position, so a reordering would
silently swap "before" and "after". This reads <name>_mpk.json from the MPK and
follows each MLA placeholder back to the graph input that feeds it (through the
cast / tessellate / ifm_pack plugins), the same idea as polima's
_mla_input_order resolver.

Usage: mpk_io.py build/bf16/cyws_bf16_mpk.tar.gz [--expect in0_before,in1_after]
Exit code 0 if the order matches, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile


def load_manifest(mpk_path: str) -> dict:
    with tarfile.open(mpk_path) as t:
        name = next(n for n in t.getnames() if n.endswith("_mpk.json"))
        return json.load(t.extractfile(name))


def resolve_order(mpk: dict) -> dict:
    plugins = {p["name"]: p for p in mpk["plugins"]}
    graph_inputs = [n["name"] for n in mpk["input_nodes"]]

    def source(name: str) -> str:
        """Walk upstream from a plugin output name to a graph input."""
        seen = set()
        while name not in graph_inputs:
            if name in seen or name not in plugins:
                return f"?{name}"
            seen.add(name)
            ins = plugins[name].get("input_nodes", [])
            if len(ins) != 1:
                return f"?{name}"
            name = ins[0]["name"].split("/")[0] if ins[0]["name"].split("/")[0] in plugins or \
                ins[0]["name"] in graph_inputs else ins[0]["name"]
        return name

    pack = next((p for p in mpk["plugins"] if p["name"].endswith("_ifm_pack_transform")), None)
    mla_order = []
    if pack:
        for slot in pack["input_nodes"]:
            m = re.search(r"placeholder_(\d+)", slot["name"])
            mla_order.append((int(m.group(1)) if m else -1, source(slot["name"])))
        mla_order.sort()
    return {"graph_inputs": graph_inputs, "mla_placeholders": mla_order,
            "mla_elfs": sum(1 for p in mpk["plugins"] if p["processor"] == "MLA"),
            "a65": sum(1 for p in mpk["plugins"] if p["processor"] == "A65")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mpk")
    ap.add_argument("--expect", default="in0_before,in1_after")
    args = ap.parse_args()
    info = resolve_order(load_manifest(args.mpk))
    want = args.expect.split(",")
    mla = [src for _, src in info["mla_placeholders"]]
    info["ok"] = info["graph_inputs"] == want and (not mla or mla == want)
    print(json.dumps(info, indent=1))
    return 0 if info["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
