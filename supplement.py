"""Build paper/supplement.tex from the integrity records (no sealed access, no outcome beyond what the
analysis already reports). Sections: deviations, freeze addendum, run inventory with GPU time, incidents,
audit report, per-layer displacement (census prediction and realized during training), literature search.
Run after the analysis; rerunnable (the output is a derived file). PROTOCOL.md ships as a separate file.
"""
import glob
import hashlib
import json
import os

import analysis_io as aio

ROOT = os.path.dirname(os.path.abspath(__file__))
STUDIES = os.path.join(ROOT, "studies")
OUT = os.path.join(ROOT, "paper", "supplement.tex")
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(s):
    return "".join(SPECIAL.get(c, c) for c in str(s))


def jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def section_records(lines):
    lines.append(r"\section{Protocol, deviations and freeze addendum}")
    lines.append(r"The binding protocol is \texttt{PROTOCOL.md} (SHA-256 \texttt{\seqsplit{" + sha(os.path.join(ROOT, "PROTOCOL.md")) +
                 r"}}), supplied as a separate file together with \texttt{FREEZE.json} (SHA-256 \texttt{\seqsplit{" +
                 sha(os.path.join(ROOT, "FREEZE.json")) + "}}).")
    lines.append(r"\subsection{Deviations}\begin{itemize}")
    for d in jsonl(os.path.join(ROOT, "DEVIATIONS.jsonl")):
        body = "; ".join(f"{k}: {v}" for k, v in d.items() if k != "utc")
        lines.append(r"\item \texttt{" + tex(d["utc"][:19]) + "} " + tex(body))
    lines.append(r"\end{itemize}")
    lines.append(r"\subsection{Freeze addendum (engineering and provenance records, in order)}\begin{itemize}")
    for d in jsonl(os.path.join(ROOT, "FREEZE_ADDENDUM.jsonl")):
        hashes = d.get("sha256", {})
        text = d.get("kind", "") + ". " + " ".join(str(d.get(k, "")) for k in ("issue", "note", "changes", "interpretations") if k in d)
        files = ", ".join(f"{k} {v[:12]}" for k, v in hashes.items()) if isinstance(hashes, dict) else ""
        extra = {k: v for k, v in d.items() if k.endswith("_sha256") and k != "sha256"}
        lines.append(r"\item \texttt{" + tex(d.get("utc", "")[:19]) + "} " + tex(text) +
                     (r" \emph{Files:} " + tex(files) if files else "") +
                     (" " + tex(json.dumps(extra)) if extra else ""))
    lines.append(r"\end{itemize}")


def section_runs(lines):
    lines.append(r"\section{Run inventory}")
    lines.append(r"Every declared run of every study, its status and its GPU time from the study ledger.")
    lines.append(r"\begin{longtable}{p{0.34\linewidth}rrrr}\toprule Study & declared & completed & pending & GPU-h\\\midrule\endhead")
    total = 0.0
    for reg in sorted(glob.glob(os.path.join(STUDIES, "**", "registry.json"), recursive=True)):
        folder = os.path.dirname(reg)
        runs = load(reg)["runs"]
        done = [r for r in runs if os.path.exists(os.path.join(folder, "runs", f"{r['run_id']}.json"))]
        gpu = sum(l["gpu_seconds"] for l in jsonl(os.path.join(folder, "ledger.jsonl"))) / 3600
        total += gpu
        rel = os.path.relpath(folder, STUDIES).replace(os.sep, "/")
        note = " (superseded)" if rel == "S2_base" else ""
        lines.append(tex(rel + note) + f" & {len(runs)} & {len(done)} & {len(runs) - len(done)} & {gpu:.2f}\\\\")
    lines.append(r"\midrule Total & & & & " + f"{total:.2f}" + r"\\\bottomrule\end{longtable}")
    excluded = sorted(glob.glob(os.path.join(ROOT, "excluded", "*", "README.json")))
    if excluded:
        lines.append(r"\paragraph{Excluded runs} Kept and reported but never analysed; their sealed files were never opened.\begin{itemize}")
        for path in excluded:
            note = load(path)
            lines.append(r"\item " + tex(os.path.basename(os.path.dirname(path)) + ": " + note["reason"].replace("\n", " ") +
                                         " Moved: " + ", ".join(note["moved"]) + "."))
        lines.append(r"\end{itemize}")


def section_incidents(lines):
    rows = []
    for path in sorted(glob.glob(os.path.join(STUDIES, "**", "incidents.jsonl"), recursive=True)):
        rows += [(os.path.relpath(os.path.dirname(path), STUDIES).replace(os.sep, "/"), r) for r in jsonl(path)]
    lines.append(r"\section{Incidents}\begin{itemize}")
    for study, r in rows:
        lines.append(r"\item " + tex(f"{study} {r.get('utc', '')[:19]} {r.get('run_id', '')}: {r.get('event', '')} "
                                     f"(files deleted unopened: {', '.join(r.get('files_deleted', [])) or 'none'})"))
    if not rows:
        lines.append(r"\item none")
    lines.append(r"\end{itemize}")


def section_audit(lines):
    lines.append(r"\section{Audits}")
    path = os.path.join(ROOT, "results", "audit.json")
    if not os.path.exists(path):
        lines.append("The audit report was not produced.")
        return
    a = load(path)
    lines.append(r"\begin{itemize}")
    for rid, r in a.get("retrain", {}).items():
        lines.append(r"\item " + tex(f"{rid}: deterministic replay max |diff| {r['deterministic_max_abs_diff']:.2e}, sealed hashes equal "
                                     f"{r['deterministic_sealed_hashes_equal']} ({'pass' if r['deterministic_pass'] else 'FAIL'}); "
                                     f"original vs replay |dev diff| {r['nondeterministic_abs_diff']:.4f} against tolerance "
                                     f"{r['tolerance_N_between_unit_sd']:.4f} ({'pass' if r['nondeterministic_pass'] else 'FAIL'})"))
    for kind, reps in a.get("evalreplay", {}).items():
        for rid, r in reps.items():
            lines.append(r"\item " + tex(f"evaluation replay ({kind}) {rid}: {json.dumps(r)}"))
    lines.append(r"\end{itemize}")


def realized_per_layer():
    """Row-weighted realized D, S2 and activation rate per layer, pooled over the units of each arm tag."""
    acc = {}
    for path in glob.glob(os.path.join(STUDIES, "S[12]", "**", "runs", "*.json"), recursive=True):
        if os.sep + "audit" + os.sep in path:
            continue
        rec = load(path)
        rid = rec["spec"]["run_id"]
        m = aio.S1_RE.match(rid) or aio.S2_RE.match(rid)
        if not m:
            continue
        regime, tag = rid[:2], m.group("tag")
        for entry in rec.get("budget_trace", []):
            for i, layer in enumerate(entry["layers"]):
                n = layer.get("rows", 0.0)
                if not n:
                    continue
                a = acc.setdefault((regime, tag), {}).setdefault(i, [0.0, 0.0, 0.0, 0.0])
                a[0] += n
                a[1] += n * layer["realized_tv"]
                a[2] += n * layer["realized_tv_sq"]
                a[3] += n * layer["active_rate"]
    return {k: {i: (v[1] / v[0], v[2] / v[0], v[3] / v[0]) for i, v in sorted(layers.items())} for k, layers in acc.items()}


def section_layers(lines):
    lines.append(r"\section{Per-layer displacement (PROTOCOL B10)}")
    census = os.path.join(ROOT, "results", "census_layers.json")
    if os.path.exists(census):
        c = load(census)
        for regime, s in c.items():
            lines.append(r"\paragraph{Census prediction, " + tex(regime) + r", $\ell=0.10$}")
            lines.append(r"\begin{tabular}{l" + "r" * len(s["layers"]) + r"}\toprule layer & " +
                         " & ".join(map(str, s["layers"])) + r"\\\midrule")
            for arm, vals in s["mean_displacement"].items():
                lines.append(tex(arm) + " $\\mathbb{E}d$ & " + " & ".join(f"{v:.3f}" for v in vals) + r"\\")
            lines.append(r"HB $\Pr(m>\tau)$ & " + " & ".join(f"{v:.2f}" for v in s["P_active_HB"]) + r"\\\bottomrule\end{tabular}")
    real = realized_per_layer()
    lines.append(r"\paragraph{Realized during training} Mean realized displacement $\bar D$ per layer, row-weighted over all "
                 r"epochs and units of each arm (training-mode rows, recorded on the GPU).")
    lines.append(r"\begin{longtable}{ll p{0.62\linewidth}}\toprule regime & arm & $\bar D$ by layer\\\midrule\endhead")
    for (regime, tag), layers in sorted(real.items()):
        lines.append(f"{regime} & {tex(tag)} & " + " ".join(f"{v[0]:.3f}" for v in layers.values()) + r"\\")
    lines.append(r"\bottomrule\end{longtable}")


def section_literature(lines):
    lines.append(r"\section{Literature search}")
    lines.append(r"The prior-art search and its adversarial refutation round are archived as \texttt{design/novelty/} "
                 r"(claim table, per-claim refuter reports and the synthesis). Coverage limits are stated in the main text.")


def main():
    lines = [r"\documentclass[11pt]{article}", r"\usepackage[margin=2.2cm]{geometry}",
             r"\usepackage{amsmath,amssymb,booktabs,longtable}", r"\usepackage[hidelinks]{hyperref}",
             r"\usepackage{seqsplit}", r"\title{Supplementary material: Capping realized attention displacement}",
             r"\date{}", r"\begin{document}", r"\maketitle", r"\sloppy\raggedright\small"]
    section_records(lines)
    section_runs(lines)
    section_incidents(lines)
    section_audit(lines)
    section_layers(lines)
    section_literature(lines)
    lines.append(r"\end{document}")
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
