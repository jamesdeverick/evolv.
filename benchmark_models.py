"""
Benchmark candidate Ollama models on the tool's real JSON jobs.

Usage (from the project root on the VM, same dir as app.py):
    python benchmark_models.py audit.md mistral:latest mistral-small3.2 gemma4:12b

Per model it runs, in a fresh subprocess (so OLLAMA_MODEL is picked up):
  1. Audit parsing (structured output) -> did JSON validate, how many
     requirements, did it capture the key named entities, how long it took
  2. Keyword batch analysis (8 keywords) -> did all 8 come back with valid fields

Pull models first:  ollama pull <tag>   (check exact tags at ollama.com/library)
"""
import json
import os
import subprocess
import sys
import time


def worker(audit_path):
    import types
    # Allow running outside Streamlit
    try:
        import streamlit  # noqa: F401
    except Exception:
        sys.modules["streamlit"] = types.SimpleNamespace(
            cache_data=lambda **k: (lambda f: f), warning=print, error=print,
            caption=print, spinner=lambda *a, **k: __import__("contextlib").nullcontext())
    from api.llm_client import LLMClient
    out = {"model": os.environ.get("OLLAMA_MODEL")}
    llm = LLMClient("ollama")
    if not llm.available:
        print(json.dumps({**out, "error": llm.error}))
        return

    # --- 1. Audit parse
    from analysis.audit_parser import AuditParser
    text = open(audit_path, encoding="utf-8").read()
    t0 = time.time()
    parsed = AuditParser().parse_audit(text)
    out["audit_seconds"] = round(time.time() - t0, 1)
    out["audit_ok"] = bool(parsed.get("has_audit"))
    out["audit_requirements"] = parsed.get("total_requirements", 0)
    blob = json.dumps(parsed).lower()
    probes = ["gartner", "forrester", "rancher", "key takeaways", "faq", "red hat"]
    out["audit_probe_hits"] = f"{sum(p in blob for p in probes)}/{len(probes)}"

    # --- 2. Keyword batch
    from analysis.keyword_analyzer import KeywordAnalyzer
    ka = KeywordAnalyzer.__new__(KeywordAnalyzer)
    ka.llm = llm
    kws = ["kubernetes management tools", "what is container orchestration", "rancher vs openshift",
           "best container management software", "kubernetes cost optimization", "gitops tutorial",
           "multi cluster management", "container security scanning"]
    t0 = time.time()
    res = ka._analyze_batch_chunk(kws, kws, "container management")
    out["kw_seconds"] = round(time.time() - t0, 1)
    out["kw_valid"] = sum(1 for r in res if r.get("rationale_for_own_page"))
    print("RESULT:" + json.dumps(out))


def main():
    if len(sys.argv) >= 2 and os.environ.get("BENCH_WORKER"):
        return worker(sys.argv[1])
    if len(sys.argv) < 3:
        print(__doc__)
        return
    audit, models = sys.argv[1], sys.argv[2:]
    rows = []
    for m in models:
        print(f"\n=== {m} ===", flush=True)
        env = {**os.environ, "OLLAMA_MODEL": m, "BENCH_WORKER": "1"}
        p = subprocess.run([sys.executable, __file__, audit], env=env,
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT:")), None)
        if line:
            rows.append(json.loads(line[7:]))
        else:
            rows.append({"model": m, "error": (p.stdout + p.stderr)[-300:]})
        print(rows[-1], flush=True)
    print("\n\nSUMMARY")
    cols = ["model", "audit_ok", "audit_requirements", "audit_probe_hits",
            "audit_seconds", "kw_valid", "kw_seconds", "error"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")) for c in cols))


if __name__ == "__main__":
    main()
