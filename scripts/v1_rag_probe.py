"""Frozen bilingual retrieval over actual scoped HTTP/PG and real embeddings."""

import argparse
import asyncio
import sys
from pathlib import Path
from time import perf_counter, sleep

from rich.console import Console

from app.schemas.v1_backend import Chunk
from app.services.releaseguard.normalization import canonical_hash
from app.services.v1_sources import corpus_manifest
from evals.v1_backend.enhancement import RetrievalDataset, summarize_retrieval
from evals.v1_backend.provenance import code_state
from scripts.mvp_product_probe import ProductProbe
from scripts.v1_provider_probe import reuse_sessions
from scripts.v1_http_analysis_probe import corpus, save


def main() -> int:
    """A report is one immutable run; unknown sends cannot be auto-resumed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:35080")
    parser.add_argument("--dataset", type=Path, default=Path("evals/datasets/v1_enhancement_rag.json"))
    parser.add_argument("--repeat", type=int, default=2, choices=(2,))
    parser.add_argument("--pr", type=int, default=2557)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("new_report_required_no_unknown_resend")
    dataset = RetrievalDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    if dataset.corpus_manifest_hash != canonical_hash(corpus_manifest()):
        parser.error("frozen_corpus_mismatch")
    probe = ProductProbe(args.url, args.compose_command, "real_rag")
    rows: list[dict] = []
    report = {
        "schema": "v1-enhancement-retrieval.1",
        "dataset_hash": canonical_hash(dataset),
        "code": code_state(),
        "rows": rows,
        "actual_cost": "unknown",
        "source": "actual_http_pg_real_query_embeddings",
    }
    try:
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
            runner.run(reuse_sessions(probe, args.compose_command))
            report["corpus"] = runner.run(corpus())
        for case in dataset.cases:
            for attempt in range(1, args.repeat + 1):
                started = perf_counter()
                body = probe.body(case.case_id + ":" + str(attempt))
                body["product"] = {"pr_number": args.pr}
                change = probe.request("POST", "/api/v1/releaseguard/changes", body)["state"]["change_id"]
                row = {
                    "case_id": case.case_id,
                    "attempt": attempt,
                    "status": "in_flight",
                    "resources": [],
                    "citations_valid": False,
                }
                rows.append(row)
                save(args.report, report)
                sleep(max(0, 3.2 - (perf_counter() - started)))
                result = probe.request(
                    "POST",
                    "/api/v1/evidence/search",
                    {
                        "change_id": change,
                        "command_id": probe.nonce + ":" + case.case_id + ":" + str(attempt),
                        "query": case.query,
                    },
                )
                chunks = [Chunk.model_validate(c) for c in result.get("items", [])]
                valid = bool(chunks) and all(
                    c.text == c.document.content[c.start_char : c.end_char]
                    and canonical_hash(c.text) == c.text_hash
                    and canonical_hash(c.document.content) == c.document.content_hash
                    for c in chunks
                )
                row.update(
                    status=result["status"],
                    reason=result.get("reason"),
                    resources=[c.document.resource for c in chunks],
                    citations_valid=valid,
                    hits=[
                        {
                            "resource": c.document.resource,
                            "document_id": c.document.document_id,
                            "chunk_id": c.chunk_id,
                            "version": c.document.version,
                            "hash": c.text_hash,
                            "locator": c.locator,
                            "generation": c.generation,
                        }
                        for c in chunks
                    ],
                )
                report["summary"] = summarize_retrieval(rows, dataset, args.repeat)
                save(args.report, report)
    except Exception as error:
        report["failure"] = type(error).__name__
    finally:
        probe.client.close()
        report["summary"] = summarize_retrieval(rows, dataset, args.repeat)
        save(args.report, report)
    Console().print(report["summary"])
    return 0 if report["summary"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
