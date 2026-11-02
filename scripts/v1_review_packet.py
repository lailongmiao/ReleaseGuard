"""Offline human review form from exact sanitized, schema-validated samples."""

import argparse
import json
from pathlib import Path

from rich.console import Console

from app.services.releaseguard.normalization import canonical_hash

FORM = """<!doctype html><html lang="zh"><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'">
<title>V1 人工复核</title><style>
body{font:16px/1.65 system-ui;max-width:1100px;margin:32px auto;padding:0 24px;background:#fafafa;color:#172034}
section{background:white;border:1px solid #ddd;padding:24px;margin:24px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f8;padding:12px}
select{font:inherit;margin:8px;padding:6px}button{font:inherit;padding:12px}details{margin:12px 0}header{position:sticky;top:0;background:#fafafa;padding:12px;border-bottom:1px solid #ddd}
</style><header><b>V1 人工解释/有依据声明复核</b> · <span id="progress"></span> <button id="export">导出 JSON</button></header>
<p>本页是本地验收材料，不是产品前端，不调用服务。对照引用原文逐条判定，再评分。5=准确完整；4=正确但轻微冗余；3=主要正确但遗漏；2=重要错误/错误拒答；1=不可用；0=失败。不能默认填满分。空值仍未验证。</p>
<p>规则问题不需要凭空要求实例姓名等细节；实例事实缺失应保留 UNKNOWN。门禁：均分≥4、每份≥3；声明依据逐例宏平均≥.95，自动契约另验。</p>
<main id="samples"></main><script>
const bundle=__BUNDLE__;const key='v1-review-'+bundle.template.output_hash;
let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const select=(name,options)=>'<select data-field="'+name+'"><option value="">待审阅</option>'+options.map(([v,l])=>'<option value="'+v+'">'+l+'</option>').join('')+'</select>';
document.getElementById('samples').innerHTML=bundle.samples.map((s,i)=>{
const o=s.output||{};return '<section data-index="'+i+'"><h2>'+esc(s.case_id)+' / '+s.attempt+'</h2><p>'+esc(s.query)+'</p><p>期望：'+esc(s.expected)+'；结构状态：'+esc(s.status)+'</p><pre>'+esc(o.explanation||s.reason||'输出失败')+'</pre><p>风险：'+esc(o.risk)+'；unknown：'+esc(JSON.stringify(o.unknowns||[]))+'</p>'+(o.claims||[]).map((c,j)=>'<div><b>声明 '+(j+1)+'</b><p>'+esc(c.text)+'</p>'+c.citation_ids.map(id=>'<details><summary>引用原文 '+esc(id)+'</summary><pre>'+esc(bundle.sources[id]||'缺少引用原文，不能判定支持')+'</pre></details>').join('')+select('claim-'+j,[[1,'有原文依据'],[0,'无依据/矛盾']])+'</div>').join('')+'<p>解释评分 '+select('rating',[[0,'0 失败'],[1,'1 不可用'],[2,'2 重要错误'],[3,'3 基本正确'],[4,'4 良好'],[5,'5 准确完整']])+'</p></section>'
}).join('');
const all=[...document.querySelectorAll('select')];all.forEach(e=>{const k=e.closest('section').dataset.index+':'+e.dataset.field;if(saved[k]!==undefined)e.value=saved[k];e.addEventListener('change',()=>{saved[k]=e.value;localStorage.setItem(key,JSON.stringify(saved));progress()})});
function progress(){document.getElementById('progress').textContent=all.filter(e=>e.value!=='').length+'/'+all.length+'项已复核'}progress();
document.getElementById('export').onclick=()=>{const t=JSON.parse(JSON.stringify(bundle.template));let complete=true;t.rows.forEach((r,i)=>{const section=document.querySelector('section[data-index="'+i+'"]');const rating=section.querySelector('[data-field="rating"]').value;const claims=[...section.querySelectorAll('select[data-field^="claim-"]')];const ready=rating!==''&&claims.every(e=>e.value!=='');r.human_rating=rating===''?null:Number(rating);r.grounded_claim_count=ready?claims.reduce((n,e)=>n+Number(e.value),0):null;r.reviewer_reference=ready?'v1-owner-human-review':null;if(!ready)complete=false});t.status=complete?'reviewed':'unverified';const blob=new Blob([JSON.stringify(t,null,2)+'\\n'],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='v1_human_review_completed.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
</script></html>"""


def build(provider: dict, dataset: dict) -> tuple[dict, str, str]:
    """Exact binding and no generated scores; escape data against script injection."""
    if provider["dataset_hash"] != canonical_hash(dataset):
        raise ValueError("review_dataset_mismatch")
    queries = {c["case_id"]: c["query"] for c in dataset["cases"]}
    rows = provider["rows"]
    template = {
        "schema": "v1-human-review.1",
        "dataset_hash": provider["dataset_hash"],
        "output_hash": canonical_hash(rows),
        "status": "unverified",
        "rows": [
            {
                "case_id": r["case_id"],
                "attempt": r["attempt"],
                "total_claims": len(r.get("output", {}).get("claims", [])),
                "human_rating": None,
                "grounded_claim_count": None,
                "reviewer_reference": None,
            }
            for r in rows
        ],
    }
    sources = {c["chunk_id"]: c["text"] for c in provider["review_sources"]}
    samples = [{**r, "query": queries[r["case_id"]]} for r in rows]
    bundle = (
        json.dumps({"template": template, "samples": samples, "sources": sources}, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    versions = sorted({r.get("telemetry", {}).get("prompt_version", "unknown") for r in rows})
    markdown = [
        "# V1 真实模型人工复核材料",
        "",
        "待人工评分；结构/拒答契约检查不能替代人工解释和声明依据。历史失败和旧复核包保留。版本："
        + ", ".join(versions),
        "",
        "[本地评分页](V1_MODEL_REVIEW_FORM.html)；导出的 JSON 放入 evals/reports/v1_human_review_completed.json。",
        "",
        "引用语料是三份获准 Markdown；以下没有原始 Provider envelope 或凭据。",
        "",
    ]
    for sample in samples:
        output = sample.get("output", {})
        markdown.extend(
            [
                "## " + sample["case_id"] + " / " + str(sample["attempt"]),
                "",
                "问题：" + sample["query"],
                "",
                "结构状态：" + sample["status"] + "；人工评分：待填写",
                "",
                output.get("explanation", "无有效输出"),
                "",
                "风险："
                + str(output.get("risk"))
                + "；unknown："
                + json.dumps(output.get("unknowns", []), ensure_ascii=False),
                "",
            ]
        )
        for claim in output.get("claims", []):
            markdown.extend(["声明：" + claim["text"], ""])
            for identifier in claim["citation_ids"]:
                markdown.extend(
                    ["引用 `" + identifier + "`：", "", "```text", sources.get(identifier, "引用原文缺失"), "```", ""]
                )
    return template, FORM.replace("__BUNDLE__", bundle), "\n".join(markdown) + "\n"


def main() -> int:
    """Generate public-safe local artifacts, preserving previous review outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-report", type=Path, required=True)
    args = parser.parse_args()
    provider = json.loads(args.provider_report.read_text(encoding="utf-8"))
    dataset = json.loads(Path("evals/datasets/v1_model_holdout.json").read_text(encoding="utf-8"))
    template, form, packet = build(provider, dataset)
    Path("evals/reports/v1_human_review_current_template.json").write_text(
        json.dumps(template, indent=2) + "\n", encoding="utf-8"
    )
    root = Path("evals/reports/v1_review")
    root.mkdir(parents=True, exist_ok=True)
    (root / "V1_MODEL_REVIEW_FORM.html").write_text(form, encoding="utf-8")
    (root / "V1_MODEL_REVIEW_PACKET_CURRENT.md").write_text(packet, encoding="utf-8")
    Console().print({"samples": len(provider["rows"]), "human_scores": "unverified_not_generated"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
