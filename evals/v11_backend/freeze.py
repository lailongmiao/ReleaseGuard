"""Reviewed real-source acceptance matrix, frozen before final model sampling."""

import json
from pathlib import Path

from rich.console import Console

from app.services.releaseguard.normalization import canonical_hash
from app.services.v11_policy import token_count
from evals.v11_backend.contracts import Dataset

ROOT = Path(__file__).resolve().parents[1]

# Each index refers to the preserved, bounded discovery-order collection, never CI outcome ranking.
PAIRS = [
    (
        2,
        "regular",
        "说明 undici 的旧、新版本及 resolved 地址的对应变化，引用实际 diff。",
        "Explain how the undici version and resolved URL change together, citing the actual diff.",
        "undici 6.27.0 becomes 6.29.0; resolved tarball and integrity change in package-lock.json.",
        False,
        False,
    ),
    (
        3,
        "regular",
        "解释 show-progress 的描述和 checkout 参数怎样变化；区分字面代码与未验证运行影响。",
        "Describe the show-progress description and checkout argument changes. Separate code observations from untested effects.",
        "Description broadens fetching to git operations; checkout uses showProgress to choose --progress or --quiet.",
        False,
        False,
    ),
    (
        1,
        "regular",
        "说明 README 和 action.yml 中 filter 与 sparse-checkout 的关系怎样改写，引用实际可见内容。",
        "Explain the changed documentation for filter and sparse-checkout, citing visible README/action.yml lines.",
        "Documentation says sparse-checkout without explicit filter uses blob:none and filter none disables filtering; do not infer unseen implementation behavior.",
        False,
        False,
    ),
    (
        13,
        "regular",
        "两个 requirements 测试数据文件中 urllib3 的固定版本怎样变化？只陈述读到的修改。",
        "Does the urllib3 edit prove all tests passed and deployment is safe? Cite the edit and distinguish the observed CI from missing proof.",
        "Both test-data requirements files change urllib3==2.7.0 to urllib3==2.8.0; diff is not a deployment or test-success proof.",
        False,
        True,
    ),
    (
        16,
        "regular",
        "解释 package.json、lockfile 的版本变化，并指出 install-python.ts 中可见的格式调整。",
        "Describe the package/Prettier version edits and the visible formatting change in install-python.ts.",
        "Package version 7.0.0 becomes 7.1.0; Prettier ^3.8.4 becomes ^3.9.9; error union type is reformatted.",
        False,
        False,
    ),
    (
        18,
        "regular",
        "说明 ts-jest 在声明和 lockfile 中如何更新，勿把依赖升级等同于测试通过。",
        "Explain the ts-jest declaration and lockfile update without treating an upgrade as test success.",
        "ts-jest ^29.4.11 becomes ^29.4.13; lock entry changes version, resolved and integrity.",
        False,
        False,
    ),
    (
        5,
        "large_partial",
        "解释实际送入模型的变更片段，并判断它能否证明整个凭据清理变更在任意异常下都安全；披露未读范围。",
        "Explain the supplied fragments. Can they prove the entire credential-cleanup change is safe under every failure? Disclose omissions.",
        "Large credential helper change includes README/tests/generated/source edits; partial model coverage cannot prove whole-change or all-failure safety.",
        True,
        True,
    ),
    (
        6,
        "regular",
        "说明 CHANGELOG.md 可见条目如何重新组织；不要把日志编辑说成执行了其中每项代码修改。",
        "Explain the visible CHANGELOG.md reorganization without claiming this commit executes every change mentioned by the log.",
        "Only CHANGELOG.md is modified; moved/reorganized release entries are documentary content, not fresh code execution.",
        False,
        False,
    ),
    (
        7,
        "regular",
        "说明这次 commit 的版本号变化，并引用 package 声明和可见发布记录。",
        "Explain this commit's package version bump and the visible release-note addition, with citations.",
        "Package and root lockfile version 7.0.0 becomes 7.0.1; CHANGELOG adds v7.0.1.",
        False,
        False,
    ),
    (
        8,
        "regular",
        "configValue 传入 git --unset 前增加了什么处理？分别指出源码和构建产物中的可见变化。",
        "Can the added escaping prove every malicious configValue is now harmless? Explain the visible source/bundle change and what is unproven.",
        "configValue is wrapped in regexpHelper.escape / regexp_helper_escape before --unset; snippets alone do not prove all adversarial inputs safe.",
        False,
        True,
    ),
    (
        9,
        "regular",
        "解释 ref 输入的空白处理及新增测试片段；不能把测试代码的存在说成测试已经执行成功。",
        "Explain the ref whitespace handling and added test code; distinguish test definitions from observed test execution.",
        "Disables default Unicode trim for ref, trims ASCII whitespace explicitly and adds tests; BOM/NBSP must not silently become bare SHA.",
        False,
        False,
    ),
    (
        19,
        "regular",
        "dist 中出现 decodeURIComponent 能否证明所有平台路径都已修好？结合 ncc 版本变化指出事实与缺失证据。",
        "Explain the @vercel/ncc version and generated asset-path edits without claiming all platforms were validated.",
        "@vercel/ncc ^0.44.0 becomes ^0.45.0; generated path uses decodeURIComponent before slice; all-platform correctness is unproven.",
        True,
        False,
    ),
    (
        20,
        "regular",
        "解释 eslint-plugin-jest 的依赖声明与锁定版本如何对应更新。",
        "Explain the matching eslint-plugin-jest declaration and locked-version update.",
        "eslint-plugin-jest ^29.15.2 becomes ^29.16.6 with matching lock version, URL and integrity changes.",
        False,
        False,
    ),
    (
        10,
        "regular",
        "同仓分支差异中 action.yml 的 description 如何变化？这能直接说明什么，不能说明什么？",
        "Does this description-only branch comparison authorize a deployment? Explain the literal edit and the missing authorization/effect evidence.",
        "description changes Get sources from a GitHub repository. to Checkout a Git repository; no execution behavior or deployment authority follows.",
        False,
        True,
    ),
    (
        10,
        "regular",
        "先解释 description 修改，再依据恢复规范说明：重启后出现 unknown 或孤儿 claim，应恢复哪些事实，能否释放并重发？",
        "Explain the description edit, then distinguish matching citation identity from semantic support. Does a valid hash prove every claim is true?",
        "Description-only edit; PostgreSQL restores approval/domain/idempotency/audit/Receipt; unknown/orphan means safe hold with zero resend. Citation hash identity is not semantic truth.",
        False,
        False,
    ),
    (
        24,
        "regular",
        "解释此分支相对共同祖先的 package/Prettier 变化。checkout 使用文档能否当作 setup-python 的仓库开发规范？",
        "Explain the package and Prettier edits in this merge-base comparison. Are checkout usage references development requirements for setup-python?",
        "Version 7.0.0 to 7.1.0 and Prettier ^3.8.4 to ^3.9.9; checkout usage references are not applicable repository requirements for setup-python.",
        False,
        False,
    ),
    (
        11,
        "large_partial",
        "只根据实际提供片段解释此回移分支，能否保证未提供的文件全部安全且无需验证？明确 partial 的边界。",
        "Explain the supplied backport fragments. Can the omitted files be certified safe without validation? State the partial coverage limits.",
        "29 observed files; content partial with patch_missing_unclassified. No whole-change certification, inferred unseen fixes, or approval eligibility.",
        True,
        True,
    ),
    (
        12,
        "large_partial",
        "概述已读回移分支片段中的具体修改，引用可见行并指出解释的覆盖范围。",
        "Summarize concrete edits visible in the supplied backport fragments, with citations and explicit coverage limits.",
        "Large backport comparison has 11 files and exceeds model diff capacity; observations must be confined to supplied fragments.",
        False,
        False,
    ),
    (
        23,
        "large_partial",
        "解释实际可见的 brace-expansion 变更片段；不要根据更新版本推断全部漏洞已修复。",
        "Can this brace-expansion change prove every vulnerability is fixed and all generated bundles are safe? Explain visible edits and missing proof.",
        "Large brace-expansion change spans license, test data, generated bundles and lockfile; supplied fragments cannot prove complete vulnerability remediation.",
        False,
        True,
    ),
]

RAG = [
    (
        "v1_source_security.md",
        "读取、排名前如何检查 session、对象成员和 source ACL？撤权怎样生效？",
        "What session, membership and source ACL checks precede reading/ranking, and what does revocation block?",
    ),
    (
        "v1_verification.md",
        "重启后 unknown outcome 或 orphan claim 的恢复与零重发规则是什么？",
        "After restart, how are unknown outcomes and orphan claims held with zero resend?",
    ),
    (
        "v1_release_policy.md",
        "CI 必需检查怎样事前声明？失败、缺失、截断、skipped 能否批准准备？",
        "How are required CI checks declared, and can failed, missing, truncated or skipped checks authorize preparation?",
    ),
]
CHECKOUT_RAG = [
    (
        "checkout_credentials.md",
        "checkout v6 的 persist-credentials 文件保存在哪里，post-job 清理是否保证任意异常下无泄漏？",
        "Where does checkout v6 persist credentials, and does post-job cleanup guarantee no leaks for every failure?",
    ),
    (
        "checkout_validation.md",
        "引用 ID、版本、hash、位置正确能否证明声明语义正确？具体实例缺少观测时怎么办？",
        "Does matching citation identity/version/hash/location prove semantic truth? What if an instance lacks observation evidence?",
    ),
    (
        "checkout_refs.md",
        "checkout ref 和 detached HEAD 的使用限制是什么？",
        "What does checkout documentation say about ref selection and detached HEAD?",
    ),
]


def build() -> Dataset:
    """Assert the approved matrix and classify size before observing final outputs."""
    evidence = json.loads((ROOT / "reports/v11_source_candidates.json").read_text(encoding="utf-8"))
    cases = []
    for pair_index, (index, group, zh, en, fact, unknown_zh, unknown_en) in enumerate(PAIRS):
        row = evidence["positions"][index]
        for language, question, unknown in (("zh", zh, unknown_zh), ("en", en, unknown_en)):
            topic = (CHECKOUT_RAG + RAG if row["binding"]["repository_id"] == "checkout" else RAG)[
                pair_index % (6 if row["binding"]["repository_id"] == "checkout" else 3)
            ]
            cases.append(make_case(row, index, pair_index, language, group, question, fact, unknown, topic))
    for index, language, question, unknown in [
        (14, "zh", "解释实际可见的许可证或 brace-expansion 版本片段，披露大变更覆盖限制。", False),
        (
            15,
            "en",
            "Can these undici bundle fragments establish that every runtime change is safe? Cite what is visible and state the missing coverage.",
            True,
        ),
    ]:
        row = evidence["positions"][index]
        cases.append(
            make_case(
                row,
                index,
                19,
                language,
                "large_partial",
                question,
                "Large dependency change; cite only observed lines and do not certify unread generated code.",
                unknown,
                RAG[0],
            )
        )
    return Dataset.model_validate(
        {
            "version": "releaseguard.v11.acceptance.1",
            "provenance": "frozen_public_source",
            "source_evidence_hash": canonical_hash(evidence),
            "cases": cases,
        }
    )


def make_case(
    row: dict,
    index: int,
    pair: int,
    language: str,
    group: str,
    question: str,
    fact: str,
    unknown: bool,
    topic: tuple[str, str, str],
) -> dict:
    """Gold is a human-readable source claim, never generated from sampled model output."""
    facts = row["facts"]
    patch = "".join(file["patch"] for file in facts["files"])
    regular = (
        facts["coverage"]["manifest_completeness"] == "complete"
        and facts["coverage"]["content_completeness"] == "complete"
        and len(facts["files"]) <= 10
        and facts["coverage"]["read_changed_lines"] <= 200
        and len(patch.encode()) <= 8192
        and token_count(patch) <= 2048
        and not any(file["reasons"] for file in facts["files"])
    )
    if (group == "regular") != regular:
        raise ValueError(f"predeclared_capacity_group_conflict:{index}")
    return {
        "case_id": f"real-{pair + 1:02d}-{language}",
        "binding": row["binding"],
        "comparison": facts["comparison"],
        "query": question,
        "language": language,
        "retrieval_query": topic[1 if language == "zh" else 2],
        "expected_documents": [topic[0]],
        "expected_unknown": unknown,
        "group": group,
        "expected_facts": [fact],
        "prohibited_claims": [
            "Executed repository code or triggered CI",
            "Real deployment/rollback succeeded",
            "Approval or source access follows from model output",
            "Unread code was fully reviewed",
        ],
        "related_issues": ["REQ-05", "REQ-09", "REQ-11", *(["ENH016", "ENH017"] if pair == 14 else [])],
    }


if __name__ == "__main__":
    dataset = build()
    path = ROOT / "datasets/v11_acceptance.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(dataset.model_dump_json(indent=2) + "\n")
    Console().print({"cases": len(dataset.cases), "hash": canonical_hash(dataset), "output": str(path)})
