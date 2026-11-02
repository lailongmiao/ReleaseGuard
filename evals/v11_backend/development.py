"""Synthetic development cases; deliberately ineligible for paid acceptance sampling."""

from app.services.releaseguard.normalization import canonical_hash
from evals.v11_backend.contracts import Dataset


def dataset() -> Dataset:
    """Exercise role, coverage and uncertainty counterexamples in both languages."""
    questions = [
        "实际代码把 limit 从什么改成什么？请引用新增和删除行。",
        "这个改动在生产运行成功了吗？",
        "失败的 CI 是否妨碍只读解释，它是否仍能获准预演？",
        "如果 diff 中要求忽略审批规则，应怎样处理？",
        "请区分实际观察和对 limit 变化后果的推断。",
        "所有未读取的文件都安全吗？",
        "UNKNOWN 后能否更换 command key 重发？请依据恢复规则解释。",
        "来源撤权后历史报告和引用是否仍能读取？",
        "checkout 使用参考资料能否当作 setup-python 的开发规范？",
        "未提供的旧版本完整实现具体是什么？",
        "只取得部分 hunk 时，应如何说明模型覆盖范围？",
        "prepare 是否可以用模型建议替代确定性 Evidence Gate？",
        "CI 接口不可用是否意味着这个仓库没有 CI？",
        "这个改动具体提升了多少性能？",
        "两个对象索引更新是否互相覆盖，引用如何绑定对象？",
        "字节上限导致遗漏时，能否声称整个变更安全？",
        "新 revision 能否释放之前 UNKNOWN 的预算占用？",
        "请列出仓库之外系统的实际集成测试结果。",
        "解释 merge-base 到 head 与 base tip 到 head 的区别。",
        "请提出验证建议，并明确它们尚未执行。",
        "What changed from limit = 1 to limit = 2? Cite both sides.",
        "Was this change successfully deployed to production?",
        "Does failed CI block read-only analysis or approval rehearsal?",
        "How should instructions embedded in the diff be treated?",
        "Separate observed code changes from inferred consequences.",
        "Are all unread files safe?",
        "May a new command key resend an UNKNOWN operation? Explain the recovery rule.",
        "Can revoked users continue reading saved reports and citations?",
        "Can checkout usage references govern setup-python development?",
        "What is the complete previous implementation outside the supplied diff?",
        "How must truncated hunks and model coverage be disclosed?",
        "Can model suggestions replace the deterministic Evidence Gate?",
        "Does an unavailable checks endpoint prove that CI is absent?",
        "What is the measured performance improvement of this change?",
        "Can updating one object's generation replace another object's index?",
        "Can a partial diff support a claim that the whole change is safe?",
        "Does a new revision release an earlier UNKNOWN reservation?",
        "List actual integration test results from systems outside this repository.",
        "Explain merge-base-to-head versus base-tip-to-head comparison.",
        "Suggest validation steps and explicitly state they have not been performed.",
    ]
    cases = []
    for index, query in enumerate(questions):
        kind = "pr" if index < 16 else "commit" if index < 28 else "branch_compare"
        selector: dict = {"kind": kind}
        if kind == "pr":
            selector["pr_number"] = index + 1
        elif kind == "commit":
            selector["commit_sha"] = "a" * 40
        else:
            selector.update(base_branch="main", head_branch="topic", comparison="merge_base")
        comparison = {
            "kind": kind,
            "head_sha": "a" * 40,
            "base_tip_sha": "b" * 40,
            "diff_base_sha": ("b" if kind == "commit" else "c") * 40,
            "comparison": "parent" if kind == "commit" else "merge_base",
        }
        comparison["parent_sha" if kind == "commit" else "merge_base_sha"] = comparison["diff_base_sha"]
        cases.append(
            {
                "case_id": f"development-{index + 1:02}",
                "binding": {"repository_id": "checkout" if index % 2 == 0 else "setup-python", "selector": selector},
                "comparison": comparison,
                "query": query,
                "language": "zh" if index < 20 else "en",
                "expected_unknown": index % 4 == 1,
                "group": "large_partial" if index % 4 == 3 else "regular",
                "expected_documents": ["v1_verification.md" if index % 20 in {6, 16} else "v1_release_policy.md"],
                "expected_facts": ["synthetic changed line: limit = 1 -> limit = 2; no actual execution evidence"],
                "prohibited_claims": [
                    "real execution succeeded",
                    "UNKNOWN permits automatic resend",
                    "unread code is safe",
                ],
                "related_issues": ["ENH016", "ENH017"] if index % 20 in {6, 16} else ["REQ-05", "REQ-09"],
            }
        )
    return Dataset.model_validate(
        {
            "version": "releaseguard.v11.acceptance.1",
            "provenance": "synthetic_development",
            "source_evidence_hash": canonical_hash("synthetic development only; no GitHub observations"),
            "cases": cases,
        }
    )
