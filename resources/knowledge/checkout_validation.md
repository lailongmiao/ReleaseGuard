# 配置变更验证与故障定位 / Validation checklist

以下为基于上游输入定义整理的验证建议，不是仓库已执行检查的观测：确认 ref/SHA、runner 版本、fetch-depth/fetch-tags、path、filter/sparse、submodules/lfs 及最小权限与变更目标一致。

历史不足先核对 fetch-depth，缺标签核对 fetch-tags，缺文件核对 path 和 sparse/filter；认证失败需要检查获准的凭据及范围，不能通过扩大权限或关闭安全规则来默认解决。

本项目 CI 接受条件另由 release policy 决定：先声明必需检查，再评价当前 head 上的结果；采集成功不等于 CI 成功。只有所有必需项存在、完整且符合成功策略，才能继续准备。验证建议本身不能证明某 PR 已通过。

来源：[actions/checkout 固定版本](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)；定位：Usage; Scenarios; Recommended permissions。本资料为审阅后的参考说明，不是执行授权或运行证据。

补充依据：[固定输入定义](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/action.yml)、[固定 README](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)。本项目边界见[发布规范](v1_release_policy.md)、[验证规范](v1_verification.md)、[来源权限规范](v1_source_security.md)。

## 本项目引用核验 / Local citation review

本节整理 ReleaseGuard 的引用质量约束，属于项目评审规则，不是上游 checkout README 的规定，也不声称原 [source security](v1_source_security.md) 已逐字表述以下语义限制。

引用 ID、版本、hash 与定位匹配负责可追溯性和内容身份，不能单独保证解释语义正确。有依据的回答仍需逐条核对声明是否由所引用片段支持、适用版本和上下文是否一致；复合声明要分别有据，正文不能另加未对应的事实。当问题要求具体实例事实，而来源缺少该实例的观测证据时，保留 UNKNOWN；解释已记录的规则不要求虚构实例观测。本次输出 schema 的枚举只能限定输出形状，不能代替所问规则或运行事实的来源证据。

Local review rule: matching citation identity, version, hash and location establishes traceability, not semantic truth. Check each claim against the cited excerpt and its version/context. Split compound claims as needed; explanation must not introduce unsupported facts. When a question asks for an instance fact without observation evidence, preserve UNKNOWN. Explaining a documented rule does not require inventing an instance observation. The current output schema enum constrains output shape; it cannot replace source evidence for the requested rule or runtime fact.
