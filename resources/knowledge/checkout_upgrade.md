# 升级、回退与模拟边界 / Upgrade and rollback

升级建议：先记录当前使用的版本或 SHA，阅读目标版本的运行要求和安全变化，再在获准隔离环境验证。v5 的 Node 24/runner 最低版本、v6 凭据位置与容器认证 runner 要求、v7 fork PR 默认拒绝策略分别适用，不能混用。

回退建议：如果验证失败，保留失败事实，评估恢复此前记录的版本或 SHA；回退是否可行还依赖目标环境与授权，不能仅凭这份建议宣称已回退或已恢复。

ReleaseGuard 的真实只读来源链路只做 Dry Run。Receipt 的 SIMULATED 表示保存模拟结果，不是发布、实际 checkout 或回滚成功。真实执行仍禁用；这些执行边界来自本项目 release policy 和 verification 规范。

来源：[actions/checkout 固定版本](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)；定位：Checkout v7; Checkout v6; Checkout v5。本资料为审阅后的参考说明，不是执行授权或运行证据。

补充依据：[固定输入定义](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/action.yml)、[固定 README](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)。本项目边界见[发布规范](v1_release_policy.md)、[验证规范](v1_verification.md)、[来源权限规范](v1_source_security.md)。

## 本项目模拟规范摘要 / Local simulation rules

本节转述 ReleaseGuard [验证规范](v1_verification.md)，不是 actions/checkout 的上游要求，也不替代服务端权限。模拟前核对输入、资源范围、环境、Evidence、权限、有效审批及所有绑定 hash；仅可建议三个候选步骤标识符 `validate_inputs`、`simulate`、`verify_predictions`。建议本身不授权调用。预测与观测必须分开；工具成功不等于验证通过。独立目标状态验证仅针对已有隔离合成沙盒；验证失败不得生成成功回执。结果未知或孤儿 claim 必须诊断并安全停止；不得自动释放 claim，不得重发效果不明的步骤，也不能用内存快照证明跨进程恢复。真实回滚与生产执行未启用。

Local project summary: check inputs, resource scope, environment, Evidence, permissions, valid approval and every binding hash before simulation. The only candidate identifiers are `validate_inputs`, `simulate`, and `verify_predictions`; suggestions grant no call authority. Keep predictions separate from observations. Tool success is insufficient; verification failure cannot produce a success Receipt. Unknown effects or orphan claims require diagnosis and a safe stop: never automatically release a claim or resend a step with uncertain effects. Memory snapshots do not prove cross-process recovery. Independent target verification applies only to the existing isolated synthetic sandbox; real rollback and production execution remain disabled. These are local project constraints, not new upstream checkout requirements.
