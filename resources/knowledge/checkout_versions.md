# 版本与 runner 要求 / Versions and runner requirements

这些说明来自固定 commit 的 README，不是某次运行记录，也不声明每个版本行为相同。

v5 升级到 Node 24，最低 Actions Runner 为 v2.327.1。v6 在 Docker container action 中运行需要认证的 git 命令，要求 Runner v2.329.0 或更新。固定 commit 的 action.yml 使用 node24。

v6 的凭据保存在 $RUNNER_TEMP 下的独立文件，不再直接存于 .git/config；README 的 v4 历史说明仍写 local git config，必须按版本理解。v7 默认拒绝 pull_request_target 或 workflow_run 触发时检出 fork PR 代码；allow-unsafe-pr-checkout 默认 false。上游提供显式危险选项，不代表 ReleaseGuard 授权开启它。

来源：[actions/checkout 固定版本](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)；定位：Checkout v7; Checkout v6; Checkout v5; Checkout v4。本资料为审阅后的参考说明，不是执行授权或运行证据。

补充依据：[固定输入定义](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/action.yml)、[固定 README](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)。本项目边界见[发布规范](v1_release_policy.md)、[验证规范](v1_verification.md)、[来源权限规范](v1_source_security.md)。
