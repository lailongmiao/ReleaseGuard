# 凭据与最小权限 / Credentials and permissions

上游建议 token/PAT 使用完成工作所需的最小权限和最小 scopes；未提供替代 token 或 ssh-key 时，推荐 GITHUB_TOKEN permissions 为 contents: read。此建议不意味着任意跨仓库读取都获得授权。

persist-credentials 默认 true；设为 false 可退出凭据持久配置。v6 文档说明凭据放在 $RUNNER_TEMP 下的独立文件；不要把 v4 的 .git/config 存储说明泛化到所有版本。post-job 会清理凭据，但这不是任意异常场景凭据绝不泄漏的保证。

本应用的 GitHub 只读凭据留在服务端，不能替代 session、对象权限或 source ACL。被指定为审批人也不自动获得来源访问权。最后两项是 ReleaseGuard 项目规范，具体依据见 source security 文档。

来源：[actions/checkout 固定版本](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)；定位：Usage token; persist-credentials; Recommended permissions。本资料为审阅后的参考说明，不是执行授权或运行证据。

补充依据：[固定输入定义](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/action.yml)、[固定 README](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)。本项目边界见[发布规范](v1_release_policy.md)、[验证规范](v1_verification.md)、[来源权限规范](v1_source_security.md)。
