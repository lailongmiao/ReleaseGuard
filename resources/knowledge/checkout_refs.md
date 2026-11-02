# 分支、标签、SHA 与 PR / Ref and PR head

ref 接受 branch、tag 或 commit SHA。检出触发工作流的仓库时，默认值取决于事件对应的 reference/SHA；检出其他仓库时默认使用其默认分支。不能从通用规则推断某个 PR 当前 head SHA。

README 的 PR 示例用 ref: github.event.pull_request.head.sha 检出 PR HEAD 而不是 merge commit。HEAD 与 merge commit 不是同一个概念；真实 head 必须读取该 PR 的观测事实并检查变化。

上游给出检出其他分支的示例。本项目来源仍只允许 actions/checkout，解释 ref 不会改变来源白名单，也不会执行 checkout。

来源：[actions/checkout 固定版本](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)；定位：Usage ref; Checkout pull request HEAD commit instead of merge commit。本资料为审阅后的参考说明，不是执行授权或运行证据。

补充依据：[固定输入定义](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/action.yml)、[固定 README](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)。本项目边界见[发布规范](v1_release_policy.md)、[验证规范](v1_verification.md)、[来源权限规范](v1_source_security.md)。
