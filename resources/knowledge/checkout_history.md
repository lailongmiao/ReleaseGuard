# 提交历史与标签 / Fetch depth and tags

fetch-depth 默认 1，即只获取一个提交。fetch-depth: 0 表示获取所有分支和标签的完整历史。fetch-tags 默认 false；设为 true 时即使 fetch-depth 大于 0 也获取标签，不等于获取完整历史。

README 的 Checkout HEAD^ 示例设置 fetch-depth: 2，然后检出前一个提交。这是示例，不证明任意工作流已有足够历史，也不授予本项目运行 git 命令的权限。

诊断建议：需要历史或标签的任务先明确需求，再核对相应输入；不能仅凭检出成功认定历史或标签完整。

来源：[actions/checkout 固定版本](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/action.yml)；定位：inputs.fetch-depth; inputs.fetch-tags。本资料为审阅后的参考说明，不是执行授权或运行证据。

补充依据：[固定输入定义](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/action.yml)、[固定 README](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)。本项目边界见[发布规范](v1_release_policy.md)、[验证规范](v1_verification.md)、[来源权限规范](v1_source_security.md)。
