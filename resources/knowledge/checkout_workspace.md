# 工作目录与清理 / Path and clean

path 是 $GITHUB_WORKSPACE 下放置仓库的相对路径。上游支持 side-by-side 与 nested 的目录示例，说明路径布局，不代表本项目支持任意多仓库来源。

clean 默认 true；其输入定义描述获取前执行 git clean -ffdx 和 git reset --hard HEAD。这里描述上游行为，不是让用户在已有工作区执行清理命令。涉及有价值的未提交内容时需先确认隔离和保护。

set-safe-directory 默认 true，会在 Git global config 加入仓库路径。这不是来源 ACL、凭据授权或业务安全审批，也不能证明任意内容可信。

来源：[actions/checkout 固定版本](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/action.yml)；定位：inputs.path; inputs.clean; inputs.set-safe-directory。本资料为审阅后的参考说明，不是执行授权或运行证据。

补充依据：[固定输入定义](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/action.yml)、[固定 README](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)。本项目边界见[发布规范](v1_release_policy.md)、[验证规范](v1_verification.md)、[来源权限规范](v1_source_security.md)。
