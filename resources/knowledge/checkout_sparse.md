# 稀疏与部分检出 / Sparse checkout and filter

filter 用于按给定过滤器进行 partial clone，若设置会覆盖 sparse-checkout。sparse-checkout 接受按换行分隔的匹配模式；sparse-checkout-cone-mode 默认 true。

README 分别展示只获取 root files、root 加 .github 和 src、以及单文件的例子。单文件示例设 sparse-checkout: README.md 且 sparse-checkout-cone-mode: false。这些是检出范围配置，不能凭配置断言某个真实目录已完整存在。

诊断建议：缺文件时核对实际需要的路径和 filter/sparse 组合，验证目标文件是否存在；不得把部分检出冒充完整仓库。

来源：[actions/checkout 固定版本](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/action.yml)；定位：inputs.filter; inputs.sparse-checkout; inputs.sparse-checkout-cone-mode。本资料为审阅后的参考说明，不是执行授权或运行证据。

补充依据：[固定输入定义](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/action.yml)、[固定 README](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)。本项目边界见[发布规范](v1_release_policy.md)、[验证规范](v1_verification.md)、[来源权限规范](v1_source_security.md)。
