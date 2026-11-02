# 子模块与 LFS / Submodules and Git LFS

lfs 默认 false；true 表示下载 Git-LFS 文件。submodules 默认 false，可设 true 获取子模块，或 recursive 递归获取子模块。

处理子模块时，未提供 ssh-key 的 git@github.com: 子模块 URL 会转换为 HTTPS。转换 URL 不等于取得私有依赖的访问权限；依赖是否可读取仍需独立授权和实际验证。

诊断建议：先核对是否需要 LFS 或子模块、所需版本和授权，再检查实际文件。一次 checkout 成功不能证明所有依赖完整。ReleaseGuard 只解释这些配置，不自动读取这些额外来源。

来源：[actions/checkout 固定版本](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/action.yml)；定位：inputs.submodules; inputs.lfs; inputs.ssh-key。本资料为审阅后的参考说明，不是执行授权或运行证据。

补充依据：[固定输入定义](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/action.yml)、[固定 README](https://github.com/actions/checkout/blob/f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a/README.md)。本项目边界见[发布规范](v1_release_policy.md)、[验证规范](v1_verification.md)、[来源权限规范](v1_source_security.md)。
