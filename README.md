# ReleaseGuard

ReleaseGuard 是一个基于 FastAPI、LangGraph 和 PostgreSQL 构建的代码变更审阅与发布预演后端。它把代码差异、CI 状态、仓库规则和检索证据组织成可追溯的分析报告，并通过确定性门禁、独立审批和 Dry Run 限制高风险操作。

> 本仓库仅包含后端。ReleaseGuard 当前定位为本地开发与工程研究项目；AI 输出仅用于辅助人工审阅，不能授予权限、自动批准变更或执行生产发布。

## 核心流程

```text
变更对象（PR / commit / branch compare）
  -> 固定来源与快照
  -> 收集 diff、CI 和仓库规则
  -> RAG 检索与带引用分析
  -> 确定性 Evidence Gate
  -> 独立人工审批
  -> Dry Run / SIMULATED 回执
  -> 审计与历史查询
```

模型负责解释和建议；权限、来源访问、证据新鲜度、审批有效期、幂等和执行资格由后端规则独立判断。即使模型给出积极结论，也不能绕过这些门禁。

## 主要能力

- 支持 PR、commit 和分支比较等多种变更对象，并绑定 revision、snapshot 和 CI observation。
- 使用 PostgreSQL 与 pgvector 保存对象、证据、报告、审批、命令和审计事实。
- 通过 RAG 组合代码片段、仓库规则与引用，保留来源、哈希和定位信息。
- 使用 LangGraph 管理有状态工作流、条件路由、暂停与恢复。
- 支持异主体审批、TTL、撤权、状态冲突检测和实时资格检查。
- 对未知命令结果采用 fail-closed 策略：允许查询原结果，不自动重发写操作。
- 提供结构化日志、Prometheus 指标、可选 Langfuse tracing 和速率限制。
- 保留 V1.0/V1.1 历史审批、原始 CI observation、命令结果和模拟回执的只读投影。

## 技术架构

| 层级 | 主要职责 | 技术 |
| --- | --- | --- |
| API | 身份、会话、审阅、来源、审批、回执 | FastAPI、Pydantic、slowapi |
| 工作流 | 状态流转、条件路由、暂停恢复 | LangGraph |
| 证据与检索 | 规则语料、代码片段、引用、向量检索 | PostgreSQL、pgvector、RAG |
| 模型适配 | OpenAI-compatible Chat / Embedding provider | LangChain、provider-neutral adapters |
| 持久化 | 业务事实、checkpoint、幂等命令、审计 | SQLModel、psycopg、Alembic |
| 可观测性 | 结构化日志、指标、可选追踪 | structlog、Prometheus、Grafana、Langfuse |

## 仓库结构

```text
app/                  FastAPI API、LangGraph、领域服务与持久化
alembic/              数据库迁移
resources/knowledge/  经过固定和校验的公开 RAG 规则语料
evals/                评测契约、数据集与离线评测工具
scripts/              本地运行、探针与维护脚本
tests/                单元、契约、持久化与安全回归测试
grafana/              Grafana provisioning
prometheus/           Prometheus 配置与告警规则
```

内部开发记录不属于本公开发行版，也不会由运行流程自动生成到仓库中。

## 环境要求

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- PostgreSQL 16+ 与 pgvector
- Docker / Docker Compose（可选，但推荐用于本地数据库和监控栈）

## 快速开始

1. 安装依赖：

   ```bash
   uv sync --all-groups
   ```

2. 创建开发配置：

   ```bash
   cp .env.example .env.development
   ```

   Windows PowerShell：

   ```powershell
   Copy-Item .env.example .env.development
   ```

3. 编辑 `.env.development`：

   - 将 `POSTGRES_HOST` 设为本机运行时的 `localhost`，或在 Compose 中使用 `db`。
   - 生成长度至少为 32 字符的随机 `JWT_SECRET_KEY`，不要使用示例占位值。
   - 按需配置 Chat、Embedding 和 Langfuse；不使用的外部能力应保持关闭。
   - ReleaseGuard V1/V1.1 能力是显式 opt-in，仅应在 `development` 或 `test` 环境启用。

4. 启动 PostgreSQL：

   ```bash
   docker compose --env-file .env.development up -d db
   ```

5. 执行迁移并启动 API：

   ```bash
   uv run alembic upgrade head
   uv run uvicorn app.main:app --reload --port 8000
   ```

启动后可访问：

- Swagger UI：`http://127.0.0.1:8000/docs`
- OpenAPI：`http://127.0.0.1:8000/api/v1/openapi.json`
- Liveness：`http://127.0.0.1:8000/live`
- Readiness：`http://127.0.0.1:8000/ready`
- Metrics：`http://127.0.0.1:8000/metrics`

也可以使用 Makefile：

```bash
make install
make dev
make docker-up ENV=development
```

## 配置原则

- 所有密钥只通过环境变量或未跟踪的 `.env.*` 文件提供。
- `V1_REAL_MODELS_ENABLED`、`V11_REAL_READS_ENABLED` 和 `V11_REAL_MODELS_ENABLED` 默认关闭。
- 不要把模型 API key、GitHub token、数据库密码或 JWT secret 写入代码、日志或 issue。
- 生产配置必须关闭 debug、限制 CORS，并使用独立的数据库、凭据与网络策略。

## 质量检查

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
uv run pre-commit run --all-files
```

测试覆盖确定性业务逻辑、状态机、权限隔离、审批、幂等、持久化、引用边界及 UNKNOWN 零重发等行为。真实模型输出具有非确定性；本项目不声称已经取得生产级 AI 质量、吞吐或延迟认证。

## 安全边界

- 当前执行能力限定为 Dry Run / `SIMULATED`，不提供真实批准、合并、部署或发布写入。
- 模型输出不能改变权限、审批、风险下限或证据新鲜度。
- 部分 diff 或缺失 CI 不能被解释为完整安全证明。
- UNKNOWN、超时或响应丢失不会自动触发同一写操作的重发。
- 使用真实仓库或模型前，应先检查来源授权、预算、日志脱敏和数据保留策略。

## 上游与许可

本项目基于 [`wassim249/fastapi-langgraph-agent-production-ready-template`](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template) 二次开发。上游原有的 FastAPI、LangGraph、认证、数据库、可观测性和部署基础被保留并扩展为 ReleaseGuard 的证据驱动审阅流程。

项目按照 [MIT License](LICENSE) 发布；原作者版权声明保持不变。二次开发归属见 [NOTICE.md](NOTICE.md)。
