"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

import asyncio
from collections.abc import Awaitable, Hashable, Mapping
from typing import (
    AsyncGenerator,
    Any,
    Callable,
    Optional,
    cast,
)
from urllib.parse import quote_plus

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import (
    END,
    StateGraph,
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import (
    Command,
    CompiledStateGraph,
)
from langgraph.types import (
    RetryPolicy,
    StateSnapshot,
)
from psycopg import (
    AsyncConnection,
    sql,
)
from psycopg.rows import (
    DictRow,
    dict_row,
)
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings
from app.core.langgraph.tools import tools
from app.core.langgraph.releaseguard_approval import ApprovalRuntime, await_approval_decision, prepare_approval_wait
from app.core.langgraph.releaseguard_execution import ExecutionRuntime, execute_releaseguard
from app.core.langgraph.releaseguard_nodes import (
    collect_and_assess_evidence,
    commit_domain_command,
    load_domain_state,
    register_or_load_change,
    register_or_load_run,
)
from app.repositories.evidence import EvidenceRepository
from app.core.langgraph.releaseguard_resume import await_typed_releaseguard_resume
from app.core.langgraph.releaseguard_routing import (
    ReleaseGuardRoute,
    route_by_domain_status,
    route_loaded_state,
    route_workflow,
    set_routing_outcome,
)
from app.core.logging import logger
from app.core.metrics import llm_inference_duration_seconds
from app.core.observability import langfuse_callback_handler
from app.core.prompts import load_system_prompt
from app.schemas import (
    GraphState,
    Message,
)
from app.schemas.graph import ReleaseGuardRouteOutcome
from app.schemas.releaseguard import CommandAction
from app.repositories.releaseguard import ReleaseGuardRepository
from app.services.releaseguard.state_machine import DomainError
from app.services.evidence_sources import EvidenceAccessContext, EvidenceSourceAdapter
from app.services.llm import llm_service
from app.services.llm.contracts import ProviderError, ProviderErrorCode
from app.services.memory import memory_service
from app.utils import (
    dump_messages,
    extract_text_content,
    prepare_messages,
    process_llm_response,
)

PostgresConnPool = AsyncConnectionPool[AsyncConnection[DictRow]]


def _releaseguard_route_node(outcome: ReleaseGuardRouteOutcome) -> Callable[[GraphState], GraphState]:
    """Build an explicit capability-boundary node for one route result."""

    def route_node(state: GraphState) -> GraphState:
        return set_routing_outcome(state, outcome)

    return route_node


def build_agent_graph(
    chat_node: Any,
    tool_call_node: Any,
    checkpointer: Any,
    releaseguard_repository: ReleaseGuardRepository | None = None,
    evidence_repository: EvidenceRepository | None = None,
    evidence_adapters: Mapping[str, EvidenceSourceAdapter] | None = None,
    evidence_access: EvidenceAccessContext | Callable[[], EvidenceAccessContext] | None = None,
    *,
    approval_runtime: ApprovalRuntime | None = None,
    execution_runtime: ExecutionRuntime | None = None,
    analysis_runtime: Callable[[GraphState], Awaitable[GraphState]] | None = None,
    name: str | None = None,
) -> CompiledStateGraph:
    """Compile the single Chat/ReleaseGuard workflow with injectable offline dependencies."""
    if approval_runtime is not None and (
        approval_runtime.service.domain is not releaseguard_repository
        or approval_runtime.service.evidence is not evidence_repository
        or checkpointer is None
    ):
        raise DomainError("approval_runtime_dependency_mismatch")
    graph_builder = StateGraph(GraphState)
    if execution_runtime is not None and (
        execution_runtime.execution.preflight.service.domain is not releaseguard_repository
        or execution_runtime.execution.preflight.service.evidence is not evidence_repository
        or checkpointer is None
        or approval_runtime is not None
        and approval_runtime.service is not execution_runtime.execution.preflight.service
    ):
        raise DomainError("execution_runtime_dependency_mismatch")
    graph_builder.add_node("chat", chat_node, destinations=("tool_call", END))
    graph_builder.add_node(
        "tool_call",
        tool_call_node,
        destinations=("chat",),
        retry_policy=RetryPolicy(max_attempts=3),
    )

    async def register_change_node(state: GraphState, config: RunnableConfig) -> GraphState:
        if releaseguard_repository is None:
            raise DomainError("releaseguard_repository_unavailable")
        return await register_or_load_change(state, releaseguard_repository, config)

    async def register_run_node(state: GraphState, config: RunnableConfig) -> GraphState:
        if releaseguard_repository is None:
            raise DomainError("releaseguard_repository_unavailable")
        return await register_or_load_run(state, releaseguard_repository, config)

    async def load_state_node(state: GraphState, config: RunnableConfig) -> GraphState:
        if releaseguard_repository is None:
            raise DomainError("releaseguard_repository_unavailable")
        if (approval_runtime is not None or execution_runtime is not None) and state.releaseguard is not None:
            command = state.releaseguard.pending_command
            if command is not None and (
                command.action is CommandAction.APPEND_EXECUTION_FACTS
                or any(
                    getattr(command.inputs, field) not in (None, [])
                    for field in (
                        "execution_events",
                        "simulation_result",
                        "recovery_authorization",
                        "verification_result",
                        "verification_evidence",
                        "receipt",
                        "recovery_approval",
                    )
                )
            ):
                raise DomainError("execution_service_required")
            if command is not None and (
                command.inputs.approval is not None
                or command.target_status is not None
                and command.target_status.value
                not in {
                    "COLLECTING_EVIDENCE",
                    "WAITING_FOR_EVIDENCE",
                    "EVIDENCE_SUFFICIENT",
                    "RISK_ASSESSED",
                    "PLAN_READY",
                }
            ):
                raise DomainError("approval_service_required")
        return await load_domain_state(state, releaseguard_repository, config)

    async def resume_node(state: GraphState, config: RunnableConfig) -> GraphState:
        return await await_typed_releaseguard_resume(state, config)

    async def commit_command_node(state: GraphState, config: RunnableConfig) -> GraphState:
        if releaseguard_repository is None:
            raise DomainError("releaseguard_repository_unavailable")
        return await commit_domain_command(state, releaseguard_repository, config)

    async def assess_evidence_node(state: GraphState, config: RunnableConfig) -> GraphState:
        return await collect_and_assess_evidence(
            state,
            evidence_repository,
            evidence_adapters or {},
            evidence_access() if callable(evidence_access) else evidence_access,
            config,
        )

    async def approval_prepare_node(state: GraphState, config: RunnableConfig) -> GraphState:
        if approval_runtime is None or state.releaseguard is None or state.releaseguard.approval_request is None:
            return set_routing_outcome(state, "awaiting_approval")
        return await prepare_approval_wait(state, config, approval_runtime)

    async def approval_wait_node(state: GraphState, config: RunnableConfig) -> GraphState:
        if approval_runtime is None:
            raise DomainError("approval_runtime_unavailable")
        return await await_approval_decision(state, config, approval_runtime)

    def approval_prepare_route(state: GraphState) -> str:
        if approval_runtime is not None and state.releaseguard is not None and state.releaseguard.approval_request:
            return "releaseguard_approval_wait"
        return END

    def approval_result_route(state: GraphState) -> str:
        if state.releaseguard is not None and state.releaseguard.approval_outcome not in {None, "decided"}:
            return END
        return route_by_domain_status(state)

    async def execution_node(state: GraphState, config: RunnableConfig) -> GraphState:
        if execution_runtime is None:
            raise DomainError("execution_runtime_unavailable")
        return await execute_releaseguard(state, config, execution_runtime)

    def execution_route(state: GraphState) -> str:
        if execution_runtime is not None and state.releaseguard is not None and state.releaseguard.execution_intent:
            return "releaseguard_execution"
        return END

    graph_builder.add_node("releaseguard_register_change", register_change_node)
    graph_builder.add_node("releaseguard_register_run", register_run_node)
    graph_builder.add_node("releaseguard_load_state", load_state_node)
    graph_builder.add_node("releaseguard_validate_resume", resume_node)
    graph_builder.add_node("releaseguard_commit_command", commit_command_node)
    graph_builder.add_node("releaseguard_assess_evidence", assess_evidence_node)
    graph_builder.add_edge("releaseguard_register_change", "releaseguard_register_run")
    graph_builder.add_edge("releaseguard_register_run", "releaseguard_load_state")
    graph_builder.add_edge("releaseguard_validate_resume", "releaseguard_commit_command")
    graph_builder.add_edge("releaseguard_assess_evidence", "releaseguard_commit_command")

    route_nodes: dict[ReleaseGuardRoute, str] = {
        "awaiting_evidence": "releaseguard_awaiting_evidence",
        "awaiting_risk_assessment": "releaseguard_awaiting_risk_assessment",
        "awaiting_plan": "releaseguard_awaiting_plan",
        "awaiting_approval": "releaseguard_awaiting_approval",
        "unsupported_execution": "releaseguard_unsupported_execution",
        "blocked_unknown_risk": "releaseguard_blocked_unknown_risk",
        "terminal": "releaseguard_terminal",
    }
    for outcome, node_name in route_nodes.items():
        if outcome == "awaiting_approval":
            graph_builder.add_node(node_name, approval_prepare_node)
            graph_builder.add_conditional_edges(node_name, approval_prepare_route)
            continue
        graph_builder.add_node(node_name, cast(Any, _releaseguard_route_node(outcome)))
        if outcome in {"unsupported_execution", "terminal"}:
            graph_builder.add_conditional_edges(node_name, execution_route)
        else:
            graph_builder.add_edge(node_name, END)
    graph_builder.add_node("releaseguard_execution", execution_node)
    graph_builder.add_edge("releaseguard_execution", END)
    graph_builder.add_node("releaseguard_approval_wait", approval_wait_node)
    graph_builder.add_conditional_edges("releaseguard_approval_wait", approval_result_route, {**route_nodes, END: END})
    loaded_route_nodes: dict[Hashable, str] = {
        **route_nodes,
        "resume_pending_command": "releaseguard_validate_resume",
        "assess_evidence": "releaseguard_assess_evidence",
    }
    graph_builder.add_conditional_edges("releaseguard_load_state", route_loaded_state, cast(Any, loaded_route_nodes))
    graph_builder.add_conditional_edges("releaseguard_commit_command", route_by_domain_status, cast(Any, route_nodes))
    graph_builder.set_conditional_entry_point(
        lambda state: "analysis"
        if state.releaseguard and state.releaseguard.analysis_command_id
        else route_workflow(state),
        {"chat": "chat", "releaseguard": "releaseguard_register_change", "analysis": "releaseguard_analysis"},
    )

    async def analysis_node(state: GraphState) -> GraphState:
        if analysis_runtime is None:
            raise DomainError("analysis_runtime_unavailable")
        return await analysis_runtime(state)

    graph_builder.add_node("releaseguard_analysis", analysis_node)
    graph_builder.add_edge("releaseguard_analysis", END)
    graph_builder.set_finish_point("chat")
    return graph_builder.compile(checkpointer=checkpointer, name=name)


class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM.

    This class handles the creation and management of the LangGraph workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self, releaseguard_repository: ReleaseGuardRepository | None = None):
        """Initialize the LangGraph Agent with necessary components."""
        # Use the LLM service with tools bound
        self.llm_service = llm_service
        self.llm_service.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Optional[PostgresConnPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        self.releaseguard_repository = releaseguard_repository
        self.local_mvp_dependencies: dict[str, Any] = {}
        logger.info(
            "langgraph_agent_initialized",
            model=settings.DEFAULT_LLM_MODEL,
            environment=settings.ENVIRONMENT.value,
        )

    async def _get_connection_pool(self) -> PostgresConnPool:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool: The open connection pool.

        Raises:
            Exception: If the pool cannot be created, in every environment.
        """
        if self._connection_pool is None:
            try:
                # Configure pool size based on environment
                max_size = settings.POSTGRES_POOL_SIZE

                connection_url = (
                    "postgresql://"
                    f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
                    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
                )

                self._connection_pool = AsyncConnectionPool(
                    connection_url,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                        "row_factory": dict_row,
                    },
                )
                await self._connection_pool.open()
                logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
            except Exception as e:
                logger.exception(
                    "connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value
                )
                # Never degrade silently: the checkpointer is the only store for
                # conversation history and HITL resume state. Serving without it
                # loses data rather than surfacing an outage.
                raise e
        return self._connection_pool

    async def _chat(self, state: GraphState, config: RunnableConfig) -> Command:
        """Process the chat state and generate a response.

        Args:
            state (GraphState): The current state of the conversation.
            config (RunnableConfig): The runnable configuration for this invocation.

        Returns:
            Command: Command object with updated state and next node to execute.
        """
        # Get the current LLM instance for metrics
        current_llm = self.llm_service.get_llm()
        model_name = (
            current_llm.model_name
            if current_llm and hasattr(current_llm, "model_name")
            else settings.DEFAULT_LLM_MODEL
        )

        username = config.get("metadata", {}).get("username")
        thread_id = config.get("configurable", {}).get("thread_id")
        SYSTEM_PROMPT = load_system_prompt(username=username, long_term_memory=state.long_term_memory)

        # Prepare messages with system prompt
        messages = prepare_messages(state.messages, SYSTEM_PROMPT)

        try:
            # Use LLM service with automatic retries and circular fallback
            with llm_inference_duration_seconds.labels(model=model_name).time():
                response_message, usage = await self.llm_service.call_with_usage(dump_messages(messages))
                if not isinstance(response_message, BaseMessage):
                    raise ProviderError(
                        ProviderErrorCode.RESPONSE_INVALID,
                        "chat provider returned a non-message response",
                    )

            # Process response to handle structured content blocks
            response_message = process_llm_response(response_message)

            logger.info(
                "llm_response_generated",
                session_id=thread_id,
                model=model_name,
                environment=settings.ENVIRONMENT.value,
                usage_quality=usage.quality.value if usage.quality else "unknown",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cost_usd=usage.cost_usd,
                latency_ms=usage.latency_ms,
                provider=usage.provider,
                availability=usage.availability.value,
                error_category=usage.error_category,
            )

            # Determine next node based on whether there are tool calls
            if isinstance(response_message, AIMessage) and response_message.tool_calls:
                goto = "tool_call"
            else:
                goto = END

            return Command(update={"messages": [response_message]}, goto=goto)
        except Exception as e:
            logger.exception(
                "llm_call_failed_all_models",
                session_id=thread_id,
                error=str(e),
                environment=settings.ENVIRONMENT.value,
            )
            raise Exception(f"failed to get llm response after trying all models: {str(e)}")

    # Define our tool node
    async def _tool_call(self, state: GraphState) -> Command:
        """Process tool calls from the last message.

        Args:
            state: The current agent state containing messages and tool calls.

        Returns:
            Command: Command object with updated messages and routing back to chat.
        """
        tool_calls = state.messages[-1].tool_calls

        async def _execute_tool(tool_call: dict) -> ToolMessage:
            tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
            return ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )

        # Execute tool calls concurrently when multiple are requested
        if len(tool_calls) == 1:
            outputs = [await _execute_tool(tool_calls[0])]
        else:
            outputs = list(await asyncio.gather(*[_execute_tool(tc) for tc in tool_calls]))

        return Command(update={"messages": outputs}, goto="chat")

    async def create_graph(self) -> CompiledStateGraph:
        """Create and configure the LangGraph workflow.

        Returns:
            CompiledStateGraph: The configured LangGraph instance, always with a checkpointer.

        Raises:
            Exception: If the graph cannot be built, in every environment.
        """
        if self._graph is None:
            try:
                # Raises if the pool cannot be created — no checkpointer, no service.
                connection_pool = await self._get_connection_pool()
                checkpointer = AsyncPostgresSaver(connection_pool)
                await checkpointer.setup()

                self._graph = build_agent_graph(
                    self._chat,
                    self._tool_call,
                    checkpointer,
                    self.releaseguard_repository,
                    **self.local_mvp_dependencies,
                    name=f"{settings.PROJECT_NAME} Agent ({settings.ENVIRONMENT.value})",
                )

                logger.info(
                    "graph_created",
                    graph_name=f"{settings.PROJECT_NAME} Agent",
                    environment=settings.ENVIRONMENT.value,
                    has_checkpointer=checkpointer is not None,
                )
            except Exception as e:
                logger.exception("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                raise e

        return self._graph

    async def _get_graph(self) -> CompiledStateGraph:
        """Return the compiled graph, creating it on first access.

        Raises:
            Exception: Propagated from ``create_graph()`` when initialisation
                fails. Callers can rely on the return being non-``None``.
        """
        if self._graph is None:
            self._graph = await self.create_graph()
        return self._graph

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> list[Message]:
        """Get a response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.
            username (Optional[str]): The display name of the user.

        Returns:
            list[Message]: The response from the LLM.
        """
        graph = await self._get_graph()
        callbacks: list[BaseCallbackHandler] = [langfuse_callback_handler] if settings.LANGFUSE_TRACING_ENABLED else []
        config: RunnableConfig = {
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks,
            "metadata": {
                "user_id": user_id,
                "username": username,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }

        try:
            # Run state check and memory search concurrently to save 200-500ms
            state, memory_result = await asyncio.gather(
                graph.aget_state(config),
                memory_service.search_with_status(user_id, messages[-1].content),
            )
            logger.info(
                "memory_lookup_completed",
                session_id=session_id,
                memory_status=memory_result.status.value,
                memory_error_category=memory_result.error_category,
            )

            if state.next:
                logger.info("resuming_interrupted_graph", session_id=session_id, next_nodes=state.next)
                response = await graph.ainvoke(
                    Command(resume=messages[-1].content),
                    config=config,
                )
            else:
                relevant_memory = memory_result.content or "No relevant memory found."
                response = await graph.ainvoke(
                    input={"messages": dump_messages(messages), "long_term_memory": relevant_memory},
                    config=config,
                )

            # Check if the graph was interrupted during this invocation
            state = await graph.aget_state(config)
            if state.next:
                interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
                logger.info("graph_interrupted", session_id=session_id, interrupt_value=str(interrupt_value))
                return [Message(role="assistant", content=str(interrupt_value))]

            openai_msgs = cast(list[dict], convert_to_openai_messages(response["messages"]))
            asyncio.create_task(memory_service.add(user_id, openai_msgs, config.get("metadata")))
            return self.__process_messages(response["messages"])
        except GraphInterrupt:
            state = await graph.aget_state(config)
            interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
            logger.info("graph_interrupted", session_id=session_id, interrupt_value=str(interrupt_value))
            return [Message(role="assistant", content=str(interrupt_value))]
        except Exception as e:
            logger.exception("get_response_failed", error=str(e), session_id=session_id)
            raise

    async def get_stream_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.
            username (Optional[str]): The display name of the user.

        Yields:
            str: Tokens of the LLM response.
        """
        callbacks: list[BaseCallbackHandler] = [langfuse_callback_handler] if settings.LANGFUSE_TRACING_ENABLED else []
        config: RunnableConfig = {
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks,
            "metadata": {
                "user_id": user_id,
                "username": username,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }
        graph = await self._get_graph()

        try:
            # Run state check and memory search concurrently to save 200-500ms
            state, memory_result = await asyncio.gather(
                graph.aget_state(config),
                memory_service.search_with_status(user_id, messages[-1].content),
            )
            logger.info(
                "memory_lookup_completed_stream",
                session_id=session_id,
                memory_status=memory_result.status.value,
                memory_error_category=memory_result.error_category,
            )

            if state.next:
                logger.info("resuming_interrupted_graph_stream", session_id=session_id, next_nodes=state.next)
                graph_input = Command(resume=messages[-1].content)
            else:
                relevant_memory = memory_result.content or "No relevant memory found."
                graph_input = {"messages": dump_messages(messages), "long_term_memory": relevant_memory}

            async for token, _ in graph.astream(
                graph_input,
                config,
                stream_mode="messages",
            ):
                if not isinstance(token, (AIMessage, AIMessageChunk)):
                    continue

                text = extract_text_content(token.content)
                if text:
                    yield text

            # After streaming completes, check for interrupt or update memory
            state = await graph.aget_state(config)
            if state.next:
                interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
                logger.info("graph_interrupted_stream", session_id=session_id, interrupt_value=str(interrupt_value))
                yield str(interrupt_value)
            elif state.values and "messages" in state.values:
                openai_msgs = cast(list[dict], convert_to_openai_messages(state.values["messages"]))
                asyncio.create_task(memory_service.add(user_id, openai_msgs, config.get("metadata")))
        except GraphInterrupt:
            state = await graph.aget_state(config)
            interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
            logger.info("graph_interrupted_stream", session_id=session_id, interrupt_value=str(interrupt_value))
            yield str(interrupt_value)
        except Exception as stream_error:
            logger.exception("stream_processing_failed", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        graph = await self._get_graph()

        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        state: StateSnapshot = await graph.aget_state(config=config)
        return self.__process_messages(state.values["messages"]) if state.values else []

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        openai_style_messages = convert_to_openai_messages(messages)
        # keep just assistant and user messages
        return [
            Message(role=message["role"], content=str(message["content"]))
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        try:
            # Make sure the pool is initialized in the current event loop
            conn_pool = await self._get_connection_pool()
            if conn_pool is None:
                raise RuntimeError("connection pool unavailable; cannot clear chat history")

            # Batch all DELETEs in a single pipeline round-trip
            async with conn_pool.connection() as conn:
                async with conn.pipeline():
                    for table in settings.CHECKPOINT_TABLES:
                        await conn.execute(
                            sql.SQL("DELETE FROM {} WHERE thread_id = %s").format(sql.Identifier(table)),
                            (session_id,),
                        )
                logger.info(
                    "checkpoint_tables_cleared_for_session",
                    tables=settings.CHECKPOINT_TABLES,
                    session_id=session_id,
                )

        except Exception:
            logger.exception(
                "clear_chat_history_operation_failed",
                session_id=session_id,
            )
            raise
