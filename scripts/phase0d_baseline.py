"""Run a redacted Phase 0D baseline against a running local application."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, LiteralString
from urllib.parse import quote_plus

import httpx
import psycopg
from dotenv import dotenv_values
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the baseline runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--env-file", default=".env.development")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def result(status: str, **details: Any) -> dict[str, Any]:
    """Build one redacted check result."""
    return {"status": status, **details}


def request_summary(response: httpx.Response) -> dict[str, Any]:
    """Return response metadata without response content."""
    return {"status_code": response.status_code, "content_type": response.headers.get("content-type", "")}


def get_json(response: httpx.Response) -> dict[str, Any]:
    """Decode an object response without logging its contents."""
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("expected object response")
    return payload


def checkpoint_counts(env_file: str, thread_id: str) -> dict[str, int]:
    """Count checkpoint rows for one exact graph thread."""
    counts: dict[str, int] = {}
    with psycopg.connect(**database_connection_kwargs(env_file)) as connection:
        for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
            row = connection.execute(f"SELECT count(*) FROM {table} WHERE thread_id = %s", (thread_id,)).fetchone()
            counts[table] = int(row[0]) if row else 0
    return counts


def database_connection_kwargs(env_file: str) -> dict[str, Any]:
    """Read local database coordinates without exposing credentials."""
    values = dotenv_values(env_file)
    return {
        "host": "localhost",
        "port": int(values.get("POSTGRES_PORT") or 5432),
        "dbname": values.get("POSTGRES_DB"),
        "user": values.get("POSTGRES_USER"),
        "password": values.get("POSTGRES_PASSWORD"),
    }


def memory_count(env_file: str, email: str) -> int:
    """Count this run's Mem0 rows without reading stored content."""
    with psycopg.connect(**database_connection_kwargs(env_file)) as connection:
        user_row = connection.execute('SELECT id FROM "user" WHERE email = %s', (email,)).fetchone()
        if not user_row:
            return 0
        row = connection.execute(
            "SELECT count(*) FROM longterm_memory WHERE payload->>'user_id' = %s",
            (str(user_row[0]),),
        ).fetchone()
        return int(row[0]) if row else 0


async def _graph_snapshot(env_file: str, thread_id: str) -> dict[str, Any]:
    """Inspect serialized graph state without exposing message content."""
    coordinates = database_connection_kwargs(env_file)
    dsn = (
        f"postgresql://{quote_plus(str(coordinates['user']))}:{quote_plus(str(coordinates['password']))}"
        f"@localhost:{coordinates['port']}/{quote_plus(str(coordinates['dbname']))}"
    )
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        checkpoint = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
    if checkpoint is None:
        return {"checkpoint_present": False}

    messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
    ai_tool_calls = [
        str(call.get("name")) for message in messages if isinstance(message, AIMessage) for call in message.tool_calls
    ]
    tool_messages = [message.name for message in messages if isinstance(message, ToolMessage)]
    pending_interrupts = sum(1 for _, channel, _ in (checkpoint.pending_writes or []) if channel == "__interrupt__")
    memory_context = checkpoint.checkpoint.get("channel_values", {}).get("long_term_memory", "")
    last_message = messages[-1] if messages else None
    return {
        "checkpoint_present": True,
        "message_count": len(messages),
        "ai_tool_calls": ai_tool_calls,
        "tool_messages": [name for name in tool_messages if name],
        "pending_interrupts": pending_interrupts,
        "last_ai_message": isinstance(last_message, AIMessage),
        "memory_context_present": bool(memory_context and memory_context != "No relevant memory found."),
    }


def graph_snapshot(env_file: str, thread_id: str) -> dict[str, Any]:
    """Run checkpoint inspection from the synchronous HTTP baseline."""
    return asyncio.run(_graph_snapshot(env_file, thread_id))


def count_rows(connection: psycopg.Connection[Any], query: LiteralString, params: tuple[Any, ...]) -> int:
    """Read a count query with a clear failure if no row is returned."""
    row = connection.execute(query, params).fetchone()
    if row is None:
        raise ValueError("count query returned no row")
    return int(row[0])


def cleanup_database(env_file: str, email: str, session_id: str | None) -> dict[str, Any]:
    """Remove only this run's database records and return redacted counts."""
    values = dotenv_values(env_file)
    connection_kwargs = {
        "host": "localhost",
        "port": int(values.get("POSTGRES_PORT") or 5432),
        "dbname": values.get("POSTGRES_DB"),
        "user": values.get("POSTGRES_USER"),
        "password": values.get("POSTGRES_PASSWORD"),
    }
    with psycopg.connect(**connection_kwargs) as connection:
        user_row = connection.execute('SELECT id FROM "user" WHERE email = %s', (email,)).fetchone()
        user_id = int(user_row[0]) if user_row else None
        if session_id:
            recorded_session = connection.execute(
                "SELECT user_id FROM session WHERE id = %s", (session_id,)
            ).fetchone()
            if recorded_session and int(recorded_session[0]) != user_id:
                raise ValueError("recorded session belongs to a different user")
        session_ids = (
            [row[0] for row in connection.execute("SELECT id FROM session WHERE user_id = %s", (user_id,)).fetchall()]
            if user_id is not None
            else []
        )
        before = {
            "user": 1 if user_id is not None else 0,
            "sessions": len(session_ids),
            "checkpoints": 0,
            "checkpoint_writes": 0,
            "checkpoint_blobs": 0,
            "longterm_memory": 0,
        }
        memory_query = "SELECT count(*) FROM longterm_memory WHERE payload->>'user_id' = %s"
        if user_id is not None:
            before["longterm_memory"] = count_rows(connection, memory_query, (str(user_id),))
        for current_session_id in session_ids:
            for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
                before[table] += count_rows(
                    connection, f"SELECT count(*) FROM {table} WHERE thread_id = %s", (current_session_id,)
                )

        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            for current_session_id in session_ids:
                connection.execute(f"DELETE FROM {table} WHERE thread_id = %s", (current_session_id,))
        if user_id is not None:
            connection.execute("DELETE FROM longterm_memory WHERE payload->>'user_id' = %s", (str(user_id),))
            connection.execute("DELETE FROM session WHERE user_id = %s", (user_id,))
            connection.execute('DELETE FROM "user" WHERE id = %s AND email = %s', (user_id, email))
        connection.commit()

        after = {
            "user": count_rows(connection, 'SELECT count(*) FROM "user" WHERE email = %s', (email,)),
            "sessions": 0,
            "checkpoints": 0,
            "checkpoint_writes": 0,
            "checkpoint_blobs": 0,
            "longterm_memory": 0,
        }
        if user_id is not None:
            after["sessions"] = count_rows(connection, "SELECT count(*) FROM session WHERE user_id = %s", (user_id,))
            after["longterm_memory"] = count_rows(connection, memory_query, (str(user_id),))
        for current_session_id in session_ids:
            for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
                after[table] += count_rows(
                    connection, f"SELECT count(*) FROM {table} WHERE thread_id = %s", (current_session_id,)
                )
    return {"before": before, "after": after, "cleared": all(value == 0 for value in after.values())}


def main() -> int:
    """Run the redacted Phase 0D API and persistence baseline."""
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    run_id = uuid.uuid4().hex[:12]
    email = f"phase0d_{run_id}@example.com"
    password = f"Phase0d!{uuid.uuid4().hex[:12]}"
    username = f"phase0d-{run_id}"
    user_token: str | None = None
    session_token: str | None = None
    session_id: str | None = None
    checks: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }

    def record(name: str, status: str, **details: Any) -> None:
        checks["checks"][name] = result(status, **details)

    with httpx.Client(timeout=args.timeout, follow_redirects=True) as client:
        try:
            for path in ("/", "/health", "/api/v1/health", "/metrics"):
                response = client.get(f"{base_url}{path}")
                record(
                    f"endpoint:{path}",
                    "passed" if response.status_code == 200 else "failed",
                    **request_summary(response),
                )

            response = client.post(
                f"{base_url}/api/v1/auth/register",
                json={"email": email, "password": password, "username": username},
            )
            registration = get_json(response) if response.is_success else {}
            record(
                "register",
                "passed" if response.status_code == 200 and "token" in registration else "failed",
                **request_summary(response),
            )
            user_token = registration.get("token", {}).get("access_token")

            response = client.post(
                f"{base_url}/api/v1/auth/login",
                data={"email": email, "password": password, "grant_type": "password"},
            )
            login = get_json(response) if response.is_success else {}
            record(
                "login",
                "passed" if response.status_code == 200 and login.get("access_token") else "failed",
                **request_summary(response),
            )
            user_token = login.get("access_token") or user_token

            if not user_token:
                raise RuntimeError("authentication did not return a user token")

            response = client.post(
                f"{base_url}/api/v1/auth/session",
                headers={"Authorization": f"Bearer {user_token}"},
            )
            session = get_json(response) if response.is_success else {}
            session_id = session.get("session_id")
            session_token = session.get("token", {}).get("access_token")
            record(
                "create_session",
                "passed" if response.status_code == 200 and session_id and session_token else "failed",
                **request_summary(response),
            )

            if not session_id or not session_token:
                raise RuntimeError("session creation did not return a session token")

            user_headers = {"Authorization": f"Bearer {user_token}"}
            session_headers = {"Authorization": f"Bearer {session_token}"}
            response = client.get(f"{base_url}/api/v1/auth/sessions", headers=user_headers)
            sessions = response.json() if response.is_success else []
            record(
                "list_sessions",
                "passed"
                if response.status_code == 200 and any(item.get("session_id") == session_id for item in sessions)
                else "failed",
                **request_summary(response),
                count=len(sessions) if isinstance(sessions, list) else None,
            )

            chat_payload = {"messages": [{"role": "user", "content": "Reply with a short greeting."}]}
            response = client.post(f"{base_url}/api/v1/chatbot/chat", headers=session_headers, json=chat_payload)
            chat = get_json(response) if response.is_success else {}
            messages = chat.get("messages", [])
            record(
                "chat_non_stream",
                "passed" if response.status_code == 200 and messages else "failed",
                **request_summary(response),
                message_count=len(messages) if isinstance(messages, list) else None,
            )

            counts_after_chat = checkpoint_counts(args.env_file, session_id)
            chat_snapshot = graph_snapshot(args.env_file, session_id)
            record(
                "checkpoint_after_chat",
                "passed"
                if counts_after_chat.get("checkpoints", 0) > 0 and chat_snapshot["checkpoint_present"]
                else "failed",
                counts=counts_after_chat,
                snapshot=chat_snapshot,
            )

            response = client.get(f"{base_url}/api/v1/chatbot/messages", headers=session_headers)
            history = get_json(response) if response.is_success else {}
            history_messages = history.get("messages", [])
            record(
                "history_read",
                "passed" if response.status_code == 200 and len(history_messages) >= 2 else "failed",
                **request_summary(response),
                message_count=len(history_messages) if isinstance(history_messages, list) else None,
            )

            stream_payload = {"messages": [{"role": "user", "content": "Reply with one short sentence."}]}
            stream_events: list[dict[str, Any]] = []
            with client.stream(
                "POST", f"{base_url}/api/v1/chatbot/chat/stream", headers=session_headers, json=stream_payload
            ) as stream_response:
                for line in stream_response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    stream_events.append(event)
            done_events = [event for event in stream_events if event.get("done") is True]
            error_event = any(event.get("content") for event in done_events)
            record(
                "chat_stream",
                "passed" if stream_response.status_code == 200 and done_events and not error_event else "failed",
                status_code=stream_response.status_code,
                event_count=len(stream_events),
                done_count=len(done_events),
                error_event=error_event,
            )

            preference = "For this synthetic test, my preferred editor is Vim. Remember that preference."
            response = client.post(
                f"{base_url}/api/v1/chatbot/chat",
                headers=session_headers,
                json={"messages": [{"role": "user", "content": preference}]},
            )
            preference_accepted = response.status_code == 200
            deadline = time.monotonic() + 20
            stored_memories = memory_count(args.env_file, email)
            while preference_accepted and stored_memories == 0 and time.monotonic() < deadline:
                time.sleep(1)
                stored_memories = memory_count(args.env_file, email)
            record(
                "memory_add",
                "passed" if preference_accepted and stored_memories > 0 else "degraded",
                **request_summary(response),
                stored_rows=stored_memories,
            )
            response = client.post(
                f"{base_url}/api/v1/chatbot/chat",
                headers=session_headers,
                json={"messages": [{"role": "user", "content": "Which editor did I say I prefer?"}]},
            )
            memory_snapshot = graph_snapshot(args.env_file, session_id)
            record(
                "memory_search_in_graph",
                "passed" if response.status_code == 200 and memory_snapshot["memory_context_present"] else "degraded",
                **request_summary(response),
                stored_rows=memory_count(args.env_file, email),
                memory_context_present=memory_snapshot["memory_context_present"],
            )

            response = client.post(
                f"{base_url}/api/v1/chatbot/chat",
                headers=session_headers,
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Use the duckduckgo_results_json tool now to search for the official Python website. "
                                "Do not answer from memory. Return only a result domain after the search."
                            ),
                        }
                    ]
                },
            )
            search_snapshot = graph_snapshot(args.env_file, session_id)
            search_completed = (
                response.status_code == 200
                and "duckduckgo_results_json" in search_snapshot["ai_tool_calls"]
                and "duckduckgo_results_json" in search_snapshot["tool_messages"]
                and search_snapshot["last_ai_message"]
            )
            record(
                "duckduckgo_graph_loop",
                "passed" if search_completed else "degraded",
                **request_summary(response),
                snapshot=search_snapshot,
            )

            response = client.post(
                f"{base_url}/api/v1/chatbot/chat",
                headers=session_headers,
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "I am considering deleting a local file. Do not delete anything. "
                                "Before any further step, call ask_human to ask for my approval and wait for my reply."
                            ),
                        }
                    ]
                },
            )
            interrupt_snapshot = graph_snapshot(args.env_file, session_id)
            interrupted = (
                response.status_code == 200
                and "ask_human" in interrupt_snapshot["ai_tool_calls"]
                and interrupt_snapshot["pending_interrupts"] > 0
            )
            record(
                "ask_human_interrupt",
                "passed" if interrupted else "degraded",
                **request_summary(response),
                snapshot=interrupt_snapshot,
            )
            if interrupted:
                response = client.post(
                    f"{base_url}/api/v1/chatbot/chat",
                    headers=session_headers,
                    json={"messages": [{"role": "user", "content": "I do not approve. Make no changes."}]},
                )
                resume_snapshot = graph_snapshot(args.env_file, session_id)
                resumed = (
                    response.status_code == 200
                    and resume_snapshot["pending_interrupts"] == 0
                    and "ask_human" in resume_snapshot["tool_messages"]
                    and resume_snapshot["last_ai_message"]
                )
                record(
                    "ask_human_resume",
                    "passed" if resumed else "degraded",
                    **request_summary(response),
                    snapshot=resume_snapshot,
                )
            else:
                record("ask_human_resume", "unverified", reason="no_interrupt_to_resume")

            metrics_response = client.get(f"{base_url}/metrics")
            metric_names = (
                "starlette_requests_total",
                "http_requests_total",
                "llm_inference_duration_seconds_count",
                "llm_stream_duration_seconds_count",
            )
            record(
                "prometheus_metrics",
                "passed" if metrics_response.status_code == 200 else "failed",
                **request_summary(metrics_response),
                families={name: name in metrics_response.text for name in metric_names},
            )

            config_values = dotenv_values(args.env_file)
            record(
                "session_naming",
                "disabled" if str(config_values.get("SESSION_NAMING_ENABLED")).lower() == "false" else "unverified",
                enabled=str(config_values.get("SESSION_NAMING_ENABLED")).lower() == "true",
            )
            record(
                "langfuse_tracing",
                "disabled" if str(config_values.get("LANGFUSE_TRACING_ENABLED")).lower() == "false" else "unverified",
                enabled=str(config_values.get("LANGFUSE_TRACING_ENABLED")).lower() == "true",
            )

            counts_before_clear = checkpoint_counts(args.env_file, session_id)
            response = client.delete(f"{base_url}/api/v1/chatbot/messages", headers=session_headers)
            counts_after_clear = checkpoint_counts(args.env_file, session_id)
            response_payload = get_json(response) if response.is_success else {}
            record(
                "clear_history",
                "passed"
                if response.status_code == 200
                and counts_before_clear.get("checkpoints", 0) > 0
                and all(value == 0 for value in counts_after_clear.values())
                else "failed",
                **request_summary(response),
                counts_before=counts_before_clear,
                counts_after=counts_after_clear,
                acknowledged=bool(response_payload.get("message")),
            )

            response = client.get(f"{base_url}/api/v1/chatbot/messages", headers=session_headers)
            history_after_clear = get_json(response) if response.is_success else {}
            cleared_messages = history_after_clear.get("messages", [])
            record(
                "history_empty_after_clear",
                "passed" if response.status_code == 200 and cleared_messages == [] else "failed",
                **request_summary(response),
                message_count=len(cleared_messages) if isinstance(cleared_messages, list) else None,
            )
        except Exception as exc:
            record("runner", "failed", error_type=type(exc).__name__)
        finally:
            if session_id and session_token:
                try:
                    response = client.delete(
                        f"{base_url}/api/v1/auth/session/{session_id}",
                        headers={"Authorization": f"Bearer {session_token}"},
                    )
                    record(
                        "cleanup_session",
                        "passed" if response.status_code in (200, 204) else "failed",
                        **request_summary(response),
                    )
                except Exception as exc:
                    record("cleanup_session", "failed", error_type=type(exc).__name__)
            if user_token and session_id:
                try:
                    response = client.get(
                        f"{base_url}/api/v1/auth/sessions",
                        headers={"Authorization": f"Bearer {user_token}"},
                    )
                    remaining = response.json() if response.is_success else []
                    record(
                        "cleanup_verified",
                        "passed"
                        if response.status_code == 200
                        and all(item.get("session_id") != session_id for item in remaining)
                        else "failed",
                        **request_summary(response),
                        remaining_count=len(remaining) if isinstance(remaining, list) else None,
                    )
                except Exception as exc:
                    record("cleanup_verified", "failed", error_type=type(exc).__name__)

            # Memory writes are intentionally asynchronous in the template. Give the
            # task a short window to finish before exact database cleanup and verify it.
            time.sleep(10)
            try:
                cleanup = cleanup_database(args.env_file, email, session_id)
                record("cleanup_database", "passed" if cleanup["cleared"] else "failed", **cleanup)
            except Exception as exc:
                record("cleanup_database", "failed", error_type=type(exc).__name__)

    print(json.dumps(checks, ensure_ascii=True, sort_keys=True))
    failed = [name for name, item in checks["checks"].items() if item.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
