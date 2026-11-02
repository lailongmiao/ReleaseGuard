"""Explicit synthetic workspace reads/writes, with readback and no blind resend."""

import asyncio
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any, Literal, Protocol

from langfuse import Langfuse
from langfuse.api.resources.score.types.create_score_request import CreateScoreRequest
from pydantic import AwareDatetime, Field, model_validator
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.services.releaseguard.normalization import canonical_hash
from evals.releaseguard.schemas import Envelope, Hash, Name, Status


class LangfuseScope(Envelope):
    """Read and write grants are independent, bounded and expiring."""

    authorization_ref: Name
    write_authorization_ref: Name | None = None
    workspace_id: Name
    eval_run_id: Name
    dataset_hash: Hash
    case_ids: tuple[Name, ...] = Field(min_length=1)
    from_timestamp: AwareDatetime
    to_timestamp: AwareDatetime
    expires_at: AwareDatetime
    tag: Literal["synthetic-phase7"] = "synthetic-phase7"
    page_size: int = Field(default=25, strict=True, ge=1, le=100)
    max_pages: int = Field(default=10, strict=True, ge=1, le=100)
    allow_write: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def check_scope(self) -> "LangfuseScope":
        """No unbounded query or combined implicit write grant."""
        if self.from_timestamp >= self.to_timestamp or self.allow_write != bool(self.write_authorization_ref):
            raise ValueError("langfuse_scope_invalid")
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("langfuse_cases_duplicate")
        return self


class TraceRef(Envelope):
    """Safe association only, never raw Provider output or trace messages."""

    trace_id: Name
    workspace_id: Name
    eval_run_id: Name
    case_id: Name
    attempt: int = Field(strict=True, ge=1)
    dataset_hash: Hash
    timestamp: AwareDatetime
    tag: Literal["synthetic-phase7"] = "synthetic-phase7"
    synthetic: Literal[True] = True
    input_present: bool = Field(strict=True)
    output_present: bool = Field(strict=True)
    judge_observation_ids: tuple[Name, ...] = ()
    change_id: Name | None = None
    run_id: Name | None = None
    revision: int | None = Field(default=None, strict=True, ge=1)
    audit_ids: tuple[Name, ...] = ()


class ScoreIntent(Envelope):
    """Version-keyed numeric intent; no free-form comment or raw reasoning."""

    trace_id: Name
    observation_id: Name | None = None
    eval_run_id: Name
    case_id: Name
    attempt: int = Field(strict=True, ge=1)
    dataset_hash: Hash
    metric_id: Name
    metric_version: Name
    prompt_hash: Hash
    model: Name
    value: float = Field(strict=True, ge=0, le=1, allow_inf_nan=False)
    provenance: Literal["offline_rules", "real_judge"]

    @property
    def key(self) -> str:
        """Value excluded so same version/key with another intent conflicts."""
        return canonical_hash(self.model_dump(exclude={"value", "provenance"}))


class StoredScore(Envelope):
    """Confirmed actual score association and intended payload hash."""

    score_id: Hash
    intent_hash: Hash
    trace_id: Name
    observation_id: Name | None = None
    name: Name
    value: float = Field(ge=0, le=1, allow_inf_nan=False)


class IOResult(Envelope):
    """Adapter contract pass is independent from real workspace acceptance."""

    status: Status
    reason_code: Name
    provenance: Literal["fake", "workspace", "off"]
    traces: tuple[TraceRef, ...] = ()
    score_id: Hash | None = None
    workspace_acceptance: Literal["unverified"] = "unverified"


class LangfuseTransport(Protocol):
    """Scoped SDK/test double boundary; methods are async."""

    @property
    def provenance(self) -> Literal["fake", "workspace"]:
        """Read-only external evidence origin."""
        ...

    async def workspaces(self) -> set[str]:
        """Resolve the credential's actual workspace."""
        ...

    async def page(self, scope: LangfuseScope, page: int) -> tuple[TraceRef, ...]:
        """Read bounded synthetic references."""
        ...

    async def score(self, score_id: str) -> StoredScore | None:
        """Read back an explicitly keyed intent."""
        ...

    async def write(self, intent: ScoreIntent) -> None:
        """Send an explicitly authorized numeric score."""
        ...


class SDKLangfuseTransport:
    """Current synchronous SDK wrapped off event loop; no client autoconstruction."""

    provenance: Literal["workspace"] = "workspace"

    def __init__(self, client: Langfuse) -> None:
        """Client creation/credentials belong to authorized composition only."""
        self.client = client

    async def workspaces(self) -> set[str]:
        """Validate credential target, never infer it from configuration."""
        response = await asyncio.to_thread(self.client.api.projects.get)
        return {item.id for item in response.data}

    async def page(self, scope: LangfuseScope, page: int) -> tuple[TraceRef, ...]:
        """Server filters plus local revalidation; discard all raw trace content."""
        response = await asyncio.to_thread(
            self.client.api.trace.list,
            page=page,
            limit=scope.page_size,
            session_id=scope.eval_run_id,
            tags=[scope.tag],
            from_timestamp=scope.from_timestamp,
            to_timestamp=scope.to_timestamp,
        )
        result = []
        for trace in response.data:
            meta = trace.metadata if isinstance(trace.metadata, dict) else {}
            if scope.tag not in (trace.tags or []) or trace.session_id != scope.eval_run_id:
                raise PermissionError("langfuse_trace_scope_denied")
            # Require explicitly synthetic trace and binding metadata. Do not
            # trust any trace's arbitrary scores/observations as Judge evidence.
            observations = tuple(meta.get("judge_observation_ids", ()))
            if not set(observations) <= set(trace.observations):
                raise ValueError("langfuse_observation_unresolved")
            for observation_id in observations:
                observation = await asyncio.to_thread(self.client.api.observations.get, observation_id)
                if observation.trace_id != trace.id or observation.type != "GENERATION":
                    raise ValueError("langfuse_observation_unresolved")
            result.append(
                TraceRef.model_validate(
                    {
                        "trace_id": trace.id,
                        "workspace_id": scope.workspace_id,
                        "eval_run_id": trace.session_id,
                        "case_id": meta.get("case_id"),
                        "attempt": meta.get("attempt"),
                        "dataset_hash": meta.get("dataset_hash"),
                        "timestamp": trace.timestamp,
                        "synthetic": meta.get("synthetic"),
                        "input_present": bool(trace.input),
                        "output_present": bool(trace.output),
                        "judge_observation_ids": observations,
                        "change_id": meta.get("change_id"),
                        "run_id": meta.get("run_id"),
                        "revision": meta.get("revision"),
                        "audit_ids": tuple(meta.get("audit_ids", ())),
                    }
                )
            )
        return tuple(result)

    async def score(self, score_id: str) -> StoredScore | None:
        """404 alone means absent; any other read failure remains unknown."""
        try:
            score: Any = await asyncio.to_thread(self.client.api.score_v_2.get_by_id, score_id)
        except Exception as error:
            if getattr(error, "status_code", None) == 404:
                return None
            raise
        metadata = score.metadata if isinstance(score.metadata, dict) else {}
        return StoredScore.model_validate(
            {
                "score_id": score.id,
                "intent_hash": metadata.get("intent_hash"),
                "trace_id": score.trace_id,
                "observation_id": score.observation_id,
                "name": score.name,
                "value": score.value,
            }
        )

    async def write(self, intent: ScoreIntent) -> None:
        """Direct API request, then separate readback; never trust queued SDK success."""
        await asyncio.to_thread(
            self.client.api.score.create,
            request=CreateScoreRequest.parse_obj(
                {
                    "id": intent.key,
                    "traceId": intent.trace_id,
                    "observationId": intent.observation_id,
                    "name": intent.metric_id,
                    "value": intent.value,
                    "dataType": "NUMERIC",
                    "comment": "synthetic_phase7_evaluation",
                    "metadata": {"intent_hash": canonical_hash(intent)},
                }
            ),
            request_options={"max_retries": 0},
        )


def in_scope(trace: TraceRef, scope: LangfuseScope) -> bool:
    """Check each result even if server/test double ignored query filters."""
    return (
        trace.workspace_id == scope.workspace_id
        and trace.eval_run_id == scope.eval_run_id
        and trace.dataset_hash == scope.dataset_hash
        and trace.case_id in scope.case_ids
        and scope.from_timestamp <= trace.timestamp < scope.to_timestamp
        and trace.synthetic
        and trace.tag == scope.tag
    )


class LangfuseIO:
    """Finite per-run coordinator; uncertain sends cannot be blindly repeated."""

    def __init__(self, transport: LangfuseTransport | None, scope: LangfuseScope | None) -> None:
        """No network in constructor; off mode never creates a client."""
        self.transport, self.scope = transport, scope
        self.uncertain: set[str] = set()
        self.lock = asyncio.Lock()
        self.accepted: dict[str, TraceRef] = {}

    async def check(self, *, at: datetime, write: bool = False) -> None:
        """Recheck expiry and workspace on every operation."""
        if not self.scope or not self.transport or at >= self.scope.expires_at:
            raise PermissionError("langfuse_scope_denied")
        if write and not self.scope.allow_write:
            raise PermissionError("langfuse_write_denied")
        if await self.transport.workspaces() != {self.scope.workspace_id}:
            raise PermissionError("langfuse_workspace_denied")

    async def read(self, *, at: datetime) -> IOResult:
        """Bounded pages, no arbitrary-score skip, incomplete pages cannot pass."""
        self.accepted.clear()
        if not self.transport or not self.scope:
            return IOResult(status=Status.UNVERIFIED, reason_code="langfuse_off", provenance="off")
        try:
            async with asyncio.timeout(30):
                await self.check(at=at)
                traces: list[TraceRef] = []
                for page in range(1, self.scope.max_pages + 1):
                    await self.check(at=at)
                    batch = await self.transport.page(self.scope, page)
                    if len(batch) > self.scope.page_size or any(not in_scope(t, self.scope) for t in batch):
                        raise PermissionError("langfuse_trace_scope_denied")
                    traces.extend(batch)
                    if len({t.trace_id for t in traces}) != len(traces):
                        raise ValueError("langfuse_duplicate_trace")
                    if len(batch) < self.scope.page_size:
                        if not traces:
                            reason, status = "langfuse_no_samples", Status.UNVERIFIED
                        elif any(not t.input_present or not t.output_present for t in traces):
                            reason, status = "langfuse_missing_io", Status.UNVERIFIED
                        else:
                            reason, status = "langfuse_read_validated", Status.PASSED
                            self.accepted = {t.trace_id: t for t in traces}
                        return IOResult(
                            status=status,
                            reason_code=reason,
                            provenance=self.transport.provenance,
                            traces=tuple(traces),
                        )
                return IOResult(
                    status=Status.UNVERIFIED, reason_code="langfuse_page_limit", provenance=self.transport.provenance
                )
        except asyncio.CancelledError:
            return self.failure(TimeoutError())
        except Exception as error:
            return self.failure(error)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.01, max=0.05),
        retry=retry_if_exception_type(TimeoutError),
        reraise=True,
    )
    async def read_score(self, key: str) -> StoredScore | None:
        """Only safe reads have bounded exponential retries."""
        assert self.transport
        return await self.transport.score(key)

    async def publish(self, intent: ScoreIntent, *, at: datetime) -> IOResult:
        """Same key/different payload rejects; lost response only reconciles."""
        if not self.transport or not self.scope:
            return IOResult(status=Status.UNVERIFIED, reason_code="langfuse_off", provenance="off")
        async with self.lock:
            started = perf_counter()
            try:
                async with asyncio.timeout(30):
                    await self.check(at=at, write=True)
                    trace = self.accepted.get(intent.trace_id)
                    if (
                        trace is None
                        or not in_scope(trace, self.scope)
                        or (intent.eval_run_id, intent.case_id, intent.attempt, intent.dataset_hash)
                        != (trace.eval_run_id, trace.case_id, trace.attempt, trace.dataset_hash)
                    ):
                        raise PermissionError("langfuse_score_binding_denied")
                    if intent.provenance == "real_judge" and intent.observation_id not in trace.judge_observation_ids:
                        raise PermissionError("langfuse_judge_observation_denied")
                    expected = StoredScore(
                        score_id=intent.key,
                        intent_hash=canonical_hash(intent),
                        trace_id=intent.trace_id,
                        observation_id=intent.observation_id,
                        name=intent.metric_id,
                        value=intent.value,
                    )
                    current = await self.read_score(intent.key)
                    if current is not None:
                        if current != expected:
                            raise ValueError("langfuse_score_conflict")
                        self.uncertain.discard(intent.key)
                        return IOResult(
                            status=Status.PASSED,
                            reason_code="langfuse_readback_confirmed",
                            provenance=self.transport.provenance,
                            score_id=intent.key,
                        )
                    if intent.key in self.uncertain:
                        raise PermissionError("langfuse_write_outcome_unknown")
                    # Process-local uncertainty cannot prevent resend after a
                    # restart. Keep live sends closed until durable coordination
                    # or the server's idempotency contract has been verified.
                    if self.transport.provenance == "workspace":
                        raise PermissionError("langfuse_durable_write_gate_missing")
                    await self.check(at=at + timedelta(seconds=perf_counter() - started), write=True)
                    # Pin before sending. Cancellation/timeout preserves uncertainty.
                    self.uncertain.add(intent.key)
                    send_error = None
                    try:
                        await self.transport.write(intent)
                    except Exception as error:
                        send_error = error  # Reconcile once; never resend.
                    current = await self.read_score(intent.key)
                    if current != expected:
                        if send_error is not None and getattr(send_error, "status_code", None) in {401, 403}:
                            return self.failure(send_error)
                        raise ValueError("langfuse_write_unconfirmed")
                    self.uncertain.discard(intent.key)
                    return IOResult(
                        status=Status.PASSED,
                        reason_code="langfuse_readback_confirmed",
                        provenance=self.transport.provenance,
                        score_id=intent.key,
                    )
            except asyncio.CancelledError:
                return self.failure(TimeoutError())
            except Exception as error:
                return self.failure(error)

    def failure(self, error: Exception) -> IOResult:
        """Expose classified status only; never exception body/credentials."""
        code = getattr(error, "status_code", None)
        if code in {401, 403}:
            reason = "langfuse_permission_denied"
        elif str(error) in {
            "langfuse_scope_denied",
            "langfuse_write_denied",
            "langfuse_workspace_denied",
            "langfuse_trace_scope_denied",
            "langfuse_duplicate_trace",
            "langfuse_observation_unresolved",
            "langfuse_score_binding_denied",
            "langfuse_judge_observation_denied",
            "langfuse_score_conflict",
            "langfuse_write_outcome_unknown",
            "langfuse_durable_write_gate_missing",
            "langfuse_write_unconfirmed",
        }:
            reason = str(error)
        elif isinstance(error, TimeoutError):
            reason = "langfuse_timeout"
        else:
            reason = "langfuse_unavailable"
        return IOResult(
            status=Status.UNVERIFIED,
            reason_code=reason,
            provenance=self.transport.provenance if self.transport else "off",
        )
