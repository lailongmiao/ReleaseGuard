"""Add review objects without rewriting legacy facts or authorization hashes."""

from typing import Sequence

from alembic import op

revision: str = "v11_backend_01"
down_revision: str | None = "v1_backend_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Only additive structures; no existing aggregate, budget or ACL is reset."""
    op.execute("ALTER TABLE v1_sources ADD COLUMN authz_version integer NOT NULL DEFAULT 0")
    op.execute("""CREATE FUNCTION v11_bump_source_authz() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN IF NEW.acl IS DISTINCT FROM OLD.acl OR NEW.deleted IS DISTINCT FROM OLD.deleted THEN
            NEW.authz_version := OLD.authz_version + 1;
        END IF; RETURN NEW; END $$""")
    op.execute(
        "CREATE TRIGGER v11_source_authz BEFORE UPDATE OF acl,deleted ON v1_sources FOR EACH ROW EXECUTE FUNCTION v11_bump_source_authz()"
    )
    op.execute("ALTER TABLE v1_commands ADD COLUMN contract_version varchar(64)")
    op.execute("ALTER TABLE v1_commands ADD COLUMN budget_category varchar(16)")
    op.execute("ALTER TABLE v1_commands ADD COLUMN intent_hash varchar(64)")
    op.execute("CREATE INDEX v11_command_intent ON v1_commands(resource,intent_hash,status)")
    op.execute("ALTER TABLE v1_generations ADD COLUMN object_id varchar(128)")
    op.execute("ALTER TABLE v1_generations ADD COLUMN snapshot_id varchar(128)")
    op.execute("""CREATE TABLE v11_repositories (
        repository_id varchar(32) PRIMARY KEY, source_id varchar(128) NOT NULL REFERENCES v1_sources(source_id),
        name text NOT NULL, github_id bigint UNIQUE, github_owner_id bigint,
        verification varchar(32) NOT NULL DEFAULT 'unverified', policy_version varchar(128) NOT NULL,
        verified_at timestamptz, CHECK ((github_id IS NULL) = (github_owner_id IS NULL)))""")
    op.execute("""CREATE TABLE v11_objects (
        object_id varchar(128) PRIMARY KEY REFERENCES local_mvp_records(change_id),
        repository_id varchar(32) NOT NULL REFERENCES v11_repositories(repository_id),
        kind varchar(32) NOT NULL CHECK(kind IN ('pr','commit','branch_compare')),
        selector jsonb NOT NULL, selector_key varchar(64) NOT NULL,
        object_version integer NOT NULL DEFAULT 0 CHECK(object_version>=0),
        active_snapshot_id varchar(128), active_generation_id varchar(128),
        collection_status varchar(32) NOT NULL DEFAULT 'not_started', sync_claim varchar(128),
        created_at timestamptz NOT NULL DEFAULT now())""")
    op.execute("CREATE INDEX v11_objects_listing ON v11_objects(repository_id,kind,created_at,object_id)")
    op.execute("""CREATE TABLE v11_snapshots (
        snapshot_id varchar(128) PRIMARY KEY, object_id varchar(128) NOT NULL REFERENCES v11_objects(object_id),
        sequence integer NOT NULL CHECK(sequence BETWEEN 1 AND 20), content_hash varchar(64) NOT NULL,
        facts jsonb NOT NULL, captured_at timestamptz NOT NULL,
        UNIQUE(object_id,sequence), UNIQUE(object_id,snapshot_id), UNIQUE(object_id,content_hash))""")
    op.execute("""CREATE TABLE v11_diff_files (
        snapshot_id varchar(128) NOT NULL REFERENCES v11_snapshots(snapshot_id), file_id varchar(128) NOT NULL,
        path text NOT NULL, facts jsonb NOT NULL, PRIMARY KEY(snapshot_id,file_id), UNIQUE(snapshot_id,path))""")
    op.execute("""CREATE TABLE v11_observations (
        observation_id varchar(128) PRIMARY KEY, object_id varchar(128) NOT NULL,
        snapshot_id varchar(128) NOT NULL, facts jsonb NOT NULL, captured_at timestamptz NOT NULL,
        FOREIGN KEY(object_id,snapshot_id) REFERENCES v11_snapshots(object_id,snapshot_id))""")
    op.execute("""CREATE TABLE v11_rule_sets (
        rule_set_id varchar(128) PRIMARY KEY, source_id varchar(128) NOT NULL REFERENCES v1_sources(source_id),
        applicability jsonb NOT NULL, content_hash varchar(64) NOT NULL, manifest jsonb NOT NULL,
        generation varchar(128) REFERENCES v1_generations(generation), version integer NOT NULL DEFAULT 0)""")
    op.execute("""CREATE TABLE v11_analyses (
        analysis_id varchar(128) PRIMARY KEY, object_id varchar(128) NOT NULL,
        revision integer NOT NULL CHECK(revision BETWEEN 1 AND 20), snapshot_id varchar(128) NOT NULL,
        generation varchar(128) NOT NULL REFERENCES v1_generations(generation),
        observation_id varchar(128) NOT NULL REFERENCES v11_observations(observation_id),
        owner_id integer NOT NULL, command_id varchar(128) NOT NULL, input_hash varchar(64) NOT NULL,
        status varchar(32) NOT NULL DEFAULT 'started', report jsonb NOT NULL DEFAULT '{}',
        dependencies jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE(object_id,revision), FOREIGN KEY(object_id,snapshot_id) REFERENCES v11_snapshots(object_id,snapshot_id),
        FOREIGN KEY(owner_id,command_id) REFERENCES v1_commands(owner_id,command_id))""")
    op.execute("CREATE INDEX v11_analyses_hold ON v11_analyses(object_id,input_hash,status)")
    op.execute("""CREATE TABLE v11_budget_carryovers (
        evidence_id varchar(128) PRIMARY KEY, evidence_hash varchar(64) NOT NULL UNIQUE,
        ledger_id varchar(128) NOT NULL DEFAULT '',
        total_charge numeric(14,8) NOT NULL CHECK(total_charge>=0),
        smoke_charge numeric(14,8) NOT NULL CHECK(smoke_charge>=0 AND smoke_charge<=total_charge),
        unknown_charge numeric(14,8) NOT NULL CHECK(unknown_charge>=0 AND unknown_charge<=total_charge),
        reconciled boolean NOT NULL DEFAULT false, created_at timestamptz NOT NULL DEFAULT now())""")
    op.execute(
        "ALTER TABLE v1_generations ADD CONSTRAINT v11_generation_object_fk FOREIGN KEY(object_id,snapshot_id) REFERENCES v11_snapshots(object_id,snapshot_id)"
    )
    op.execute(
        "ALTER TABLE v1_generations ADD CONSTRAINT v11_generation_scope UNIQUE(object_id,snapshot_id,generation)"
    )
    op.execute(
        "ALTER TABLE v11_objects ADD CONSTRAINT v11_active_snapshot FOREIGN KEY(object_id,active_snapshot_id) REFERENCES v11_snapshots(object_id,snapshot_id)"
    )
    op.execute(
        "ALTER TABLE v11_objects ADD CONSTRAINT v11_active_generation FOREIGN KEY(object_id,active_snapshot_id,active_generation_id) REFERENCES v1_generations(object_id,snapshot_id,generation)"
    )
    op.execute(
        "ALTER TABLE v11_analyses ADD CONSTRAINT v11_analysis_generation FOREIGN KEY(object_id,snapshot_id,generation) REFERENCES v1_generations(object_id,snapshot_id,generation)"
    )


def downgrade() -> None:
    """Rollback must never delete saved history or release unknown charges."""
    raise RuntimeError("v11_destructive_downgrade_requires_separate_review")
