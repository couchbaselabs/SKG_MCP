"""
mcp_server.py

Repo-agnostic FastMCP server. Exposes tools to Factory.ai Droid for
understanding any repo indexed into Couchbase.

Env vars required:
  OPENAI_API_KEY   — for text-embedding-3-large (embeddings)
  GEMINI_API_KEY   — for generate_design_doc (gemini-2.5-flash)

Static config (CLI flags or env vars at startup):
  --cb-endpoint / CB_ENDPOINT
  --cb-username / CB_USERNAME
  --cb-password / CB_PASSWORD
  --bucket      / CB_BUCKET
  --chunk-collection    / CB_CHUNK_COLLECTION     (default: DesignDocdata)
  --structure-collection/ CB_STRUCTURE_COLLECTION (default: file_structure)

Per-request dynamic config (URL query params — resolved by middleware):
  ?scope=<scope>               (default: mcp_test)
  ?vector_index=<index_name>   (default: empty)
"""

from __future__ import annotations

import json
import os
import sys
from contextvars import ContextVar
from datetime import timedelta
from pathlib import Path
from typing import Any

from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware

from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, QueryOptions
from couchbase.auth import PasswordAuthenticator

# ── Config (static — passed as CLI flags at server startup) ───────────────────
ENDPOINT  = os.environ.get("CB_ENDPOINT", "")
USERNAME  = os.environ.get("CB_USERNAME", "")
PASSWORD  = os.environ.get("CB_PASSWORD", "")
BUCKET    = os.environ.get("CB_BUCKET", "")

CHUNK_COLLECTION     = os.environ.get("CB_CHUNK_COLLECTION",     "DesignDocdata")
STRUCTURE_COLLECTION = os.environ.get("CB_STRUCTURE_COLLECTION", "file_structure")

EMBEDDING_MODEL = "text-embedding-3-large"

# ── Per-request dynamic config (from URL query params) ────────────────────────
DEFAULT_SCOPE        = "mcp_test"
DEFAULT_VECTOR_INDEX = ""

_current_scope:        ContextVar[str] = ContextVar("scope",        default=DEFAULT_SCOPE)
_current_vector_index: ContextVar[str] = ContextVar("vector_index", default=DEFAULT_VECTOR_INDEX)


class _DynamicConfigMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        scope        = request.query_params.get("scope",        DEFAULT_SCOPE)
        vector_index = request.query_params.get("vector_index", DEFAULT_VECTOR_INDEX)
        tok_s = _current_scope.set(scope)
        tok_v = _current_vector_index.set(vector_index)
        try:
            return await call_next(request)
        finally:
            _current_scope.reset(tok_s)
            _current_vector_index.reset(tok_v)

# ── Result limits (not exposed to Droid) ──────────────────────────────────────
TOP_K_DESIGN_DOCS = 4
TOP_K_CODEBASE    = 5
TOP_K_TRACE       = 5

# ── State ─────────────────────────────────────────────────────────────────────
_state: dict[str, Any] = {}

def _bn() -> str:
    return _state["bucket_name"]


# ══════════════════════════════════════════════════════════════════════════════
# MCP server
# ══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    "skg_mcp",
    instructions=(
        "Call startup(working_dir=<your current working directory>) at session start.\n"
        "It returns repo overviews for repos matching your directory — read them carefully.\n"
        "The overview contains subsystems, key classes, and implementation patterns already\n"
        "established in the codebase. When building something new, check these patterns first —\n"
        "the right base class and extension point is likely already there for a similar feature.\n\n"

        "Before grepping broadly or reading multiple files to understand a feature or plan an\n"
        "implementation, call trace_feature(feature, repo) first. It searches internal design\n"
        "docs and PRDs that Grep and Read cannot access, plus the files that implement the feature.\n"
        "Starting with Grep on a large repo means missing design context that only exists in docs.\n\n"

        "If trace_feature results feel thin on design context, follow up with search_design_docs —\n"
        "it often surfaces architectural decisions and constraints that are absent from the code.\n\n"

        "When the user asks to generate or write a design doc:\n"
        "  → Call generate_design_doc(feature, context, reference_feature) directly.\n"
        "  → Do NOT call search_design_docs  — that returns only 4 chunks and is\n"
        "    the wrong tool. generate_design_doc fetches the full reference doc internally.\n"
        "  → If you already have context about the feature, pass it and call immediately.\n"
        "  → Only call trace_feature or search_codebase first if you are genuinely missing\n"
        "    details about the existing implementation that you cannot answer from memory.\n\n"
        "Tools available:\n"
        "- trace_feature(feature, repo) — design docs + implementing files in one call.\n"
        "- search_design_docs(query) — for architectural decisions and 'why' questions only.\n"
        "  NOT for gathering reference content before generate_design_doc.\n"
        "- search_codebase(query, repo, include_config) — pre-indexed file summaries and class names.\n"
        "  Returns only code files by default. Set include_config=True ONLY when you specifically\n"
        "  need config/infra files (YAML, Terraform, Dockerfile, proto, etc.).\n"
        "- get_file_summaries(file_paths, repo) — triage grep results without opening files.\n"
        "- get_dependencies(file_path, repo) — top callers ranked by centrality.\n"
        "- generate_design_doc(feature, context, reference_feature) — industry-level design doc\n"
        "  with diagrams. Fetches the full reference doc internally. Call directly when asked.\n"
        "- get_repo_overview(repo) — full repo overview on demand.\n"
        "- Pass repo= on any search to scope to a single repo."
    ),
)


def _log_result(tool_name: str, result: str) -> str:
    sep = "─" * 60
    print(f"\n{sep}\n[{tool_name}]\n{sep}\n{result}\n{sep}\n")
    return result


def _init_state(endpoint: str, username: str, password: str, bucket: str):
    openai_key = os.environ.get("OPENAI_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    print("[startup] Connecting to Couchbase...")
    auth    = PasswordAuthenticator(username, password)
    cluster = Cluster.connect(endpoint, ClusterOptions(auth, enable_dns_srv=False))
    cluster.wait_until_ready(timedelta(seconds=60))
    print("[startup] Couchbase ready.")

    _state["cluster"]    = cluster
    _state["bucket"]     = cluster.bucket(bucket)
    _state["bucket_name"] = bucket
    _state["openai"]     = OpenAI(api_key=openai_key)
    _state["gemini_key"] = gemini_key


# ── Tools ─────────────────────────────────────────────────────────────────────

def _fetch_overview(repo_name: str) -> str:
    doc_id = f"{repo_name}::__overview__"
    try:
        col      = _state["bucket"].scope(_current_scope.get()).collection(STRUCTURE_COLLECTION)
        result   = col.get(doc_id)
        overview = result.content_as[dict].get("overview_text", "")
        if overview:
            return overview
    except Exception:
        pass
    return f"(no overview cached — run generate_overviews.py for '{repo_name}')"


def _match_repos(repos: list[str], working_dir: str) -> list[str]:
    """Return repos whose name appears as a path component in working_dir,
    or as an immediate subdirectory of working_dir."""
    if not working_dir:
        return []
    p = Path(working_dir)
    # All path components of working_dir (e.g. /home/user/couchbase-cloud → 'couchbase-cloud')
    parts = set(p.parts)
    # Immediate subdirectories (e.g. /home/user/Repos contains couchbase-cloud/, autovec/)
    try:
        parts.update(child.name for child in p.iterdir() if child.is_dir())
    except Exception:
        pass
    return [r for r in repos if r in parts]


@mcp.tool()
def startup(working_dir: str = "") -> str:
    """
    MANDATORY: Call this first at every session start, before anything else.
    Pass your current working directory as working_dir.

    Returns: all indexed repo names + full overviews for repos that match
    your working directory. Everything in one call — no follow-up needed.
    If working_dir is omitted or no repos match, only the repo list is returned.
    """
    rows  = list(_state["cluster"].query(
        f"SELECT DISTINCT repo FROM `{_bn()}`.`{_current_scope.get()}`.`{STRUCTURE_COLLECTION}` "
        f"WHERE repo IS NOT MISSING AND (type IS MISSING OR type != '__repo_overview__')"
    ))
    repos = [r["repo"] for r in rows if r.get("repo")]

    if not repos:
        return _log_result("startup", "No repos indexed.")

    matched = _match_repos(repos, working_dir)
    sections = [f"Indexed repos: {', '.join(repos)}"]

    if matched:
        sections.append(f"Matched to working_dir '{working_dir}': {', '.join(matched)}\n")
        for repo_name in matched:
            overview = _fetch_overview(repo_name)
            sections.append(f"{'='*40}\nOVERVIEW: {repo_name}\n{'='*40}\n{overview}")
    else:
        sections.append(
            f"No repos matched working_dir '{working_dir}'.\n"
            "If you need an overview for a specific repo, call get_repo_overview(repo_name)."
        )

    return _log_result("startup", "\n\n".join(sections))


@mcp.tool()
def get_repo_overview(repo_name: str) -> str:
    """
    Returns the cached overview for a specific repo on demand.
    Use this when the user explicitly asks about a repo that startup() did not load,
    or when working on a repo that did not match the working directory.
    """
    overview = _fetch_overview(repo_name)
    return _log_result("get_repo_overview", f"OVERVIEW: {repo_name}\n\n{overview}")


@mcp.tool()
def search_design_docs(query: str) -> str:
    """
    Semantic search over design docs, architecture docs, and PRDs.
    Use for questions about intended behavior, design decisions, or 'why' something
    was built a certain way. Returns the most relevant passages with source info.

    DO NOT use this to gather reference content before calling generate_design_doc.
    generate_design_doc fetches the full reference doc internally — calling this first
    only gives you 4 chunks and wastes context. When asked to generate a design doc,
    call generate_design_doc directly.
    """
    openai  = _state["openai"]
    cluster = _state["cluster"]

    resp   = openai.embeddings.create(input=query, model=EMBEDDING_MODEL)
    vector = resp.data[0].embedding

    rows = list(cluster.query(
        f"""
        SELECT META().id AS chunk_id,
               `text-to-embed`,
               `meta-data`.`associated-titles` AS titles,
               `xmeta-data`.`filename`         AS filename
        FROM `{_bn()}`.`{_current_scope.get()}`.`{CHUNK_COLLECTION}`
        ORDER BY APPROX_VECTOR_DISTANCE(`text-embedding`, $vector, "L2")
        LIMIT {TOP_K_DESIGN_DOCS}
        """,
        QueryOptions(named_parameters={"vector": vector}),
    ))

    chunk_ids = [r.get("chunk_id", "") for r in rows]
    print(f"[search_design_docs] chunks: {chunk_ids}")

    chunks = [
        f"[{r.get('filename','')} › {r.get('titles','')}]\n{r.get('text-to-embed','')}"
        for r in rows
    ]
    result = "\n\n---\n\n".join(chunks) if chunks else "No relevant design doc passages found."
    return _log_result("search_design_docs", result)


CODE_LANGUAGES = {"python", "go", "javascript", "typescript", "java", "rust", "c"}


@mcp.tool()
def search_codebase(query: str, repo: str = "", include_config: bool = False) -> str:
    """
    Keyword search over file summaries, class names, file paths, and subsystems.
    Use to find which file/class implements something or where specific logic lives.

    By default returns only code files (Python, Go, JS/TS, Java, Rust, C).
    - Pass repo= to limit to one repo (preferred when you know the repo).
    - Omit repo= to search across all indexed repos.
    - Set include_config=True ONLY when you specifically need config/infra files
      (YAML, Terraform, Dockerfile, Makefile, proto, etc.). Do NOT set this for
      general feature or implementation searches.

    Query strategy: if the user described a feature in natural language, translate
    it to technical terms the code likely uses before calling this tool.
    Results are ordered: code files first, then config/doc files.
    """
    cluster = _state["cluster"]
    tokens  = [t.lower() for t in query.split() if len(t) > 1]
    if not tokens:
        return "No matching files found."

    # Noise file patterns to always exclude from search results
    NOISE_SUFFIXES = (
        "_test.go", ".test.ts", ".test.js", ".test.tsx",
        ".spec.ts", ".spec.js", ".spec.tsx",
        ".stories.ts", ".stories.tsx", ".stories.js",
        # Generated files — they contain every class name in the system and dominate rankings
        ".gen.go", ".pb.go", ".generated.go", "openapi.gen.go",
    )

    conditions, params = [], {}
    for i, tok in enumerate(tokens):
        p = f"tok{i}"
        params[p] = f"%{tok}%"
        # class-name match scores 2, summary/subsystem match scores 1, path-only match scores 0
        conditions.append(
            f"(LOWER(file_summary) LIKE ${p} OR LOWER(file_path) LIKE ${p} "
            f"OR LOWER(subsystem) LIKE ${p} "
            f"OR ANY c IN classes SATISFIES LOWER(c.`name`) LIKE ${p} END)"
        )

    where = " OR ".join(conditions)
    if repo:
        where = f"repo = $repo AND ({where})"
        params["repo"] = repo
    if not include_config:
        params["code_langs"] = list(CODE_LANGUAGES)
        where = f"({where}) AND `language` IN $code_langs"

    # Always exclude test files, story files — they match generically and add noise
    noise_clauses = " AND ".join(f"file_path NOT LIKE '%{s}'" for s in NOISE_SUFFIXES)
    where = f"({where}) AND {noise_clauses}"

    # Build per-token class-match score for ranking (files where a class name matches rank highest)
    class_score_parts = []
    for i in range(len(tokens)):
        p = f"tok{i}"
        class_score_parts.append(
            f"(CASE WHEN ANY c IN classes SATISFIES LOWER(c.`name`) LIKE ${p} END THEN 2 ELSE 0 END)"
        )
    summary_score_parts = []
    for i in range(len(tokens)):
        p = f"tok{i}"
        summary_score_parts.append(f"(CASE WHEN LOWER(file_summary) LIKE ${p} THEN 1 ELSE 0 END)")

    score_expr = (
        "(CASE WHEN `language` IN ['python','go','javascript','typescript','java','rust','c'] THEN 4 ELSE 0 END)"
        + (" + " + " + ".join(class_score_parts) if class_score_parts else "")
        + (" + " + " + ".join(summary_score_parts) if summary_score_parts else "")
    )

    rows = list(cluster.query(
        f"SELECT file_path, subsystem, file_summary, repo, `language`, "
        f"ARRAY c.`name` FOR c IN classes END AS class_names "
        f"FROM `{_bn()}`.`{_current_scope.get()}`.`{STRUCTURE_COLLECTION}` "
        f"WHERE {where} "
        f"ORDER BY ({score_expr}) DESC "
        f"LIMIT {TOP_K_CODEBASE}",
        QueryOptions(named_parameters=params),
    ))
    print(f"[search_codebase] repo={repo!r} tokens={tokens} include_config={include_config} → {len(rows)} hits")

    hits = []
    for doc in rows:
        entry = f"**{doc.get('file_path','')}** ({doc.get('repo','')} / {doc.get('subsystem','')})"
        classes = ", ".join(c for c in (doc.get("class_names") or []) if c)
        if classes:
            entry += f"\nClasses: {classes}"
        if doc.get("file_summary"):
            entry += f"\n{doc['file_summary']}"
        hits.append(entry)

    result = "\n\n---\n\n".join(hits) if hits else "No matching files found."
    return _log_result("search_codebase", result)


@mcp.tool()
def get_file_summaries(file_paths: list, repo: str = "") -> str:
    """
    Returns stored summaries, class names, and subsystem for a list of specific file paths.
    Use this after a grep returns many matches to triage which files are worth reading,
    rather than opening each file individually.
    Pass up to 10 file paths. Much cheaper than reading the files themselves.
    """
    cluster = _state["cluster"]
    paths   = file_paths[:10]  # hard cap — not configurable

    params = {"paths": paths}
    where  = "file_path IN $paths"
    if repo:
        where += " AND repo = $repo"
        params["repo"] = repo

    rows = list(cluster.query(
        f"SELECT file_path, repo, subsystem, file_summary, `language`, "
        f"ARRAY c.`name` FOR c IN classes END AS class_names "
        f"FROM `{_bn()}`.`{_current_scope.get()}`.`{STRUCTURE_COLLECTION}` "
        f"WHERE {where}",
        QueryOptions(named_parameters=params),
    ))

    if not rows:
        return _log_result("get_file_summaries", "No summaries found for the given paths.")

    hits = []
    for doc in rows:
        classes = ", ".join(c for c in (doc.get("class_names") or []) if c)
        entry   = f"**{doc.get('file_path','')}** ({doc.get('repo','')}/{doc.get('subsystem','')})"
        if classes:
            entry += f"\nClasses: {classes}"
        entry += f"\n{doc.get('file_summary','(no summary)')}"
        hits.append(entry)

    return _log_result("get_file_summaries", "\n\n---\n\n".join(hits))


@mcp.tool()
def get_dependencies(file_path: str, repo: str = "") -> str:
    """
    Returns the full dependency graph for a file: what it imports and what imports it.
    Use to understand impact of changing a file or trace a data flow.

    Pass repo= when you know which repo the file belongs to (avoids ambiguity
    if multiple repos have the same file path).
    """
    cluster = _state["cluster"]
    params  = {"file_path": file_path}
    where   = "file_path = $file_path"
    if repo:
        where += " AND repo = $repo"
        params["repo"] = repo

    rows = list(cluster.query(
        f"SELECT file_path, repo, subsystem, file_summary, imports, imported_by, "
        f"ARRAY c.name FOR c IN classes END AS class_names "
        f"FROM `{_bn()}`.`{_current_scope.get()}`.`{STRUCTURE_COLLECTION}` "
        f"WHERE {where} LIMIT 1",
        QueryOptions(named_parameters=params),
    ))

    if not rows:
        return f"No document found for: {file_path}"

    doc   = rows[0]
    lines = [f"**{doc.get('file_path')}** ({doc.get('repo','')} / {doc.get('subsystem','')})", ""]

    if doc.get("file_summary"):
        lines += [doc["file_summary"], ""]

    classes = [c for c in (doc.get("class_names") or []) if c]
    if classes:
        lines += [f"Classes: {', '.join(classes)}", ""]

    imports = doc.get("imports") or []
    if imports:
        lines += ["**Imports:**"] + [f"  {i}" for i in imports] + [""]

    imported_by = doc.get("imported_by") or []
    if not imported_by:
        lines.append("Not imported by any tracked file.")
    elif len(imported_by) <= 20:
        lines.append(f"**Imported by {len(imported_by)} files:**")
        for doc_id in sorted(imported_by):
            lines.append(f"  - {doc_id.split('::', 1)[-1]}")
    else:
        # Too many dependents — rank by centrality (how many things import each dependent)
        # and filter out test/story files so only meaningful callers surface
        dep_rows = list(cluster.query(
            f"SELECT file_path, repo, subsystem, ARRAY_LENGTH(imported_by) AS dep_count "
            f"FROM `{_bn()}`.`{_current_scope.get()}`.`{STRUCTURE_COLLECTION}` "
            f"WHERE META().id IN {json.dumps(imported_by[:500])} "
            f"AND file_path NOT LIKE '%_test%' "
            f"AND file_path NOT LIKE '%.test.%' "
            f"AND file_path NOT LIKE '%.stories.%' "
            f"AND file_path NOT LIKE '%.spec.%' "
            f"ORDER BY dep_count DESC LIMIT 15"
        ))
        lines.append(
            f"**Imported by {len(imported_by)} files total. "
            f"Top 15 most-central callers (ranked by how many files import them):**"
        )
        for d in dep_rows:
            lines.append(
                f"  - **{d.get('file_path','')}** "
                f"({d.get('repo','')}/{d.get('subsystem','')}) "
                f"— itself imported by {d.get('dep_count', 0)} files"
            )

    return _log_result("get_dependencies", "\n".join(lines))


@mcp.tool()
def trace_feature(feature: str, repo: str = "") -> str:
    """
    Cross-references design docs with implementation for a feature or concept.
    Returns: what the design says, which files implement it, and blast radius.

    Pass repo= to limit the codebase search to one repo.
    Omit repo= for cross-repo features.
    """
    cluster = _state["cluster"]
    openai  = _state["openai"]
    lines   = [f"# Feature trace: {feature}", ""]

    # ── 1. Design docs ────────────────────────────────────────────────────────
    lines.append("## Design")
    try:
        resp   = openai.embeddings.create(input=feature, model=EMBEDDING_MODEL)
        vector = resp.data[0].embedding
        doc_rows = list(cluster.query(
            f"SELECT `text-to-embed`, `meta-data`.`associated-titles` AS titles, "
            f"`xmeta-data`.`filename` AS filename "
            f"FROM `{_bn()}`.`{_current_scope.get()}`.`{CHUNK_COLLECTION}` "
            f"ORDER BY APPROX_VECTOR_DISTANCE(`text-embedding`, $vector, 'L2') LIMIT 3",
            QueryOptions(named_parameters={"vector": vector}),
        ))
        lines += [
            f"[{r.get('filename','')} › {r.get('titles','')}]\n{r.get('text-to-embed','')}"
            for r in doc_rows
        ] or ["No design doc passages found."]
    except Exception as e:
        lines.append(f"Design doc search failed: {e}")

    lines.append("")

    # ── 2. Implementing files ─────────────────────────────────────────────────
    lines.append("## Implementation")
    tokens = [t.lower() for t in feature.split() if len(t) > 1]
    conditions, params = [], {}
    for i, tok in enumerate(tokens):
        p = f"tok{i}"
        params[p] = f"%{tok}%"
        conditions.append(
            f"(LOWER(file_summary) LIKE ${p} OR LOWER(file_path) LIKE ${p} "
            f"OR ANY c IN classes SATISFIES LOWER(c.`name`) LIKE ${p} END)"
        )
    where = " OR ".join(conditions) if conditions else "TRUE"
    if repo:
        where = f"repo = $repo AND ({where})"
        params["repo"] = repo

    impl_rows = list(cluster.query(
        f"SELECT file_path, repo, subsystem, file_summary, imported_by, "
        f"ARRAY c.`name` FOR c IN classes END AS class_names "
        f"FROM `{_bn()}`.`{_current_scope.get()}`.`{STRUCTURE_COLLECTION}` "
        f"WHERE {where} "
        f"ORDER BY (CASE WHEN `language` IN ['python','go','javascript','typescript','java','rust','c'] THEN 0 ELSE 1 END), ARRAY_LENGTH(imported_by) DESC "
        f"LIMIT {TOP_K_TRACE}",
        QueryOptions(named_parameters=params),
    ))

    if not impl_rows:
        lines.append("No implementing files found.")
    else:
        for doc in impl_rows:
            classes = ", ".join(c for c in (doc.get("class_names") or []) if c)
            lines.append(f"**{doc.get('file_path')}** ({doc.get('repo','')} / {doc.get('subsystem','')}) — used by {len(doc.get('imported_by') or [])} files")
            if classes:
                lines.append(f"  Classes: {classes}")
            if doc.get("file_summary"):
                lines.append(f"  {doc['file_summary']}")

    lines.append("")

    # ── 3. Blast radius ───────────────────────────────────────────────────────
    if impl_rows:
        top         = impl_rows[0]
        imported_by = top.get("imported_by") or []
        if imported_by:
            lines.append(f"## Blast radius: dependents of {top['file_path']}")
            dep_rows = list(cluster.query(
                f"SELECT file_path, repo, subsystem "
                f"FROM `{_bn()}`.`{_current_scope.get()}`.`{STRUCTURE_COLLECTION}` "
                f"WHERE META().id IN {json.dumps(imported_by)} LIMIT 10"
            ))
            for d in dep_rows:
                lines.append(f"  - **{d.get('file_path')}** ({d.get('repo','')} / {d.get('subsystem','')})")

    return _log_result("trace_feature", "\n".join(lines))


@mcp.tool()
def generate_design_doc(feature: str, context: str, reference_feature: str = "") -> str:
    """
    Generates an industry-level design doc with architecture diagrams for a new feature.

    IMPORTANT: When asked to generate a design doc, call this directly. Do NOT call
    search_design_docs first — this tool fetches the reference doc internally.
    If you already have enough context about the feature, call this immediately.
    Only use trace_feature or search_codebase first if you are missing specific details
    about the existing implementation or the new feature's API.

    What to pass as context:
      - Everything you already know about the existing implementation
      - The new feature's external API (from web search / user input / provider docs)
      - Which components, interfaces, schemas, and bindings need to change

    Parameters:
      feature:           short name of the new feature (e.g. "Azure AI Foundry model integration")

      context:           Everything gathered before calling this, written in plain English prose.
                         NO implementation code, NO function signatures, NO struct definitions,
                         NO JSON/YAML blocks. Describe everything in words — names, field shapes,
                         flows, and constraints as prose.

                         Must cover ALL of the following:

                         1. BACKGROUND: What problem this solves. Which existing integration it
                            most resembles and what is fundamentally different about this one.

                         2. SYSTEM COMPONENTS & INTERACTIONS: Every system component involved
                            (control plane, data plane, Eventing functions, external provider API,
                            Capella UI, integration store). How they connect and who calls whom.
                            Used to generate the architecture diagram — be precise about
                            which component initiates each call and what it sends.

                         3. END-TO-END FLOW: The exact sequence of steps from user action to
                            embedding stored — integration creation → workflow deployment →
                            pre-flight validation → data plane embedding call → vector write-back.
                            Used to generate the sequence diagram — name every actor and message.

                         4. REGIONS & AVAILABILITY: Which geographic regions the provider supports.
                            Any per-model regional restrictions. Exact region identifiers.

                         5. SUPPORTED MODELS: For each embedding model — display name, model ID,
                            supported regions, output dimensions, batch size limit, max tokens per
                            input, context window size. Tabular data.

                         6. INTEGRATION CREATION FIELDS: Every field the user fills when creating
                            the Capella integration. For each: name, valid values, WHY it is
                            needed, and what breaks if it is wrong.

                         7. WORKFLOW DEPLOYMENT FIELDS: Every field set at workflow/AI service
                            deployment time. For each: name, valid values, why it is needed.

                         8. AUTHENTICATION: All supported auth types, what headers or params carry
                            credentials, how tokens are obtained and whether they expire/refresh,
                            any regional auth restrictions.

                         9. API INTERACTION: The base endpoint URL pattern in plain text. How the
                            URL is parameterised (by region, resource name, deployment, api-version).
                            Every request field — name, type, purpose. Where the embedding vector
                            sits in the response. How this differs across models from the same
                            provider (per-model field differences described in prose).

                         10. CONTROL PLANE ↔ DATA PLANE INTERACTIONS: Which Eventing constant
                             bindings the control plane passes to the data plane. Exact name and
                             value of EMBEDDING_SERVICE_PROVIDER for this provider. Whether
                             OUTPUT_VECTOR_DIMENSIONS or new bindings are needed and why.
                             How the data plane reads and acts on each binding.

                         11. MODEL CARDS: Provider-specific fields required in the model card
                             document in the Capella control plane database.

                         12. NETWORK & SECURITY: Private connectivity options (e.g. Private
                             Endpoints, VPC peering). Data residency considerations. TLS requirements.

                         13. IMPLEMENTATION SCOPE: For every component that changes — file,
                             interface, type, constant, schema, binding, OpenAPI spec, UI view:
                             - WHAT changes (specifically what is added, modified, or extended)
                             - WHY it needs to change (the requirement or constraint driving it)
                             - WHAT TO KEEP IN MIND (gotchas, ordering, backward compatibility,
                               patterns that must be followed, decisions already locked in)
                             Each component is its own subsection. This is the most critical section.

                         14. ERROR HANDLING & EDGE CASES: How each failure mode is handled —
                             invalid credentials, unsupported region, model not available, quota
                             exceeded, malformed response, partial batch failure.

                         15. OPEN QUESTIONS: Unresolved decisions that need team alignment.

                         16. TESTING SCENARIOS: Numbered, concrete error cases — wrong API key,
                             mismatched region, model not in catalog, incorrect dimensions,
                             batch size violations, network path variations (with/without private
                             endpoint), etc.

                         17. REFERENCES: Use web search to find the actual documentation URLs
                             for the new provider. Include: the provider's embeddings API reference,
                             region availability page, authentication guide, model catalog, any
                             Marketplace or private connectivity docs. Pass these as a list so the
                             doc has accurate references — do NOT leave this to Gemini to infer.

      reference_feature: keyword to find the reference design doc (e.g. "bedrock").
                         If omitted, semantic search finds the closest match automatically.

    Output: an industry-level design doc with Mermaid diagrams, tables, and full technical
    depth — no implementation code, suitable for engineering review and approval.
    """
    cluster    = _state["cluster"]
    gemini_key = _state["gemini_key"]

    # ── 0. Strip any code blocks from context so they never reach Gemini ─────
    import re
    context = re.sub(r"```[\s\S]*?```", "[code block removed]", context)
    context = re.sub(r"`[^`\n]+`", "", context)  # inline backtick spans

    # ── 1. Get full section structure of reference doc (titles only, cheap) ───
    ref_filter = f"LOWER(`xmeta-data`.`filename`) LIKE $ref" if reference_feature else "TRUE"
    ref_params = {"ref": f"%{reference_feature.lower()}%"} if reference_feature else {}

    # Fetch all titles from the reference doc to reconstruct full section structure
    title_rows = list(cluster.query(
        f"SELECT `meta-data`.`associated-titles` AS titles, "
        f"`xmeta-data`.`filename` AS filename "
        f"FROM `{_bn()}`.`{_current_scope.get()}`.`{CHUNK_COLLECTION}` "
        f"WHERE {ref_filter} "
        f"ORDER BY `meta-data`.`page-number`, TONUMBER(SPLIT(`element-id`, '/')[-1])",
        QueryOptions(named_parameters=ref_params) if ref_params else QueryOptions(),
    ))

    if not title_rows:
        return _log_result("generate_design_doc", "No reference design doc found.")

    ref_filename = title_rows[0].get("filename", reference_feature or "unknown")

    # Deduplicate titles while preserving order — gives the complete section structure
    seen, ordered_titles = set(), []
    for row in title_rows:
        for t in (row.get("titles") or []):
            if t and t not in seen:
                seen.add(t)
                ordered_titles.append(t)

    section_structure = "\n".join(f"  - {t}" for t in ordered_titles)
    print(f"[generate_design_doc] reference='{ref_filename}' sections={len(ordered_titles)}")

    # ── 2. Fetch all chunks from the reference doc in order ───────────────────
    # Cap at 200 chunks — ~300 tokens each ≈ 60K tokens max for style reference.
    # The source of truth is `context`, not the reference doc.
    MAX_REF_CHUNKS = 200
    content_rows = list(cluster.query(
        f"SELECT `text-to-embed`, `meta-data`.`associated-titles` AS titles "
        f"FROM `{_bn()}`.`{_current_scope.get()}`.`{CHUNK_COLLECTION}` "
        f"WHERE {ref_filter} "
        f"ORDER BY `meta-data`.`page-number`, TONUMBER(SPLIT(`element-id`, '/')[-1]) "
        f"LIMIT {MAX_REF_CHUNKS}",
        QueryOptions(named_parameters=ref_params) if ref_params else QueryOptions(),
    ))

    if len(content_rows) == MAX_REF_CHUNKS:
        print(f"[generate_design_doc] WARNING: reference content capped at {MAX_REF_CHUNKS} chunks — doc may be truncated")
    print(f"[generate_design_doc] reference='{ref_filename}' chunks={len(content_rows)}")

    # Group chunks by primary section so Gemini sees the document as a hierarchy,
    # not a flat bag of text. Each chunk gets a position marker and full section path.
    total = len(content_rows)
    sections_seen: dict[str, list] = {}
    section_order: list[str] = []
    for i, r in enumerate(content_rows):
        titles = r.get("titles") or []
        if isinstance(titles, list):
            primary  = titles[0] if titles else "Preamble"
            breadcrumb = " > ".join(t for t in titles if t)
        else:
            primary = breadcrumb = str(titles)
        if primary not in sections_seen:
            sections_seen[primary] = []
            section_order.append(primary)
        sections_seen[primary].append((i + 1, total, breadcrumb, r.get("text-to-embed", "")))

    ref_parts = []
    for section_name in section_order:
        ref_parts.append(f"\n{'━'*60}\nSECTION: {section_name}\n{'━'*60}")
        for idx, tot, breadcrumb, text in sections_seen[section_name]:
            ref_parts.append(f"[chunk {idx}/{tot} | {breadcrumb}]\n{text}")
    reference_content = "\n\n".join(ref_parts)

    # ── 2. Call Gemini ────────────────────────────────────────────────────────
    from google import genai
    client = genai.Client(api_key=gemini_key)

    prompt = f"""You are a principal engineer at a cloud infrastructure company writing an
industry-standard design doc that will go through engineering review and approval.

You have three inputs:
1. The section structure of a reference design doc (ordered table of contents)
2. The full content of that reference doc grouped by section — each chunk labelled
   [chunk N/total | Section > Subsection] so you can trace the document hierarchy
   and calibrate how much depth each section deserves
3. Gathered context about the new feature (source of truth for everything you write)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 — NO IMPLEMENTATION CODE.
No Go, Python, JavaScript, or any programming language. No JSON/YAML blocks. No curl.
No pseudocode. Nothing inside triple-backtick fences except Mermaid diagrams (see Rule 2).
Describe API request fields by name, type, and purpose in plain prose. Describe response
fields in prose. No exceptions. A reviewer must be able to read this without being a coder.

RULE 2 — DIAGRAMS ARE MANDATORY. Use Mermaid inside triple-backtick mermaid fences.
Required diagrams — exactly two, do not skip either:
  a) SYSTEM ARCHITECTURE (flowchart LR or TD): all components in the system — Capella UI,
     Control Plane, Integration Store, Data Plane (Eventing worker + controller functions),
     External Provider API, Couchbase KV Store. Show which components call which, label
     the arrows with the action (e.g. "InvokeModel", "write embedding", "fetch binding").
  b) END-TO-END SEQUENCE (sequenceDiagram): the full lifecycle from user action to
     embedding stored — integration creation, workflow deployment, pre-flight validation,
     runtime embedding call, vector write-back. Every actor and every message labelled.
  Keep diagrams accurate to the gathered context. Do not draw components that don't exist.

RULE 3 — FULL TECHNICAL DEPTH WITHOUT CODE.
The doc must cover every one of these topics — missing any is a defect:
  □ Background and objective
  □ Regions and availability (table: region → supported models)
  □ Supported embedding models (table: model name, model ID, dimensions, batch size,
    max tokens, context window, region availability)
  □ Integration creation fields (per field: name, valid values, Why this field?)
  □ Workflow deployment fields (per field: name, valid values, Why this field?)
  □ Authentication mechanism (all auth types, header/param names, token lifecycle)
  □ API endpoint URL pattern in plain text — how region, resource, deployment parameterise it
  □ API request shape — every field, its type, and what the provider does with it (prose)
  □ API response shape — where the embedding vector sits, what else is returned (prose)
  □ Per-model API differences — how request/response fields differ across models (prose+table)
  □ Control plane ↔ data plane interactions — exact binding names and values
  □ Model card schema — provider-specific fields required
  □ Network and security — private connectivity options, data residency, TLS
  □ Implementation scope — see Rule 4
  □ Error handling and edge cases — per failure mode
  □ Open questions and next steps
  □ Testing scenarios — numbered, concrete, all error paths
  □ References

RULE 4 — IMPLEMENTATION SCOPE IS THE MOST CRITICAL SECTION.
For every component that changes (file, interface, type, constant, schema, binding,
OpenAPI spec, UI view), write a dedicated subsection with exactly three headings:
  **What changes**: specific name of what is added, modified, or extended
  **Why**: the requirement or constraint that forces this change
  **What to keep in mind**: gotchas, ordering constraints, backward compatibility concerns,
    patterns that must be followed, decisions already locked in that must not be revisited,
    anything that will cause a bug if the engineer forgets it
Do not collapse multiple components into a flat list. Each gets its own subsection.

RULE 5 — SPECIFICITY. Use exact values from the gathered context: endpoint URL templates,
model IDs, region identifiers, constant names, binding values, field names. Never write
placeholders like "<model-id>" or "the relevant endpoint". If a value is unknown, say so
explicitly and flag it as an open question.

RULE 6 — TABLES. Use markdown tables for: supported models, region availability, model
limits, auth options, binding values, per-model API differences. Prose alone is not enough
for comparative or tabular data.

RULE 7 — VERSIONING TABLE. The versioning table must have exactly one row:
  | 0 | [Date TBD] | [Author TBD] | Initial Draft |
  Do NOT copy the date, author name, or remarks text from the reference doc. Leave them
  as "[Date TBD]" and "[Author TBD]" — the human author will fill these in.

RULE 8 — REFERENCES MUST BE FOR THE NEW FEATURE ONLY.
  The References section must list documentation for the NEW feature being designed
  (e.g. Azure AI Foundry docs, the new provider's API reference, Marketplace links).
  Do NOT copy any reference from the reference doc. The reference doc's bibliography
  is about a different provider and is completely irrelevant here. If references were
  provided in the gathered context, use those. Otherwise write descriptive names
  (e.g. "Azure AI Foundry — Embeddings API reference") without copying Bedrock links.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFERENCE SECTION STRUCTURE ({ref_filename}):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{section_structure}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFERENCE CONTENT grouped by section (calibrate depth — write about the NEW feature):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{reference_content}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATHERED CONTEXT (source of truth — every section draws from this):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Feature: {feature}

{context}

Write the full design doc now.
Begin with: versioning table → table of contents → each section in order.
Mermaid diagrams go where they add the most clarity (after the relevant section heading).
No implementation code anywhere. Every item in the Rule 3 checklist must be covered.
"""

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    result   = (
        f"# Design Doc: {feature}\n\n"
        f"_Reference: '{ref_filename}' — {len(ordered_titles)} sections, {len(content_rows)} content chunks_\n\n"
        f"{response.text.strip()}"
    )
    return _log_result("generate_design_doc", result)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="MCP server")
    parser.add_argument("--port",                type=int, default=8000)
    parser.add_argument("--workers",             type=int, default=4,    help="Number of uvicorn worker processes")
    parser.add_argument("--cb-endpoint",         default=os.environ.get("CB_ENDPOINT", ""),                  help="Couchbase connection string")
    parser.add_argument("--cb-username",         default=os.environ.get("CB_USERNAME", ""),                  help="Couchbase username")
    parser.add_argument("--cb-password",         default=os.environ.get("CB_PASSWORD", ""),                  help="Couchbase password")
    parser.add_argument("--bucket",              default=os.environ.get("CB_BUCKET", ""),                    help="Couchbase bucket")
    parser.add_argument("--chunk-collection",    default=os.environ.get("CB_CHUNK_COLLECTION", "DesignDocdata"),      help="Collection for design doc chunks")
    parser.add_argument("--structure-collection",default=os.environ.get("CB_STRUCTURE_COLLECTION", "file_structure"), help="Collection for file structure")
    args = parser.parse_args()

    for flag, val in [("--cb-endpoint", args.cb_endpoint), ("--cb-username", args.cb_username),
                      ("--cb-password", args.cb_password), ("--bucket", args.bucket)]:
        if not val:
            print(f"ERROR: {flag} is required (or set env var)"); sys.exit(1)

    CHUNK_COLLECTION     = args.chunk_collection
    STRUCTURE_COLLECTION = args.structure_collection

    _init_state(args.cb_endpoint, args.cb_username, args.cb_password, args.bucket)

    app = mcp.streamable_http_app()
    app.add_middleware(_DynamicConfigMiddleware)

    print(f"Starting MCP server → http://0.0.0.0:{args.port}/mcp")
    print(f"  scope and vector_index are dynamic — pass as URL query params")
    print(f"  e.g. http://server:{args.port}/mcp?scope=team_xyz&vector_index=my_index")
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=args.workers)
