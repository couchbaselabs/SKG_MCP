## MCP Server — Startup Guide

### Install dependencies
```
pip install -r requirements.txt
```

### Environment variables (set before starting)
```
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
```

### Start the server
```
python mcp_server.py --cb-endpoint couchbases://your-cluster.cloud.couchbase.com --cb-username your-user --cb-password your-password --bucket YourBucket --chunk-collection DesignDocdata --structure-collection file_structure --port 8000 --workers 1
```

`--chunk-collection` and `--structure-collection` are optional — they default to `DesignDocdata` and `file_structure`.

`--workers` defaults to `4` — each worker is a separate process with its own Couchbase connection, so concurrent users don't block each other. Increase if more team members are hitting the server simultaneously.

All flags can also be set as env vars (`CB_ENDPOINT`, `CB_USERNAME`, `CB_PASSWORD`, `CB_BUCKET`, `CB_CHUNK_COLLECTION`, `CB_STRUCTURE_COLLECTION`) and omitted from the command.

---

### How the middleware works

`scope` and `vector_index` are **not fixed at startup** — they are resolved per request from the URL query parameters. This means one server instance serves multiple users, each pointing at their own Couchbase scope.

The middleware (`_DynamicConfigMiddleware`) intercepts every incoming request, reads `scope` and `vector_index` from the query string, and sets them as async-safe context variables (`ContextVar`) that all tool functions read during that request. When the request ends, the values are reset.

**Connecting from Droid (or any MCP client):**
```
http://your-server:8000/mcp?scope=your_scope&vector_index=your_vector_index_name
```

- `scope` — the Couchbase scope your data lives in (e.g. `mcp_test`, `team_alpha`)
- `vector_index` — the FTS vector index name for your design doc collection (e.g. `hyperscale_AzureFoundry_text-embedding`)

If either param is omitted, `scope` defaults to `mcp_test` and `vector_index` defaults to empty.

---

### AWS deployment (EC2 / ECS)

The server binds to `0.0.0.0` so it is reachable from outside the container. Put an ALB in front on port 443 — users connect via `https://your-domain/mcp?scope=...` with no port in the URL. The internal port (`--port`) is an ECS task definition / EC2 security group detail only.
