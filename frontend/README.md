# AgentDoc web console

A browser front end for AgentDoc reports: upload the JSON that
`agentdoc diagnose --json` writes and read the diagnosis as an interaction
graph, a swimlane timeline, and a filterable list of MAST failures.

Everything runs client-side. There is no backend and no network call at
runtime — the uploaded file is read with the browser's File API and parsed
in the page.

## Develop

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

`npm run build` emits a single self-contained `dist/index.html` with all JS
and CSS inlined, so it can be opened straight from the filesystem or handed
to someone as one file.

## Feeding it a report

The dashboard consumes the schema version 2 report shape (see
`../src/agentdoc/report/json_export.py`). Generate one with:

```bash
uv run agentdoc diagnose examples/langgraph_trace_flawed_example.json --json report.json
```

Reports written before schema v2 have no `turns` array; the upload page
detects that and says so rather than rendering an empty graph. The embedded
sample trace is real output from the flawed example fixture, not mock data.

## Layout

- `src/lib/schema.ts` — the report shape, MAST labels/colors, and upload
  validation
- `src/lib/derive.ts` — turns a report into agents, handoff edges, and the
  stats the panels show
- `src/components/` — the two views (`Landing`, `Dashboard`) and the three
  visualizations (`AgentGraph`, `Timeline`, `NetworkBackdrop`)
