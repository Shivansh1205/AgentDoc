/**
 * A real AgentDoc report, not hand-written mock data.
 *
 * Produced by `agentdoc diagnose examples/langgraph_trace_flawed_example.json`
 * against the deliberately-flawed regression fixture, so every value here -
 * the justifications, the confidences, the turn indices - is genuine
 * classifier output. Embedded so the dashboard has something real to show
 * before anyone uploads a file.
 */

import type { AgentDocReport } from './schema'

export const SAMPLE_REPORT: AgentDocReport = {
  "schema_version": 2,
  "model": "openai/gpt-oss-120b",
  "source_framework": "langgraph",
  "trace_turn_count": 7,
  "total_failures": 5,
  "narrative": "This run shows 5 failures.",
  "category_counts": [
    {
      "category": "system_design_issues",
      "count": 2
    },
    {
      "category": "inter_agent_misalignment",
      "count": 1
    },
    {
      "category": "task_verification",
      "count": 2
    }
  ],
  "ranked_failure_modes": [
    {
      "failure_mode": "FM-1.2",
      "count": 1
    },
    {
      "failure_mode": "FM-1.5",
      "count": 1
    },
    {
      "failure_mode": "FM-2.5",
      "count": 1
    },
    {
      "failure_mode": "FM-3.1",
      "count": 1
    },
    {
      "failure_mode": "FM-3.3",
      "count": 1
    }
  ],
  "flagged_failures": [
    {
      "failure_mode": "FM-2.5",
      "category": "inter_agent_misalignment",
      "turn_indices": [
        5
      ],
      "justification": "The writer gave a revenue figure ($380 million) that contradicts the researcher's confirmed figure ($412 million), showing the writer disregarded the prior agent's input.",
      "confidence": 0.95
    },
    {
      "failure_mode": "FM-1.2",
      "category": "system_design_issues",
      "turn_indices": [
        5
      ],
      "justification": "The writer's role is to draft the final answer based on the researcher’s gathered facts; providing an inconsistent revenue number breaches the role specification.",
      "confidence": 0.93
    },
    {
      "failure_mode": "FM-3.3",
      "category": "task_verification",
      "turn_indices": [
        5
      ],
      "justification": "The writer failed to verify the revenue number against the researcher’s confirmed data before responding, resulting in an incorrect answer.",
      "confidence": 0.9
    },
    {
      "failure_mode": "FM-3.1",
      "category": "task_verification",
      "turn_indices": [
        6
      ],
      "justification": "The supervisor marked the task complete and closed the conversation without any review of the writer’s answer, terminating the interaction prematurely.",
      "confidence": 0.92
    },
    {
      "failure_mode": "FM-1.5",
      "category": "system_design_issues",
      "turn_indices": [
        6
      ],
      "justification": "By ending the dialogue without checking the writer’s output, the supervisor demonstrated a lack of awareness of the termination condition that verification is required.",
      "confidence": 0.85
    }
  ],
  "turns": [
    {
      "step": 0,
      "role": "system",
      "agent": null,
      "content": "You are part of a two-agent research team. Researcher gathers facts; Writer drafts the final answer for the user.",
      "tool_calls": [],
      "timestamp": "2026-08-27T14:00:00Z",
      "parent_step": null,
      "handoff_to": null,
      "metadata": {
        "chunk_index": 0,
        "node": "researcher"
      }
    },
    {
      "step": 1,
      "role": "human",
      "agent": null,
      "content": "What was Acme Corp's revenue in their most recent quarterly earnings report?",
      "tool_calls": [],
      "timestamp": "2026-08-27T14:00:00Z",
      "parent_step": null,
      "handoff_to": null,
      "metadata": {
        "chunk_index": 0,
        "node": "researcher"
      }
    },
    {
      "step": 2,
      "role": "agent",
      "agent": "researcher",
      "content": "I'll pull up their latest earnings report.",
      "tool_calls": [
        {
          "name": "web_search",
          "call_id": "call_1",
          "args": {
            "query": "Acme Corp Q2 2026 earnings report revenue"
          },
          "result": "Acme Corp Investor Relations: Q2 2026 revenue was $412 million, up 8% year-over-year.",
          "error": null
        }
      ],
      "timestamp": "2026-08-27T14:00:02Z",
      "parent_step": null,
      "handoff_to": null,
      "metadata": {
        "chunk_index": 1,
        "node": "researcher"
      }
    },
    {
      "step": 3,
      "role": "agent",
      "agent": "researcher",
      "content": "Let me double check that figure with another search before confirming.",
      "tool_calls": [
        {
          "name": "web_search",
          "call_id": "call_2",
          "args": {
            "query": "Acme Corp Q2 2026 earnings report revenue"
          },
          "result": "Acme Corp Investor Relations: Q2 2026 revenue was $412 million, up 8% year-over-year.",
          "error": null
        }
      ],
      "timestamp": "2026-08-27T14:00:05Z",
      "parent_step": null,
      "handoff_to": null,
      "metadata": {
        "chunk_index": 2,
        "node": "researcher"
      }
    },
    {
      "step": 4,
      "role": "agent",
      "agent": "researcher",
      "content": "Confirmed: Acme Corp's Q2 2026 revenue was $412 million, up 8% year-over-year, per their investor relations page. Handing off to writer for the final answer.",
      "tool_calls": [],
      "timestamp": "2026-08-27T14:00:08Z",
      "parent_step": null,
      "handoff_to": null,
      "metadata": {
        "chunk_index": 3,
        "node": "researcher"
      }
    },
    {
      "step": 5,
      "role": "agent",
      "agent": "writer",
      "content": "Acme Corp's most recent quarterly revenue was $380 million, a slight decline from the prior quarter. Let me know if you need any other financial details.",
      "tool_calls": [],
      "timestamp": "2026-08-27T14:00:11Z",
      "parent_step": null,
      "handoff_to": null,
      "metadata": {
        "chunk_index": 4,
        "node": "writer"
      }
    },
    {
      "step": 6,
      "role": "agent",
      "agent": "supervisor",
      "content": "Writer has produced a response, so the task is marked complete. Closing the conversation without further review.",
      "tool_calls": [],
      "timestamp": "2026-08-27T14:00:13Z",
      "parent_step": null,
      "handoff_to": null,
      "metadata": {
        "chunk_index": 5,
        "node": "supervisor"
      }
    }
  ]
}
