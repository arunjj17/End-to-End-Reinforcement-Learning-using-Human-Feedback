# AgentRLHF Project Notes

## Memory and tool registry extension

The project should include synthetic memory and an MCP-style local tool registry.

### Synthetic memory

Use deterministic synthetic memory entries such as:

- User prefers concise answers.
- User is beginner-level.
- User prefers Python examples.
- User wants cost-focused comparisons.
- User prefers step-by-step explanations.
- User prefers safety-first answers.

Memory should affect `personalized_answer` tasks and should be evaluated in reward scoring.

### MCP-style local tool registry

Use a local Python registry, not a real MCP server.

Each tool should include:

- name
- action
- description
- cost
- latency
- risk
- recommended_task_types

The registry should power:

- tool execution
- reward scoring
- cost/latency metrics
- CLI reporting

Do not add external network tools or closed-source APIs.

