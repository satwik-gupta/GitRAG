---
name: sop
document_type: SOP
version: "1.0"
description: "Standard Operating Procedure — deployment, operations, and maintenance runbook"
---

# {REPO_NAME} — Standard Operating Procedure

## Purpose & Scope
{{context: repo_name, repo_url, language_distribution, entry_points}}
State the purpose of this document: what system it covers, who its intended audience is (operators, on-call engineers, DevOps), and what operational activities it governs. Identify the primary entry points that operators interact with and the system boundaries in scope.

## Prerequisites & Environment Setup
{{context: external_dependencies, language_distribution, module_structure}}
List all prerequisites an operator must have before performing any procedure: required software and exact versions, environment variables that must be set (name, format, description, whether optional), network access requirements, credentials or secrets needed, and any one-time setup steps. Format as a checklist.

## Deployment Procedure
{{context: entry_points, external_dependencies, module_structure}}
{{diagram: sequenceDiagram}}
Provide a numbered, step-by-step deployment sequence covering: pre-deployment checks, database migration steps, service startup order, health check verification, and rollback procedure. Include a Mermaid sequenceDiagram showing the deployment coordination between the operator, the service, and its dependencies. Flag any steps that require manual approval or downtime.

## Standard Operational Workflows
{{context: entry_points, call_graph, module_structure}}
Document 3–5 common operational tasks with step-by-step instructions for each. Examples include: starting/stopping the service, triggering a re-index, clearing the cache, rotating credentials, or scaling up workers. Each workflow should be a numbered list with the expected output or success criteria stated.

## Health Monitoring & Observability
{{context: entry_points, external_dependencies, module_summaries}}
Describe what operators should monitor: key health endpoints (URLs, expected responses), critical log patterns that indicate problems, metrics to watch (latency, queue depth, cache hit rate, error rate), and alerting thresholds. Include any known false-positive alert patterns and how to distinguish them from genuine incidents.

## Troubleshooting Guide
{{context: module_summaries, call_graph, external_dependencies}}
Provide a symptom → diagnosis → resolution table for the 5 most common failure modes. For each: describe the observable symptom, the most likely root cause, diagnostic commands or log queries to confirm, and the remediation steps. Include at least one scenario for each external dependency (database, vector store, LLM API).

## Maintenance & Upgrade Procedures
{{context: external_dependencies, module_structure, language_distribution}}
Document scheduled and on-demand maintenance tasks: dependency updates (how to test, what to watch for), database migrations (forward and rollback), model or index refreshes, certificate rotations, and log rotation. Specify who is authorised to perform each task and any change-management requirements.
