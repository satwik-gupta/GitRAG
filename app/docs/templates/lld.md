---
name: lld
document_type: LLD
version: "1.0"
description: "Low-Level Design Document — class interfaces, algorithms, and internal contracts"
---

# {REPO_NAME} — Low-Level Design Document

## Module Overview
{{context: repo_name, module_structure, language_distribution, total_files}}
Provide a concise technical summary of each module, its internal sub-components, and the design pattern it follows (e.g., Repository pattern, Strategy, Factory, Observer). Reference actual directory and file names. State the module's cohesion level and any notable coupling concerns.

## Class & Interface Specifications
{{context: module_summaries, call_graph, language_distribution}}
{{diagram: classDiagram}}
For every significant class or interface in the codebase, provide: full qualified name, parent classes or implemented interfaces, list of public methods with signatures (name, parameters, return type), key attributes, and a one-line responsibility statement. Include a Mermaid classDiagram covering the most architecturally significant class hierarchy. Mark abstract classes and interfaces clearly.

## API Contract
{{context: entry_points, module_summaries, external_dependencies}}
Document every public-facing API endpoint or public function boundary. For each, provide: the function or endpoint signature, expected input types and constraints, output types and structure, and error conditions with their types. If it is an HTTP API, include method, path, request body schema, and response schema.

## Data Models & Storage Schema
{{context: module_summaries, external_dependencies, call_graph}}
{{diagram: erDiagram}}
Describe the internal data models: dataclasses, ORM models, Pydantic schemas, or structs. For persistence layers, include a Mermaid erDiagram showing the entity-relationship structure including foreign keys. For each entity: field names, types, constraints, and indexes.

## Algorithm & Control Flow
{{context: call_graph, module_summaries}}
{{diagram: sequenceDiagram}}
For the 2–3 most complex or critical algorithms in the codebase, provide: a step-by-step description of the algorithm's logic, its time and space complexity, the data structures it uses, and any concurrency or synchronisation requirements. Include a Mermaid sequenceDiagram for the most complex control flow to illustrate the method call sequence.

## Error Handling, Retries & Edge Cases
{{context: module_summaries, call_graph, external_dependencies}}
Document the error handling strategy: which error types are defined, how they propagate, retry policies (backoff strategy, max attempts), fallback behaviours, and how failures are surfaced to callers. Identify any known edge cases in the code (e.g., empty input handling, race conditions, size limits) and how they are addressed.

## Concurrency & Async Model
{{context: module_structure, call_graph, language_distribution}}
Describe the concurrency model: whether the system is async/await, thread-pool based, or event-driven. Identify all thread-pool executors, semaphores, locks, or async queues used. Flag any blocking calls that need executor delegation. Describe the session/connection lifecycle for shared resources (DB sessions, HTTP clients, ML models).
