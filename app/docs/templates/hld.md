---
name: hld
document_type: HLD
version: "1.0"
description: "High-Level Design Document — comprehensive system architecture overview"
---

# {REPO_NAME} — High-Level Design Document

## Executive Overview
{{context: repo_name, repo_url, language_distribution, total_files, commit_sha, entry_points}}
Write 2–3 paragraphs covering: (1) what this system does and the specific problem it solves, (2) the primary programming languages and their roles in the architecture, (3) the scale of the codebase and its main entry points. Be technically precise. Reference actual file counts, language names, and the repository URL.

## System Architecture
{{context: module_structure, call_graph, external_dependencies, entry_points}}
{{diagram: flowchart}}
Describe the top-level system architecture by identifying the major subsystems and how they interact. Include a Mermaid flowchart (flowchart TD) that shows the high-level component graph — include at least the primary modules as nodes and arrows labelled with the type of interaction (HTTP, async call, DB query, FFI, etc.). Reference actual module and directory names from the codebase.

## Module Breakdown
{{context: module_structure, language_distribution, file_tree}}
For each major module or package identified in the repository, provide a subsection with: its name, primary responsibility, the language(s) it uses, approximate size (file count), and its key public interfaces or exports. Structure this as a table or bulleted list for readability.

## Data Flow
{{context: call_graph, external_dependencies, module_structure}}
{{diagram: sequenceDiagram}}
Describe how data moves through the system end-to-end, from an initial trigger (user request, scheduled job, webhook) through processing, persistence, and response. Include a Mermaid sequenceDiagram tracing the primary happy-path request from entry point to final output, naming the actual classes or modules involved.

## External Integrations & Dependencies
{{context: external_dependencies, call_graph}}
List all external systems, APIs, libraries, and services that this repository depends on. For each, state: the dependency name, its purpose in the system, the integration pattern (HTTP, gRPC, library call, environment variable), and whether it is a hard or optional dependency.

## Key Design Decisions & Trade-offs
{{context: module_structure, language_distribution, call_graph}}
Identify 3–5 significant architectural or design decisions evident in the codebase. For each decision, state: what was chosen, what alternatives exist, and the likely rationale based on the code structure. Focus on decisions that have system-wide implications.
