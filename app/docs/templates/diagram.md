---
name: diagram
document_type: DIAGRAM
version: "1.0"
description: "Architecture Diagram Suite — multi-view Mermaid diagram collection"
---

# {REPO_NAME} — Architecture Diagram Suite

## System Overview
{{context: module_structure, call_graph, entry_points, external_dependencies}}
{{diagram: flowchart}}
Generate a comprehensive system overview as a Mermaid flowchart (flowchart TD). The diagram must include: all top-level modules as rectangular nodes, external services as stadium-shaped nodes (([name])), databases as cylindrical nodes ([(name)]), and labelled arrows showing the direction and type of each interaction. Group related components using Mermaid subgraphs. Every node must have a human-readable label. Cover the complete component surface with no orphaned nodes.

## Component Interaction Sequence
{{context: call_graph, module_summaries, entry_points}}
{{diagram: sequenceDiagram}}
Generate a Mermaid sequenceDiagram showing the primary end-to-end request flow. Include at minimum: the external caller (User/API Client), the main application entry point, all intermediate service layers touched during request processing, the data store, and the response path. Use activate/deactivate blocks to show synchronous blocking calls. Use dashed arrows (-->>) for async responses. Label every message with the actual function or method name where determinable.

## Class Hierarchy
{{context: module_summaries, call_graph, language_distribution}}
{{diagram: classDiagram}}
Generate a Mermaid classDiagram covering the most architecturally significant classes and interfaces. Include: inheritance relationships (--|>), composition (--*), aggregation (--o), and dependency (..>). Show the 3–5 most important methods per class. Mark abstract classes with <<abstract>> and interfaces with <<interface>>. Group classes that belong to the same module using Mermaid namespace blocks.

## Data & State Flow
{{context: call_graph, module_summaries, external_dependencies}}
{{diagram: flowchart}}
Generate a Mermaid flowchart (flowchart LR) showing how data objects flow and transform through the system. Start from the data ingestion or input point and trace through each transformation, enrichment, validation, and storage step to the final output or retrieval point. Label each arrow with the data type or object name being passed. Use decision diamonds for branching logic (cache hit/miss, error paths, conditional processing).
