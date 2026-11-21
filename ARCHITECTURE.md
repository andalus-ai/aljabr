# Aljabr architecture documentation

> **Authoritative architecture:** [Unified Computational Architecture](docs/architecture/ALJABR_UNIFIED_ARCHITECTURE.md)

This file is retained as a stable entry point for existing links. Its former module map and SPI-elevation plan described an earlier repository layout and are no longer authoritative. Do not use the historical module names, dependency examples or ownership proposals from older copies of this document to place new code.

## Start here

| Need | Source of truth |
|---|---|
| Product scope, ecosystem ownership, layering and module-boundary decisions | [Unified Computational Architecture](docs/architecture/ALJABR_UNIFIED_ARCHITECTURE.md) |
| Contributor rules, separation of concerns, pragmatic SOLID, canonical ownership and review checklist | [AGENTS.md](AGENTS.md) |
| Practical implementation, testing and documentation workflow | [Developer Guide](docs/DEVELOPER_GUIDE.md) |
| Framework-level programming model and computation/execution boundaries | [Framework Architecture](docs/architecture/FRAMEWORK_ARCHITECTURE.md) |
| Provider discovery, compatibility, initialization and configuration ownership | [Backend Ownership ADR](docs/architecture/ADR-0001-BACKEND-OWNERSHIP.md) |
| Published artifacts, dependency profiles and consumer distribution | [Modular Distribution](docs/architecture/MODULAR_DISTRIBUTION.md) |
| Current repository/module inventory and implementation status | [Capability Inventory](docs/architecture/COMPUTATIONAL_CAPABILITY_INVENTORY.md) |

## Current ownership summary

- **Aljabr** owns Java computational APIs and semantics, reusable tensor/model/format-loading primitives, execution contracts, provider SPIs and provider compatibility/conformance mechanisms.
- **AQLI** owns inference-serving workflows: serving-time model/session lifecycle, request admission and scheduling policy, generation policy, client-facing streaming/cancellation, serving APIs and deployment operations.
- **Tafkir** owns training and fine-tuning workflow orchestration, datasets, experiments and training lifecycle; Aljabr supplies reusable computational primitives.
- Product-specific policy belongs to the consuming product/runtime. Aljabr exposes reusable mechanisms and contracts without becoming an application or deployment control plane.

These are ownership boundaries, not a claim that every capability is already implemented or validated. Check the capability inventory and module documentation before depending on a feature.

## Placement rule

Use the actual Gradle project graph and current source tree as evidence. Do not infer a module path from this document, create a new module to match a diagram, or treat a source package as a stable public API merely because it currently exists. Before moving or introducing a type, inspect its consumers, dependency direction, publication metadata, resources/service descriptors, compatibility requirements and tests.

For a new architectural decision, update the unified architecture or add a focused ADR. Keep this page as a navigation/compatibility entry point; do not duplicate detailed architecture rules here.
