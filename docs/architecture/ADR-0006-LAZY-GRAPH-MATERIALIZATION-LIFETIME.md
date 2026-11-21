# ADR-0006: Lazy Graph Materialization Lifetime and Ownership

- **Status:** Proposed
- **Date:** 2026-11-21
- **Decision owners:** Aljabr maintainers
- **Scope:** Tensor lazy execution, graph materialization, optimizer rewrites, and lazy tensor lifecycle

## Context

The lazy tensor implementation records immutable `GraphNode` DAGs and caches forward results on operation nodes. A graph can contain shared upstream nodes: two lazy expressions may depend on the same intermediate node. Evaluation materializes every uncached operation in the reachable DAG.

The current lifecycle is not graph-complete:

- `GraphExecutor.evaluate()` caches each operation result on its node.
- `LazyTensor.release()` releases only the wrapper's root materialization and does not clear the root cache or release intermediate operation materializations.
- A released root can therefore remain marked evaluated while still referencing a released tensor.
- Releasing an upstream wrapper can invalidate a downstream expression that shares its node.
- `GraphOptimizer` can rebuild nodes and copy materialized tensor references, so node replacement must not create duplicate tensor ownership or lose cleanup responsibility.

A recursive traversal from one wrapper's root is not safe by itself. Shared nodes may still be used by other live expressions, and the `Tensor` interface has no universal retain operation. `TensorStorage.retain()` is not a substitute: not every tensor implementation exposes storage, and storage lifetime is not the same abstraction as graph-node lifetime.

## Decision

Adopt an explicit graph-materialization lease model before implementing graph-wide release.

### 1. Separate graph-node lifetime from tensor-storage lifetime

The lifetime unit is a graph node's materialization, not an assumed storage implementation. The graph lifecycle must use the public `Tensor.release()` contract and must not depend on CPU buffers, FFM segments, Metal resources, CUDA pointers, or storage-specific retain methods.

A graph lease pins the graph nodes it needs. Multiple live lazy expressions may hold leases over overlapping node sets. Releasing one lease must not invalidate nodes still pinned by another lease.

### 2. Distinguish borrowed leaves from graph-owned outputs

- **CONSTANT nodes:** reference caller-owned tensors. Graph cleanup must never release them.
- **Operation nodes:** own the tensor materialized for that operation, subject to the graph lease count.
- **Returned evaluation value:** `LazyTensor.eval()` exposes the materialized result as a borrowed value. Its validity is bounded by the owning lazy graph's live lifetime; evaluation does not silently transfer ownership.
- **Gradients:** retain their existing separate ownership rules. Leaf-owned accumulated gradients are not graph forward materializations and must not be released by graph cleanup.

A future API for obtaining an independently owned result must make that transfer or copy explicit; it must not be inferred from calling `eval()`.

### 3. Release only after the final graph lease

A materialized operation node may be cleared and its tensor released only after no live graph lease requires that node. Releasing a wrapper is idempotent. Releasing one wrapper must not release a shared upstream materialization while another wrapper still depends on it.

When a graph lease is finally released, cleanup should proceed in reverse topological order so dependent operation outputs are released before their inputs. Cleanup must attempt all eligible releases; the first failure is propagated and later cleanup failures are attached as suppressed exceptions.

After the final lease is released:

- operation-node materialization references are cleared;
- those materialized tensors are released exactly once;
- CONSTANT leaf references remain untouched;
- the released wrapper rejects operations that require a live graph;
- a cleared operation node is no longer reported as evaluated.

### 4. Make optimizer rewrites ownership-neutral

Optimization must not duplicate ownership when it rebuilds or eliminates nodes.

An optimized graph must participate in the same effective lifetime accounting as the source graph. Copying a materialized tensor reference to a replacement node must not create a second independent release obligation. Rewrites that return an existing input node must preserve that node's lease. Rewrites that eliminate nodes must make their materializations eligible for cleanup only when no remaining lease needs them.

The implementation may use a shared materialization scope, node-level lease counters, or another internal mechanism, but the observable rules above are mandatory. The mechanism must support DAG identity and must not rely on structural equality.

### 5. Define evaluation and release concurrency

The implementation must prevent release from racing an active evaluation into a use-after-release. It must choose and document one consistent policy:

- serialize evaluation and final lease release for a graph scope; or
- use an explicit active-operation pin so release is deferred until in-flight evaluation/backward work exits.

A released wrapper must reject new evaluation and backward operations. Concurrent release calls must remain idempotent. No graph lock may be held while invoking arbitrary backend work if doing so can introduce avoidable lock-order or reentrancy hazards.

### 6. Preserve failure safety

Failed evaluation must continue to roll back only materializations created by that evaluation attempt, preserving pre-existing cached nodes and borrowed CONSTANT leaves.

Graph cleanup must attempt every eligible tensor release. A cleanup failure must not replace the primary evaluation or backward failure; subsequent cleanup failures must be suppressed. The implementation must avoid clearing a node's cache unless it still refers to the exact materialization being released.

## Required conformance tests

The lifecycle implementation is not complete until tests cover:

1. Releasing an unevaluated lazy expression does not release any CONSTANT input.
2. Releasing an evaluated root releases its owned operation materializations, including intermediates, exactly once.
3. Two expressions sharing an upstream operation can be released in either order; the surviving expression remains evaluable and usable.
4. Releasing an upstream wrapper after constructing a downstream expression does not invalidate the downstream expression.
5. Multiple wrappers over the same graph node do not cause premature or duplicate release.
6. Repeated and concurrent release attempts are idempotent.
7. A released wrapper rejects evaluation, backward, and graph-building operations.
8. Optimized graphs that rebuild, eliminate, or reuse nodes preserve one cleanup obligation per materialized tensor.
9. Failed evaluation rolls back only outputs created by that attempt and preserves pre-existing cached outputs.
10. Cleanup attempts every eligible release and preserves the primary failure with later cleanup failures suppressed.
11. Backward retains access to required forward materializations until backward completes.
12. CONSTANT leaves and leaf-owned gradients are not released by graph materialization cleanup.

Use identity-based tracking in tests. Include tensors whose `release()` records calls and tensors whose `release()` throws, so tests can verify exactly-once behavior and failure aggregation without depending on a particular backend.

## Migration and compatibility

The implementation should preserve the existing lazy tensor arithmetic API and avoid adding backend-specific ownership methods to `Tensor`. Any new public lifecycle surface should be minimal and consistent with `Tensor.release()`; prefer an internal lease/scope abstraction unless a public ownership operation is demonstrably necessary.

Do not make `GraphNode` recursively release arbitrary tensors as a standalone side effect. Do not add a universal tensor `retain()` method solely to accommodate one lazy graph implementation without first auditing eager tensors, views, aliases, and every backend.

## Consequences

- Lazy expressions can safely share DAG nodes without one wrapper invalidating another.
- Intermediate forward materializations have a defined cleanup path.
- Graph optimization must preserve materialization identity and ownership.
- The graph lifecycle becomes explicit and testable independently of CPU/GPU storage details.
- Implementation complexity increases: graph leases and active evaluation/backward work must be coordinated. That complexity is preferable to silent leaks, double releases, and use-after-release.

## Implementation sequence

1. Add focused lifecycle tests using identity-tracking tensors.
2. Introduce the internal lease/materialization-scope mechanism and make `LazyTensor.release()` use it.
3. Integrate optimized/rebuilt nodes into the same ownership scope.
4. Add evaluation/backward concurrency and failure-path tests.
5. Document the borrowed lifetime of values returned by `eval()` in the public API.

This ADR defines the target contract; it does not claim that the current implementation already satisfies it.
