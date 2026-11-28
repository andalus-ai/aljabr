# Tensor lifecycle conformance

Tensor providers should share the public lifecycle checks from the
`:plugins:ml:aljabr-tensor` test-fixtures variant rather than reimplementing
basic ownership assertions in each backend module.

## Add the test fixtures

In a provider's test dependencies:

```kotlin
testImplementation(testFixtures(project(":plugins:ml:aljabr-tensor")))
```

Use the shared assertions with a factory that creates a fresh tensor for every
invocation:

```java
TensorLifecycleAssertions.assertReleaseIsIdempotent(this::newTensor);

TensorLifecycleAssertions.assertOwnedGradientReleasedExactlyOnce(
        this::newTensor,
        this::newTensor);
```

## What the shared checks guarantee

- Repeated `release()` calls are harmless.
- `close()` remains safe after release.
- A gradient accepted by `setGrad` is owned by the receiving tensor.
- Closing the owner releases that transferred gradient exactly once.

The gradient assertion wraps the supplied gradient with a transparent
`Tensor` proxy that observes calls to `release()`; all other calls delegate
to the original tensor.

## Provider-specific checks still required

The shared checks deliberately do not assume every tensor exposes host
storage or has synchronous execution. Each provider should add focused tests
for capabilities it implements:

- **Views and aliases:** a derived view retains the storage it needs and
  remains valid for its documented lifetime after the source tensor closes.
- **Copies:** operations documented to return independent storage do not
  alias the source.
- **Borrowed access:** borrowed storage and gradient references are not
  released by the borrower; retaining storage creates a separately balanced
  reference.
- **Failure cleanup:** failed construction or operation setup does not leak
  partially acquired resources.
- **Asynchronous/device execution:** pending work and device fences keep
  referenced storage alive until completion.
- **Use after release:** resource-owning implementations reject operations
  that require resources already released, using their documented exception
  contract.

Do not infer view semantics from method names alone. Test each operation's
documented ownership behavior, and keep provider-specific assumptions out of
the common conformance helper.
