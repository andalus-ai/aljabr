# Aljabr Provider Testkit

Optional checks for authors implementing `AljabrProvider`.

Add this module as a test dependency in a provider project:

```kotlin
dependencies {
    testImplementation("tech.kayys.aljabr:aljabr-spi-testkit:<version>")
}
```

Then run the provider check from the provider's own test suite:

```java
var result = ProviderConformance.verify(
    MyProvider::new,
    new ProviderContext("0.1.0", Map.of())
);
assertTrue(result.isConformant(), result.violations().toString());
```

The initial testkit checks only:
- provider construction;
- descriptor availability;
- initialization with a supplied context;
- cleanup, including cleanup after failed initialization.

It does **not** validate numerical correctness, hardware availability, performance,
thread safety, deterministic behavior, or the truth of capability declarations.
Those checks must be added alongside concrete capability SPIs and their reference
semantics. The testkit does not discover or download providers and does not create
security isolation boundaries.
