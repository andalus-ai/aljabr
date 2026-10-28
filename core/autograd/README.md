# Legacy autograd and trainer migration area

This directory is **not currently a supported, independently buildable training module**.

The sources mix two concerns:

- IR-based graph differentiation under `tech.kayys.aljabr.autograd`, which imports `tech.kayys.aljabr.ir.*`.
- Training utilities and optimizer implementations under `tech.kayys.aljabr.train`.

The current repository tree does not contain the source modules for the declared `aljabr-ir` and `aljabr-trainer-api` snapshot dependencies. The Maven descriptor also refers to a trainer API source path that is not present in this checkout. Do not infer that enabling this module proves it can compile.

## Build behavior

The default build intentionally excludes this legacy module. Maintainers can opt in to inspect its current integration state:

```shell
./gradlew -PwithLegacyAutograd=true :core:autograd:compileJava
```

This is a diagnostic opt-in, **not a promise that compilation succeeds**. Until the missing IR and trainer contracts are migrated to source modules or otherwise provided reproducibly, failures are expected and should be reported as unresolved integration work. Do not enable this module by default as a workaround.

## Ownership and migration direction

- Aljabr should own reusable tensor differentiation, gradient rules, parameter abstractions, and optimizer primitives.
- Tafkir should consume those primitives for datasets, fine-tuning, and higher-level training workflows; it must not be a required dependency for Aljabr's foundational engine or AQLI inference builds.
- Keep the inference build independent from this legacy training area.

Before declaring training support ready, migrate the IR contracts to an owned module boundary, define parameter/tensor ownership and release semantics, and add deterministic numerical tests for gradient propagation and optimizer updates. A source-only audit or successful configuration alone is not validation.
