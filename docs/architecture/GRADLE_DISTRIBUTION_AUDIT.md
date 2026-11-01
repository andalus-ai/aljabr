# Aljabr Gradle Distribution Audit

**Snapshot:** 2026-11-01  
**Branch audited:** `main`  
**Method:** Static inspection of `settings.gradle.kts`, root `build.gradle.kts`, `gradle.properties`, selected module build files and the computational capability inventory.  
**Validation:** No Gradle command or dependency report was executed in this audit.

> This is a source-level build-configuration audit, not a resolved dependency graph. Findings below distinguish confirmed build-script behavior from transitive runtime dependency claims. Run Gradle dependency reports and external-consumer builds before changing publication or compiler behavior.

## 1. Executive summary

Aljabr has useful module boundaries already, but the root build currently applies several framework-wide build policies to every subproject. This creates build-time coupling even where production dependencies are narrow.

The highest-priority findings are:

1. **Global JDK and incubator compiler flags:** every subproject receives a JDK 25 toolchain, preview flags and `jdk.incubator.vector` compiler/runtime flags. The Vector API requirement should be scoped to modules that actually use it.
2. **Global Jandex indexing:** the root build adds Jandex to the root buildscript classpath and registers an index-generation task for every subproject, wiring it into every `classes` task. This is unnecessary work for plain numerical modules unless they consume Jandex.
3. **Global publication metadata:** all subprojects receive Maven publishing configuration and a publication with stale Wayang Platform URLs/developer information and Apache-2.0 metadata, while the repository advertises MIT. This is a release-integrity issue.
4. **Large default project inclusion:** CUDA, Metal, GGUF/storage integrations and dynamically discovered model-family projects are included by default in settings. Inclusion is not the same as a runtime dependency, but it increases configuration surface and can complicate build isolation.
5. **Global resolution policy:** dependency substitutions and version rules are installed on every configuration of every subproject, including a hard-coded Tafkir snapshot substitution. These rules should be scoped to the modules that need them or centralized as version constraints with explicit ownership.

The narrow production dependency examples inspected are healthy in direction: linear algebra implements numerics; NN and CPU backend implement tensor; the tensor module places CPU backend on test runtime only; and the facade exports tensor plus model SPI. However, a resolved graph is still needed to prove the absence of unintended transitives.

## 2. Confirmed module-level dependency edges

The following are directly visible in the inspected module build files:

| Module | Declared production dependencies | Test-only dependencies / notes |
|---|---|---|
| `:core:aljabr-numerics` | None declared | JUnit Jupiter |
| `:core:aljabr-linear-algebra` | `:core:aljabr-numerics` as `implementation` | JUnit Jupiter |
| `:core:aljabr-tensor` | None declared in its build file | `:core:aljabr-core` as `testImplementation`; CPU backend as `testRuntimeOnly` |
| `:core:aljabr-core` | `:core:aljabr-tensor` and `:core:aljabr-spi-model` as `api` | JUnit Jupiter |
| `:core:aljabr-nn` | `:core:aljabr-tensor` as `implementation` | Root test conventions apply |
| `:backend:cpu:aljabr-backend-cpu` | `:core:aljabr-tensor` as `implementation` | No extra runtime project dependency declared in inspected file |

Interpretation:

- The tensor-to-CPU relationship shown here is test-only, not a production dependency edge.
- The `:core:aljabr-core` facade is not a numerics-only facade: its public API exposes tensor and model SPI.
- The existing numerics module is a promising minimal foundation because it declares no production dependencies in its own build file.
- The existing linear algebra edge points inward to numerics.
- No dedicated statistics or classical-ML project is included in the current default settings snapshot; legacy training/autograd is opt-in.

These observations are not a substitute for inspecting all module build files and resolved configurations.

## 3. Root build policies that cross module boundaries

### 3.1 JDK, preview and Vector API flags

The root build configures every subproject with:
- Java toolchain language version 25.
- `--enable-preview` for every Java compile.
- `--add-modules=jdk.incubator.vector` for every Java compile.
- Test and JavaExec JVM arguments that include preview, Vector API and `--enable-native-access=ALL-UNNAMED`.

The root build comments associate preview/native-access flags with possible FFM use and Vector API flags with incubator vector support. These are not necessarily requirements of numerics, linear algebra, tensor or model SPI.

**Recommendation:** retain a clearly documented repository baseline if JDK 25 is an intentional product decision, but scope incubator module flags to modules that reference the Vector API. Scope native-access and preview flags to modules/tests that actually need them. Do not lower the toolchain as part of this audit without checking source language/API usage.

### 3.2 Jandex indexing

The root buildscript declares `io.smallrye:jandex` and the root `subprojects` block registers `generateJandexIndex` for every project, then makes each `classes` task depend on it.

**Recommendation:** move Jandex into a convention plugin or opt-in helper applied only to modules that require Quarkus/Jandex indexing. Plain numerical and tensor modules should not pay this build cost or inherit the buildscript concern.

### 3.3 Publication defaults and POM identity

The root build applies `maven-publish` to every subproject and creates `mavenJava` from each Java component. The generated POM metadata currently names Wayang Platform, points to `wayang-platform/wayang-platform`, identifies a Wayang developer and declares Apache License 2.0. The repository metadata identifies this project as MIT-licensed and the repository is `andalus-ai/aljabr`.

**Recommendation:** correct the POM metadata to the actual project identity and license, and make publication opt-in for supported public artifacts. Do not publish every model catalogue, experimental backend or internal SPI by default merely because it has a Java component. Maintain an explicit publication allowlist and verify generated POM/Gradle module metadata.

### 3.4 Global dependency substitutions and version rules

The root build installs substitutions for multiple Aljabr coordinates and an unconditional substitution of `tech.kayys.aljabr:tafkir-ml-core` to `tech.kayys.tafkir:tafkir-ml-core:0.1.0-SNAPSHOT`. It also applies version fallback rules for Quarkus, Mutiny, Caffeine, Jakarta APIs, Jackson, JUnit and other dependencies to all subproject configurations.

A substitution rule does not itself add a dependency, but it changes resolution whenever a matching dependency is requested. The hard-coded Tafkir snapshot makes an otherwise independent consumer's resolution behavior depend on a training integration decision.

**Recommendation:** scope Tafkir substitution to opt-in integration projects; scope other rules to the modules that need them or express supported shared versions through a version catalog/platform. Prefer explicit versions/constraints over broad silent fallback rules.

### 3.5 Repository configuration

All projects search Google Maven, Maven Central and Maven Local. Maven Local can make builds depend on artifacts present on a developer workstation and can hide missing published artifacts.

**Recommendation:** use Maven Central (and Google only for modules that need it) for normal resolution. Keep Maven Local behind an explicit development property/profile. Add CI verification that a clean build succeeds without developer-local artifacts.

## 4. Settings and project inclusion

The settings file confirms:

- Core numerical, tensor, NN and SPI projects are included directly.
- Legacy autograd is excluded unless `-PwithLegacyAutograd=true`.
- Tafkir is included as a composite build only with `-PwithTafkir=true`.
- Safetensors quantization is opt-in with Tafkir.
- HAT is opt-in and requires a separate Babylon JDK 26.
- CUDA and Metal backend projects are included by default in the Gradle project graph.
- GGUF and storage projects are included directly.
- Every model-family directory matching `aljabr-model-*` with a Gradle build file is dynamically included.

Project inclusion/configuration does not automatically add that project's JAR to every consumer runtime. However, the large catalogue increases Gradle configuration cost and increases the chance that unrelated project configuration breaks broad lifecycle tasks.

**Recommendation:** preserve direct inclusion for stable core APIs; introduce explicit opt-in settings properties for heavyweight/experimental backend, model-catalogue and storage groups where practical. Keep opt-in behavior deterministic and documented. Do not move projects behind flags until CI workflows and aggregate tasks are updated.

## 5. Build-only dependencies versus runtime dependencies

The root build adds JUnit Jupiter, JUnit Platform launcher and AssertJ to test configurations across subprojects. These are test/build dependencies, not production runtime dependencies.

Likewise:
- The Jandex dependency is currently a root buildscript dependency, not automatically a published runtime dependency.
- Java toolchains/compiler arguments and Gradle resolution strategies are build behavior, not Maven runtime dependencies.
- Settings inclusion is project configuration, not a runtime dependency declaration.

This distinction matters: fixing build-time coupling improves configuration and minimal builds, but does not by itself change a consumer's published runtime dependency graph.

## 6. Priority remediation plan

| Priority | Change | Risk | Acceptance evidence |
|---|---|---|---|
| P0 | Correct POM identity, repository URL and license; publication allowlist | High release-integrity risk if left wrong | Inspect generated POM for each supported artifact |
| P1 | Scope Vector API, preview and native-access flags to modules that need them | Medium; must inventory actual API usage | Numerics/tensor compile without incubator flag; CPU backend still compiles/tests with Vector API |
| P1 | Scope Jandex indexing to actual Jandex/Quarkus consumers | Low-medium | Plain module `classes` no longer runs Jandex; indexed modules still emit valid index |
| P1 | Scope Tafkir snapshot substitution to opt-in integration | Medium | Core build resolves without Tafkir checkout or snapshot artifact |
| P2 | Move shared versions to catalog/platform and remove broad fallback rules | Medium | Dependency reports show deterministic versions without silent global overrides |
| P2 | Make heavy project groups opt-in where beneficial | Medium-high | Core tasks configure/build without model/experimental groups; explicit aggregate workflows still work |
| P3 | Add minimal external consumer fixtures | Low | Clean consumer resolves numerics, linear algebra, tensor+CPU independently |

## 7. Required Gradle evidence before source changes

Run and archive the following in CI or a clean maintainer checkout:

1. `./gradlew projects` — actual project inclusion and configuration surface.
2. `./gradlew :core:aljabr-numerics:dependencies --configuration runtimeClasspath`
3. `./gradlew :core:aljabr-linear-algebra:dependencies --configuration runtimeClasspath`
4. `./gradlew :core:aljabr-tensor:dependencies --configuration runtimeClasspath`
5. `./gradlew :core:aljabr-core:dependencies --configuration runtimeClasspath`
6. `./gradlew :backend:cpu:aljabr-backend-cpu:dependencies --configuration runtimeClasspath`
7. `./gradlew :core:aljabr-numerics:outgoingVariants` and equivalent for each supported public artifact.
8. Generate and inspect `generatePomFileForMavenJavaPublication` output for each proposed publication.
9. Build clean consumer fixtures using only published artifacts, without composite substitution or Maven Local.

Use `dependencyInsight` for any unexpected transitive dependency. Do not claim a minimal published artifact graph until the clean consumer fixture passes.

## 8. Recommended next implementation slice

Start with the release-integrity and build-isolation changes that can be proven independently:

1. Correct shared POM metadata and add an explicit publication allowlist.
2. Restrict Jandex task wiring to actual consumers.
3. Scope Vector API and native-access flags after a source usage scan.
4. Scope Tafkir substitution to the opt-in integration.
5. Add a minimal numerics consumer fixture and verify the generated POM.

Do not create the BOM in this slice. The BOM should follow after artifact publication coordinates, publication policy and clean external resolution are reliable.

## 9. Related documents

- [Modular Distribution Architecture](MODULAR_DISTRIBUTION.md)
- [Computational Capability Inventory](COMPUTATIONAL_CAPABILITY_INVENTORY.md)
- [Computational Framework Scope](COMPUTATIONAL_FRAMEWORK_SCOPE.md)
