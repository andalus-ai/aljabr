rootProject.name = "aljabr-engine"

// ── Optional composite build: Tafkir (training/quantizer integration) ─────────
// Aljabr's foundational compute modules must remain buildable without Tafkir.
// Opt in to source-level quantizer integration with -PwithTafkir=true. This keeps
// an unrelated Tafkir configuration failure from breaking core Aljabr builds.
val withTafkir = providers.gradleProperty("withTafkir")
    .map { it.toBoolean() }
    .getOrElse(false)
val tafkirDir = file("../tafkir")
val tafkirSettingsExists = tafkirDir.exists() &&
    (tafkirDir.resolve("settings.gradle.kts").isFile ||
     tafkirDir.resolve("settings.gradle").isFile)
if (withTafkir && !tafkirSettingsExists) {
    throw GradleException("withTafkir=true requires a sibling ../tafkir checkout with Gradle settings")
}
if (withTafkir) {
    includeBuild("../tafkir")
}
// ─────────────────────────────────────────────────────────────────────────────

fun includeOptionalProject(projectPath: String, vararg candidatePaths: String) {
    val projectDir = candidatePaths
        .map { file(it) }
        .firstOrNull { candidate ->
            candidate.resolve("build.gradle.kts").isFile || candidate.resolve("build.gradle").isFile
        }
        ?: return

    include(projectPath)
    project(":$projectPath").projectDir = projectDir
}

includeOptionalProject("core:aljabr-core", "core/aljabr-core")

// Autograd is training-only; exclude from foundational builds
val skipAutograd = true
if (!skipAutograd) {
    include("core:autograd")
}

include("core:aljabr-tensor")
include("core:aljabr-core")
include("core:aljabr-error-code")
include("core:aljabr-nn")

include("core:aljabr-spi-model")
// New SPI modules — formalises Aljabr as the inference/serving framework layer
// These will host contracts elevated from Aqli (Phase 2 source migration)
include("core:aljabr-spi-inference")
include("core:aljabr-spi-plugin")
include("core:aljabr-spi-tokenizer")
include("core:aljabr-3d")

//include("backend:blackwell:aljabr-kernel-blackwell")

include("backend:cpu:aljabr-backend-cpu")
include("backend:cuda:aljabr-backend-cuda")
include("backend:cuda:aljabr-kernel-cuda")
//include("backend:cuda:aljabr-plugin-kernel-cuda")
//include("backend:directml:aljabr-plugin-kernel-directml")
// HAT requires a separate Babylon JDK 26 distribution with incubator APIs.
// Keep this experimental backend out of default builds unless explicitly enabled.
val withHat = providers.gradleProperty("withHat")
    .map { it.toBoolean() }
    .getOrElse(false)
val babylonJdkHome = providers.gradleProperty("babylonJdkPath")
    .orElse(providers.environmentVariable("BABYLON_JDK_HOME"))
    .orNull
if (withHat) {
    if (babylonJdkHome.isNullOrBlank()) {
        throw GradleException(
            "withHat=true requires -PbabylonJdkPath=/path/to/babylon-jdk-26 " +
                "or BABYLON_JDK_HOME pointing to a Babylon JDK 26 installation"
        )
    }
    val babylonJdk = file(babylonJdkHome)
    if (!babylonJdk.resolve("bin/javac").isFile || !babylonJdk.resolve("bin/java").isFile) {
        throw GradleException("Babylon JDK is missing bin/javac or bin/java: $babylonJdk")
    }
    include("backend:hat:aljabr-backend-hat")
}
include("backend:metal:aljabr-backend-metal")
//include("backend:metal:aljabr-mlx-binding")

//include("backend:rocm:aljabr-kernel-rocm")
//include("backend:rocm:aljabr-plugin-kernel-rocm")

// Dynamically include model family projects under models/
file("models")
    .listFiles { candidate ->
        candidate.isDirectory &&
                candidate.name.startsWith("aljabr-model-") &&
                (candidate.resolve("build.gradle.kts").isFile || candidate.resolve("build.gradle").isFile)
    }
    ?.sortedBy { it.name }
    ?.forEach { modelProject ->
        include("models:${modelProject.name}")
        project(":models:${modelProject.name}").projectDir = modelProject
    }

// GGUF Suite: aljabr-gguf-api, aljabr-gguf-llamacpp, aljabr-gguf-core, aljabr-gguf-java
include("core:aljabr-gguf-api")
include("core:aljabr-gguf-llamacpp")
include("core:aljabr-gguf-core")
include("core:aljabr-gguf-java")

include("core:aljabr-rocksdb")
include("core:aljabr-helixdb")

// ── Modules with external dependencies (tafkir / SPI) ─────────────────────
// These modules require tafkir quantizer artifacts (tech.kayys.tafkir:*) which
// are resolved via composite build (../tafkir) when available, or from a local
// Maven repository. Use includeOptionalProject so they are silently skipped on
// checkouts where the module directories are absent.
includeOptionalProject("core:aljabr-safetensor-api",          "core/aljabr-safetensor-api")
includeOptionalProject("core:aljabr-safetensor-spi",          "core/aljabr-safetensor-spi")
includeOptionalProject("core:aljabr-safetensor-core",         "core/aljabr-safetensor-core")
includeOptionalProject("core:aljabr-safetensor-loader",       "core/aljabr-safetensor-loader")
// This adapter depends on Tafkir quantizer artifacts and is intentionally opt-in.
// Core Aljabr builds do not need training/quantization modules.
if (withTafkir) {
    includeOptionalProject("core:aljabr-safetensor-quantization", "core/aljabr-safetensor-quantization")
}
includeOptionalProject("core:aljabr-gguf-bridge",             "core/aljabr-gguf-bridge")
includeOptionalProject("core:aljabr-gguf-fast-bridge",        "core/aljabr-gguf-fast-bridge")
includeOptionalProject("core:aljabr-gguf-converter",          "core/aljabr-gguf-converter")
includeOptionalProject("core:aljabr-gguf-converter-java",     "core/aljabr-gguf-converter-java")
