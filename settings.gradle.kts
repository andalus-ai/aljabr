rootProject.name = "aljabr"

fun includeProjectsUnder(
    rootPath: String,
    prefix: String = rootPath.replace('/', ':'),
    excludedRelativePaths: Set<String> = emptySet()
) {
    val rootDir = file(rootPath)
    if (!rootDir.isDirectory) return
    rootDir.walkTopDown()
        .filter { it.isDirectory && it != rootDir }
        .filter { it.resolve("build.gradle.kts").isFile || it.resolve("build.gradle").isFile }
        .filter {
            val relative = it.relativeTo(rootDir).invariantSeparatorsPath
            excludedRelativePaths.none { excluded ->
                relative == excluded || relative.startsWith("$excluded/")
            }
        }
        .sortedBy { it.relativeTo(rootDir).invariantSeparatorsPath }
        .forEach { dir ->
            val relative = dir.relativeTo(rootDir).invariantSeparatorsPath
            val path = ":$prefix:${relative.replace('/', ':')}"
            include(path)
            project(path).projectDir = dir
        }
}

fun includeDirectProjectsUnder(
    rootPath: String,
    prefix: String = rootPath.replace('/', ':'),
    excludedRelativePaths: Set<String> = emptySet()
) {
    val rootDir = file(rootPath)
    if (!rootDir.isDirectory) return
    rootDir.listFiles()
        ?.filter { it.isDirectory && (it.resolve("build.gradle.kts").isFile || it.resolve("build.gradle").isFile) }
        ?.filter { it.name !in excludedRelativePaths }
        ?.sortedBy { it.name }
        ?.forEach { dir ->
            val path = ":$prefix:${dir.name}"
            include(path)
            project(path).projectDir = dir
        }
}

// Foundational contracts and runtime.
// The plugin-core implementation still targets a removed node-plugin API and
// an older plugin metadata contract. Keep it available for its ongoing port,
// but do not make the legacy implementation part of the default core build.
// Use -PwithLegacyPluginCore=true to include it explicitly.
val withLegacyPluginCore = providers.gradleProperty("withLegacyPluginCore")
    .map(String::toBoolean)
    .getOrElse(false)
includeDirectProjectsUnder(
    "core",
    excludedRelativePaths = if (withLegacyPluginCore) emptySet() else setOf("aljabr-plugin-core")
)
includeDirectProjectsUnder("spi")

// Format engines and independently installable scientific/ML extensions.
includeProjectsUnder("engine")

// The legacy autograd integration still depends on IR and trainer API snapshot
// artifacts that are not published or included as projects in this repository.
// Keep it out of the default build until those dependencies are brought in-tree
// or published; opt in explicitly when working on that integration.
val withAutograd = providers.gradleProperty("withAutograd")
    .map(String::toBoolean)
    .getOrElse(false)
includeProjectsUnder(
    "plugins",
    excludedRelativePaths = if (withAutograd) emptySet() else setOf("ml/autograd")
)

// The Blackwell runner still targets the pre-SPI runner/extension contracts and
// cannot compile against the current module graph. Keep its sources available
// for the ongoing port, but exclude it from the default build until that port is
// complete. Use -PwithBlackwell=true to include it explicitly while migrating.
val withBlackwell = providers.gradleProperty("withBlackwell")
    .map(String::toBoolean)
    .getOrElse(false)

// Discover regular hardware providers and kernel plugins. Experimental HAT
// remains opt-in because it requires a separate Babylon JDK installation.
val excludedBackendPaths = buildSet {
    add("hat")
    if (!withBlackwell) add("blackwell")
}
includeProjectsUnder("backend", excludedRelativePaths = excludedBackendPaths)

// Optional experimental Babylon HAT backend.
val withHat = providers.gradleProperty("withHat").map(String::toBoolean).getOrElse(false)
val babylonJdkHome = providers.gradleProperty("babylonJdkPath")
    .orElse(providers.environmentVariable("BABYLON_JDK_HOME")).orNull
if (withHat) {
    if (babylonJdkHome.isNullOrBlank()) {
        throw GradleException("withHat=true requires -PbabylonJdkPath or BABYLON_JDK_HOME")
    }
    val babylonJdk = file(babylonJdkHome)
    if (!babylonJdk.resolve("bin/javac").isFile || !babylonJdk.resolve("bin/java").isFile) {
        throw GradleException("Babylon JDK is missing bin/javac or bin/java: $babylonJdk")
    }
    include("backend:hat:aljabr-backend-hat")
}

// Model family modules remain discoverable without coupling them to core.
file("models").listFiles { dir ->
    dir.isDirectory && dir.name.startsWith("aljabr-model-") &&
        (dir.resolve("build.gradle.kts").isFile || dir.resolve("build.gradle").isFile)
}?.sortedBy { it.name }?.forEach { dir ->
    val path = ":models:${dir.name}"
    include(path)
    project(path).projectDir = dir
}
