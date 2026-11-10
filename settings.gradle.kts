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

fun includeDirectProjectsUnder(rootPath: String, prefix: String = rootPath.replace('/', ':')) {
    val rootDir = file(rootPath)
    if (!rootDir.isDirectory) return
    rootDir.listFiles()
        ?.filter { it.isDirectory && (it.resolve("build.gradle.kts").isFile || it.resolve("build.gradle").isFile) }
        ?.sortedBy { it.name }
        ?.forEach { dir ->
            val path = ":$prefix:${dir.name}"
            include(path)
            project(path).projectDir = dir
        }
}

// Foundational contracts and runtime.
includeDirectProjectsUnder("core")
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

// Discover every regular hardware provider and kernel plugin. Experimental HAT
// remains opt-in because it requires a separate Babylon JDK installation.
includeProjectsUnder("backend", excludedRelativePaths = setOf("hat"))

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
