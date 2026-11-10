rootProject.name = "aljabr"

fun includeProjectsUnder(rootPath: String, prefix: String = rootPath.replace('/', ':')) {
    val rootDir = file(rootPath)
    if (!rootDir.isDirectory) return
    rootDir.walkTopDown()
        .filter { it.isDirectory && it != rootDir }
        .filter { it.resolve("build.gradle.kts").isFile || it.resolve("build.gradle").isFile }
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
includeProjectsUnder("plugins")

// Hardware providers remain separate from the portable framework modules.
include("backend:cpu:aljabr-backend-cpu")
include("backend:cuda:aljabr-backend-cuda")
include("backend:cuda:aljabr-kernel-cuda")
include("backend:metal:aljabr-backend-metal")

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
