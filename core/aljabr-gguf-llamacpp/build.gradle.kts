plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

repositories {
    mavenCentral()
    mavenLocal()
}

dependencies {
    // Aljabr modules
    api(project(":core:aljabr-gguf-api"))
    api(project(":core:aljabr-gguf-core"))
    api(project(":core:aljabr-spi-inference"))
    implementation(project(":core:aljabr-spi-model"))

    // Logging
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")

    // Reactive (for LlamaCppRunner streaming)
    implementation("io.smallrye.reactive:mutiny:2.5.5")

    // Metrics
    implementation("io.micrometer:micrometer-core:1.14.4")

    // Chat template
    implementation("com.hubspot.jinjava:jinjava:2.7.3")

    // SmallRye Config (for LlamaCppProviderConfig @ConfigMapping interface)
    compileOnly("io.smallrye.config:smallrye-config:3.8.3")

    // Jakarta CDI API (annotations only, no container needed)
    compileOnly("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    compileOnly("jakarta.inject:jakarta.inject-api:2.0.1")


    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testRuntimeOnly(group = "org.junit.platform", name = "junit-platform-launcher")
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
        }
    }
    repositories {
        mavenLocal()
    }
}

tasks.jar {
    manifest {
        attributes(
            mapOf(
                "Plugin-Id" to "gguf-runner",
                "Plugin-Type" to "runner",
                "Plugin-Provider" to "tech.kayys.aljabr.gguf.llamacppbackend.LlamaCppGgufBackendProvider",
                "Plugin-Version" to "0.1.0-SNAPSHOT",
                "Supported-Formats" to ".gguf"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}




// Recreate the jextract-generated header class that the checked-in struct
// wrappers reference. This is intentionally opt-in: a normal Java build must
// not silently download or build a native llama.cpp checkout.
val jextractExecutable = providers.gradleProperty("jextractExecutable")
    .orElse(providers.environmentVariable("JEXTRACT"))
    .orElse("jextract")
val llamaCppInclude = providers.gradleProperty("llamaCppInclude")
    .orElse(providers.environmentVariable("LLAMA_CPP_INCLUDE"))
val ggmlInclude = providers.gradleProperty("ggmlInclude")
    .orElse(providers.environmentVariable("GGML_INCLUDE"))

val generateLlamaFfmBindings = tasks.register<Exec>("generateLlamaFfmBindings") {
    group = "code generation"
    description = "Regenerate llama_h.java with jextract from a compatible llama.cpp header checkout."

    val generatedRoot = layout.buildDirectory.dir("generated/llama-jextract").get().asFile
    doFirst {
        val includeDirValue = llamaCppInclude.orNull
            ?: throw GradleException(
                "Missing llama.cpp include directory. Set -PllamaCppInclude=/path/to/llama.cpp/include " +
                    "or LLAMA_CPP_INCLUDE."
            )
        val includeDir = file(includeDirValue).absoluteFile.normalize()
        val header = includeDir.resolve("llama.h")
        if (!header.isFile) {
            throw GradleException("Expected llama.h at $header")
        }

        generatedRoot.deleteRecursively()
        generatedRoot.mkdirs()

        val command = mutableListOf(
            jextractExecutable.get(),
            "--output", generatedRoot.absolutePath,
            "--target-package", "tech.kayys.aljabr.llama",
            "--header-class-name", "llama_h",
            // glibc exposes fopencookie through stdio.h, but jextract cannot
            // model its internal _IO_cookie_io_functions_t on the CI toolchain.
            // It is unrelated to llama.cpp and is not part of this module's API.
            "--exclude-symbols", "^fopencookie$",
            "-I", includeDir.absolutePath
        )
        val ggmlDir = ggmlInclude.orNull?.let { file(it).absoluteFile.normalize() }
            ?: includeDir.resolve("../ggml/include").takeIf { it.isDirectory }
        if (ggmlDir != null && ggmlDir.isDirectory) {
            command.addAll(listOf("-I", ggmlDir.absolutePath))
        }
        command.add(header.absolutePath)
        commandLine(command)
    }

    doLast {
        val generatedPackage = generatedRoot.resolve("tech/kayys/aljabr/llama")
        val generatedHeader = generatedPackage.resolve("llama_h.java")
        if (!generatedHeader.isFile) {
            throw GradleException(
                "jextract completed but did not produce $generatedHeader. " +
                    "Check the jextract version and its --output layout."
            )
        }

        // jextract may split a large header binding across llama_h.java and
        // numbered or shared helper classes (for example llama_h_1.java and
        // llama_h$shared.java). Copy that generated class family together,
        // but never copy generated struct/callback wrappers over the maintained
        // wrappers in src/main/java (their explicit layouts are required by callers).
        val generatedHeaderFiles = generatedPackage.listFiles()
            ?.filter { it.isFile && it.name.matches(Regex("llama_h(?:[_$].*)?\\.java")) }
            .orEmpty()
        if (generatedHeaderFiles.none { it.name == "llama_h.java" }) {
            throw GradleException("jextract output is missing llama_h.java under $generatedPackage")
        }

        val destination = file("src/main/java/tech/kayys/aljabr/llama")
        destination.mkdirs()
        val generatedNames = generatedHeaderFiles.map { it.name }.toSet()

        // Remove only stale split-header helpers from earlier generations.
        destination.listFiles()
            ?.filter {
                it.isFile &&
                    it.name.matches(Regex("llama_h(?:[_$].*)?\\.java")) &&
                    it.name !in generatedNames
            }
            ?.forEach { stale ->
                if (!stale.delete()) {
                    throw GradleException("Could not remove stale generated binding $stale")
                }
            }

        generatedHeaderFiles.forEach { generatedFile ->
            generatedFile.copyTo(destination.resolve(generatedFile.name), overwrite = true)
        }

        // llama.h references the GGUF init-params struct, which is emitted as a
        // separate jextract wrapper rather than part of the split header family.
        // Keep the maintained wrapper when one exists; otherwise install the generated
        // definition so the generated downcall descriptors can compile.
        val generatedGgufInitParams = generatedPackage.resolve("gguf_init_params.java")
        if (!generatedGgufInitParams.isFile) {
            throw GradleException(
                "jextract output is missing gguf_init_params.java under $generatedPackage"
            )
        }
        val ggufInitParamsWrapper = destination.resolve("gguf_init_params.java")
        if (!ggufInitParamsWrapper.exists()) {
            generatedGgufInitParams.copyTo(ggufInitParamsWrapper)
        }

        // The generated header can contain nested opaque declarations with the same
        // names as maintained top-level wrappers. Fully qualify the two concrete
        // layouts needed by this header so Java does not resolve the opaque nested type.
        val generatedHeaderText = generatedHeader.readText()
            .replace(
                "ggml_backend_graph_copy.layout()",
                "tech.kayys.aljabr.llama.ggml_backend_graph_copy.layout()"
            )
            .replace(
                "gguf_init_params.layout()",
                "tech.kayys.aljabr.llama.gguf_init_params.layout()"
            )
        generatedHeader.writeText(generatedHeaderText)

        // Guard the exact source-level contract that failed in CI. This catches
        // accidental changes to the maintained wrapper before Gradle reaches javac.
        val graphCopyWrapper = destination.resolve("ggml_backend_graph_copy.java")
        if (!graphCopyWrapper.isFile ||
            !graphCopyWrapper.readText().contains("layout()")) {
            throw GradleException(
                "Expected maintained ggml_backend_graph_copy.java with layout() at $graphCopyWrapper"
            )
        }
        logger.lifecycle(
            "Refreshed llama_h.java; preserved checked-in FFM wrapper package at " +
                destination.relativeTo(projectDir)
        )
    }
}


// Only CI/native-integration builds opt into regeneration. Default builds stay offline
// and do not require a jextract installation or a llama.cpp checkout.
if (providers.gradleProperty("withLlamaFfmBindings").orNull?.toBoolean() == true) {
    tasks.named("compileJava") {
        dependsOn(generateLlamaFfmBindings)
    }
}
