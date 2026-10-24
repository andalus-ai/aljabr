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

        // Always refresh the header class, but preserve existing checked-in
        // struct/callback wrappers. Some C types are opaque to jextract in this
        // invocation even though Aljabr's wrappers intentionally provide layouts
        // for them; overwriting those wrappers breaks llama_h's layout calls.
        // Copy newly emitted helper classes only when no checked-in counterpart
        // exists, so generated output cannot silently replace the maintained API.
        val destination = file("src/main/java/tech/kayys/aljabr/llama")
        val generatedFiles = generatedPackage.walkTopDown()
            .filter { it.isFile && it.extension == "java" }
            .toList()
        if (generatedFiles.none { it.name == "llama_h.java" }) {
            throw GradleException("No llama_h.java was generated under $generatedPackage")
        }

        var copied = 0
        var preserved = 0
        generatedFiles.forEach { source ->
            val target = destination.resolve(source.relativeTo(generatedPackage).path)
            val isHeaderBinding = source.name == "llama_h.java"
            if (isHeaderBinding || !target.exists()) {
                target.parentFile.mkdirs()
                source.copyTo(target, overwrite = isHeaderBinding)
                copied++
            } else {
                preserved++
                logger.info("Preserving checked-in FFM wrapper: ${target.relativeTo(projectDir)}")
            }
        }
        logger.lifecycle(
            "Refreshed llama_h.java and copied $copied generated Java bindings; " +
                "preserved $preserved existing wrappers in " + destination.relativeTo(projectDir)
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
