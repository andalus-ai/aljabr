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

tasks.register<Exec>("generateLlamaFfmBindings") {
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
        val generatedHeader = generatedRoot.resolve("tech/kayys/aljabr/llama/llama_h.java")
        if (!generatedHeader.isFile) {
            throw GradleException(
                "jextract completed but did not produce $generatedHeader. " +
                    "Check the jextract version and its --output layout."
            )
        }
        val destination = file("src/main/java/tech/kayys/aljabr/llama/llama_h.java")
        destination.parentFile.mkdirs()
        generatedHeader.copyTo(destination, overwrite = true)
        logger.lifecycle("Updated ${destination.relativeTo(projectDir)}")
    }
}
