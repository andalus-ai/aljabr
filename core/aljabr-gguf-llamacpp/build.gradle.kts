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


