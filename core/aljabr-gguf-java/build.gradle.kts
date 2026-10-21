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
    implementation(project(":core:aljabr-gguf-api"))
    implementation(project(":core:aljabr-gguf-core"))
    implementation(project(":core:aljabr-spi-model"))
    implementation(project(":core:aljabr-spi-tokenizer"))
    implementation(project(":core:aljabr-spi-inference"))

    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testRuntimeOnly(group = "org.junit.platform", name = "junit-platform-launcher")
    testRuntimeOnly(project(":backend:metal:aljabr-backend-metal"))
    testRuntimeOnly(project(":models:aljabr-model-gemma4"))
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
                "Plugin-Provider" to "tech.kayys.aljabr.gguf.javabackend.JavaNativeGgufBackendProvider",
                "Plugin-Version" to "0.1.0-SNAPSHOT",
                "Supported-Formats" to ".gguf"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    maxHeapSize = "16g"
    jvmArgs("--add-modules", "jdk.incubator.vector")
}
