plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"

java { toolchain { languageVersion.set(JavaLanguageVersion.of(25)) } }

dependencies {
    api(project(":models:aljabr-model-embedding-gemma2"))
    api(project(":engine:safetensor:aljabr-safetensor-loader"))
    implementation(project(":spi:aljabr-spi-tokenizer"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("jakarta.inject:jakarta.inject-api:2.0.1")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api")
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("io.micrometer:micrometer-core:1.16.3")
    testImplementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.10.2")
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") { from(components["java"]) }
    }
    repositories { mavenLocal() }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    // The opt-in official checkpoint test materializes large BF16 tensors as Java arrays.
    // Give only that test invocation a larger worker heap; ordinary module tests keep
    // Gradle's default test-worker memory limit.
    if (providers.environmentVariable("ALJABR_EMBEDDING_GEMMA2_CHECKPOINT").isPresent) {
        maxHeapSize = "4g"
    }
}
