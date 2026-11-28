plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"

java { toolchain { languageVersion.set(JavaLanguageVersion.of(25)) } }

repositories { mavenCentral(); mavenLocal() }

dependencies {
    implementation(project(":spi:aljabr-spi-model"))
    implementation(project(":spi:aljabr-spi-tokenizer"))
    // Reuse Whisper-compatible PCM and log-Mel DSP; no Qwen-specific DSP fork.
    implementation(project(":models:aljabr-model-whisper"))
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api")
    implementation("jakarta.inject:jakarta.inject-api")
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

publishing {
    publications { create<MavenPublication>("mavenJava") { from(components["java"]) } }
    repositories { mavenLocal() }
}

tasks.withType<Test>().configureEach { useJUnitPlatform() }
