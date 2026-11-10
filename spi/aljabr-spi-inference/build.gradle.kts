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

dependencies {
    // Inference contracts depend on tensor primitives and model metadata
    api(project(":core:aljabr-tensor"))
    api(project(":core:aljabr-spi-model"))
    api(project(":core:aljabr-error-code"))

    implementation("io.smallrye.reactive:mutiny:2.5.5")
    api("org.reactivestreams:reactive-streams:1.0.4")
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("jakarta.validation:jakarta.validation-api:3.0.2")
    compileOnly("org.jetbrains:annotations:24.1.0")

    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
