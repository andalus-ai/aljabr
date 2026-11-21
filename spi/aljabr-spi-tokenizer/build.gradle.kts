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
    // Tokenizer SPI: depends on tensor types for encoding offsets/ids
    api(project(":plugins:ml:aljabr-tensor"))
    api(project(":spi:aljabr-spi-model"))

    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")

    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()

    // Forward opt-in parity inputs from Gradle's -D properties into the forked
    // JUnit worker JVM. Without this, @EnabledIfSystemProperty cannot see them.
    listOf(
        "aljabr.embeddinggemma2.tokenizer",
        "aljabr.embeddinggemma2.tokenizerCorpus"
    ).forEach { propertyName ->
        System.getProperty(propertyName)?.let { propertyValue ->
            systemProperty(propertyName, propertyValue)
        }
    }
}
