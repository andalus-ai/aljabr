plugins {
    `java-library`
    `maven-publish`
}

java {
    toolchain { languageVersion.set(JavaLanguageVersion.of(25)) }
    withSourcesJar()
    withJavadocJar()
}

dependencies {
    api(project(":spi:aljabr-spi-audio"))
}

tasks.withType<Test>().configureEach { useJUnitPlatform() }
