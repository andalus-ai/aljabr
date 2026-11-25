plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
        withSourcesJar()
        withJavadocJar()
    }
}

dependencies {
    api(project(":models:diffusion:aljabr-diffusion-api"))
    implementation(project(":engine:safetensor:aljabr-safetensor-loader"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.10.2")
}

publishing {
    publications { create<MavenPublication>("mavenJava") { from(components["java"]) } }
    repositories { mavenLocal() }
}

tasks.withType<Test>().configureEach { useJUnitPlatform() }
