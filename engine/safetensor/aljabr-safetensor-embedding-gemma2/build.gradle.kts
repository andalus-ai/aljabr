plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"

java { toolchain { languageVersion.set(JavaLanguageVersion.of(25)) } }

dependencies {
    api(project(":models:aljabr-model-embedding-gemma2"))
    implementation(project(":engine:safetensor:aljabr-safetensor-loader"))
    implementation("jakarta.inject:jakarta.inject-api:2.0.1")
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.10.2")
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") { from(components["java"]) }
    }
    repositories { mavenLocal() }
}

tasks.withType<Test>().configureEach { useJUnitPlatform() }
