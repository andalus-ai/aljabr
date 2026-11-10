plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"

java {
    toolchain { languageVersion.set(JavaLanguageVersion.of(25)) }
}

repositories {
    mavenCentral()
    mavenLocal()
}

dependencies {
    api(project(":spi:aljabr-spi-plugin"))
    implementation(project(":plugins:ml:aljabr-tensor"))
    implementation(project(":backend:rocm:aljabr-kernel-rocm"))
    compileOnly(group = "org.jboss.logging", name = "jboss-logging")
}

publishing {
    publications { create<MavenPublication>("mavenJava") { from(components["java"]) } }
    repositories { mavenLocal() }
}
