plugins {
    `java-library`
    `maven-publish`
}
group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"
java { toolchain { languageVersion.set(JavaLanguageVersion.of(25)) } }
repositories { mavenCentral(); mavenLocal() }
dependencies {
    api(project(":spi:aljabr-spi-model"))
    implementation(project(":spi:aljabr-spi-inference"))
    implementation(project(":spi:aljabr-spi-tokenizer"))
    implementation(project(":models:aljabr-model-embedding-common"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation(group = "jakarta.enterprise", name = "jakarta.enterprise.cdi-api")
    implementation(group = "jakarta.inject", name = "jakarta.inject-api")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testRuntimeOnly(group = "org.junit.platform", name = "junit-platform-launcher", version = "1.10.2")
}
publishing {
    publications { create<MavenPublication>("mavenJava") { from(components["java"]) } }
    repositories { mavenLocal() }
}
tasks.withType<Test>().configureEach { useJUnitPlatform() }
