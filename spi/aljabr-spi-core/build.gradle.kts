plugins {
    `java-library`
    `maven-publish`
}
java {
    toolchain { languageVersion.set(JavaLanguageVersion.of(25)) }
}
dependencies {
    api(project(":core:aljabr-diagnostics"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}
tasks.withType<Test>().configureEach { useJUnitPlatform() }
