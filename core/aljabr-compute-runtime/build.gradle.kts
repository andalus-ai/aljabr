plugins {
    `java-library`
    `maven-publish`
}

dependencies {
    api(project(":core:aljabr-spi-compute"))
    testRuntimeOnly(project(":backend:cpu:aljabr-backend-cpu"))
}

tasks.withType<Test>().configureEach { useJUnitPlatform() }
