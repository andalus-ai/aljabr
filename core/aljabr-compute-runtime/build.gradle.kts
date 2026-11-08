plugins {
    `java-library`
    `maven-publish`
}

dependencies {
    api(project(":core:aljabr-spi-compute"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testRuntimeOnly(project(":backend:cpu:aljabr-backend-cpu"))
}

tasks.withType<Test>().configureEach { useJUnitPlatform() }
