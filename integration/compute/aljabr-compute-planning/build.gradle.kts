plugins {
    `java-library`
}

dependencies {
    api(project(":spi:aljabr-spi-compute"))
    api(project(":plugins:ml:aljabr-tensor"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test>().configureEach { useJUnitPlatform() }
