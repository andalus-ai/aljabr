plugins {
    `java-library`
}

dependencies {
    implementation(project(":core:aljabr-tensor"))
    api(project(":core:aljabr-spi-compute"))
    testImplementation(project(":core:aljabr-spi-testkit"))
}