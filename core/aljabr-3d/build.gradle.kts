plugins {
    java
}

dependencies {
    implementation(project(":core:aljabr-tensor"))
    implementation(project(":core:aljabr-spi-model"))

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
}
