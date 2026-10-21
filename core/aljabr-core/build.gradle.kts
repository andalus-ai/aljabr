plugins {
    `java-library`
}

// Artifact coordinates for publishing
group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"


java {
    withSourcesJar()
}

dependencies {
    // Core building blocks exposed as stable API for consumers (e.g., aljabr)
    api(project(":core:aljabr-tensor"))
    // Removed orphaned dependencies that were causing build failures

    // SPIs that consumers may need to implement or call
    api(project(":core:aljabr-spi-model"))


    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
