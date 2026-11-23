plugins {
    `java-library`
}

group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"

java { withSourcesJar() }

dependencies {
    api(project(":spi:aljabr-spi-model"))
    testImplementation(project(":plugins:ml:aljabr-tensor"))
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test { useJUnitPlatform() }
