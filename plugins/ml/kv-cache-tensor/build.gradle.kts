plugins {
    `java-library`
}

group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"

java { withSourcesJar() }

dependencies {
    api(project(":core:aljabr-core"))
    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test { useJUnitPlatform() }
