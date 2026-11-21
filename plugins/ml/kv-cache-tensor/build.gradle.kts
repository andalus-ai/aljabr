plugins {
    `java-library`
}

group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"

java { withSourcesJar() }

dependencies {
    api(project(":core:aljabr-core"))
}
