plugins {
    `java-library`
    `maven-publish`
}

dependencies {
    api(project(":plugins:ml:aljabr-tensor"))
}
