plugins {
    `java-library`
}

dependencies {
    api(project(":plugins:ml:aljabr-nn"))
    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
}
