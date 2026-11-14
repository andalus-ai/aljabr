plugins {
    `java-library`
}

dependencies {
    api(project(":plugins:ml:aljabr-nn"))
    api(project(":plugins:ml:aljabr-tensor"))
    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
}
