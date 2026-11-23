plugins {
    `java-library`
}

dependencies {
    api(project(":core:aljabr-core"))
    api(project(":plugins:ml:aljabr-tensor"))
    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
}
