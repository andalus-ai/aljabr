plugins {
    `java-library`
}

dependencies {
    api(project(":core:aljabr-core"))
    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
}
