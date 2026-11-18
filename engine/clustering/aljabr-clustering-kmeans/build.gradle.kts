plugins {
    `java-library`
}

dependencies {
    api(project(":engine:clustering:aljabr-clustering-api"))
    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
}
