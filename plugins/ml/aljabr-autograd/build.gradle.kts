plugins {
    java
}

dependencies {
    api(project(":plugins:ml:aljabr-tensor"))
    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
}
