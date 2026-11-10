plugins {
    java
}

dependencies {
    implementation(project(":plugins:ml:aljabr-tensor"))
    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
}
