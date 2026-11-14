plugins {
    `java-library`
}

dependencies {
    api(project(":plugins:ml:aljabr-nn"))
    api(project(":plugins:ml:aljabr-tensor"))
    api(project(":plugins:ml:aljabr-tensor"))
    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
}

tasks.withType<org.gradle.api.tasks.testing.Test>().configureEach {
    jvmArgs("--add-modules=jdk.incubator.vector")
}
