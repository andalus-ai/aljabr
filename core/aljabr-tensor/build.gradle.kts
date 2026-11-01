
plugins {
    java
}

dependencies {
    val aljabrCoreProject = findProject(":core:aljabr-core")
    if (aljabrCoreProject != null) {
        testImplementation(aljabrCoreProject)
    }

    testRuntimeOnly(project(":backend:cpu:aljabr-backend-cpu"))
}
