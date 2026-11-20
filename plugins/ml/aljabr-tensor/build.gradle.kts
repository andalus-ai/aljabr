
plugins {
    java
}

dependencies {
    api(project(":core:aljabr-diagnostics"))
    val aljabrCoreProject = findProject(":core:aljabr-core")
    if (aljabrCoreProject != null) {
        testImplementation(aljabrCoreProject)
    }

    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
}
