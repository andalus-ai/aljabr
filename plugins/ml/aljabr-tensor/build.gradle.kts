
plugins {
    java
}

dependencies {
    val aljabrCoreProject = findProject(":core:aljabr-core")
    if (aljabrCoreProject != null) {
        testImplementation(aljabrCoreProject)
    }

    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
}
