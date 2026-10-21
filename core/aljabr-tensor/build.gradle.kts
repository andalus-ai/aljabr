plugins {
    java
}

dependencies {
    val aljabrCoreProject = findProject(":core:aljabr-core")
    if (aljabrCoreProject != null) {
        testImplementation(aljabrCoreProject)
    }
}
