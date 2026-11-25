
plugins {
    `java-library`
    `java-test-fixtures`
}

dependencies {
    testFixturesApi(platform("org.junit:junit-bom:5.10.0"))
    testFixturesApi("org.junit.jupiter:junit-jupiter-api")
    api(project(":core:aljabr-diagnostics"))
    val aljabrCoreProject = findProject(":core:aljabr-core")
    if (aljabrCoreProject != null) {
        testImplementation(aljabrCoreProject)
    }

    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
}
