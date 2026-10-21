plugins {
    java
}

dependencies {
    // IR types (GValueId, GOp, GGraph, GContext, etc.) used by all Grad implementations
    implementation("tech.kayys.aljabr:aljabr-ir:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-trainer-api:0.1.0-SNAPSHOT")
    implementation("org.slf4j:slf4j-api:2.0.12")

    // Core tensor types
    implementation(project(":core:aljabr-tensor"))
    implementation(project(":core:aljabr-core"))
}
