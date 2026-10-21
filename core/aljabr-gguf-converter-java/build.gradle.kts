plugins {
    java
}

dependencies {
    implementation(project(":core:aljabr-gguf-converter"))
    implementation(project(":core:aljabr-gguf-core"))
    implementation(project(":core:aljabr-core"))
    implementation(project(":core:aljabr-safetensor-loader"))
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("com.google.code.gson:gson:2.11.0")
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api")
    implementation("jakarta.inject:jakarta.inject-api")
    implementation("org.slf4j:slf4j-api:2.0.13")
}
