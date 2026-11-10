plugins {
    java
}

dependencies {
    // MLX binding: bridges Apple MLX tensors into the Aljabr ComputeBackend contract
    implementation(project(":plugins:ml:aljabr-tensor"))
    implementation(project(":engine:safetensor:aljabr-safetensor-core"))
    implementation("io.quarkus:quarkus-arc")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}

