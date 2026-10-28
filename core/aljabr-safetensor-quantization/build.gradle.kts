plugins {
    java
}

// Detect whether tafkir is available as a composite build (../tafkir).
// When it is, Gradle automatically substitutes the Maven coordinates below
// with source-project dependencies — no SNAPSHOT binary resolution needed.
// When it is not (e.g., binary-only downstream consumers), the coordinates
// are resolved normally from whatever repositories the consumer configures.
val tafkirAvailable = file("../../../tafkir").let {
    it.isDirectory &&
    (it.resolve("settings.gradle.kts").isFile || it.resolve("settings.gradle").isFile)
}

dependencies {
    implementation(project(":core:aljabr-safetensor-api"))
    implementation(project(":core:aljabr-safetensor-core"))
    implementation(project(":core:aljabr-safetensor-bridge"))
    implementation(project(":core:aljabr-safetensor-loader"))
    implementation("io.quarkus:quarkus-core")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("jakarta.ws.rs:jakarta.ws.rs-api:3.1.0")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("org.jboss.logging:jboss-logging:3.5.3.Final")

    // Tafkir quantizer modules.
    // When tafkir is included as a composite build (see settings.gradle.kts),
    // Gradle substitutes these coordinates with the corresponding source projects.
    // When running standalone (binary release), consumers must publish tafkir
    // to a local/remote Maven repository first.
    implementation("tech.kayys.tafkir:tafkir-quantizer-gptq:0.1.0-SNAPSHOT")
    implementation("tech.kayys.tafkir:tafkir-quantizer-awq:0.1.0-SNAPSHOT")
    implementation("tech.kayys.tafkir:tafkir-quantizer-autoround:0.1.0-SNAPSHOT")
    implementation("tech.kayys.tafkir:tafkir-quantizer-turboquant:0.1.0-SNAPSHOT")

    testImplementation("org.awaitility:awaitility:4.2.1")
    testImplementation("io.quarkus:quarkus-junit5")
    testImplementation("io.rest-assured:rest-assured:5.4.0")
}

tasks.test {
    useJUnitPlatform()
}
