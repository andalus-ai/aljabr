plugins {
    java
}

dependencies {
    implementation(project(":engine:safetensor:aljabr-safetensor-api"))
    implementation(project(":engine:safetensor:aljabr-safetensor-core"))
    implementation(project(":engine:safetensor:aljabr-safetensor-bridge"))
    implementation(project(":engine:safetensor:aljabr-safetensor-loader"))
    implementation("io.quarkus:quarkus-core")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("jakarta.ws.rs:jakarta.ws.rs-api:3.1.0")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("org.jboss.logging:jboss-logging:3.5.3.Final")


    testImplementation("org.awaitility:awaitility:4.2.1")
    testImplementation("io.quarkus:quarkus-junit5")
    testImplementation("io.rest-assured:rest-assured:5.4.0")
}

sourceSets {
    test {
        java {
            // TurboQuant, AWQ, AutoRound, and BnB adapters moved to Tafkir integration
            exclude("tech/kayys/aljabr/safetensor/quantization/QuantizationRoundTripTest.java")
            // Quarkus REST integration test requires Quarkus application runtime
            exclude("tech/kayys/aljabr/safetensor/quantization/rest/QuantizationResourceTest.java")
        }
    }
}

tasks.test {
    useJUnitPlatform()
}
