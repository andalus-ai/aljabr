plugins {
    java
}

dependencies {
    implementation(project(":engine:safetensor:aljabr-safetensor-core"))
    implementation(project(":engine:safetensor:aljabr-safetensor-bridge"))
    implementation(project(":engine:safetensor:aljabr-safetensor-loader"))
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.annotation:jakarta.annotation-api:3.0.0")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("org.jboss.logging:jboss-logging:3.5.3.Final")

    testImplementation("org.awaitility:awaitility:4.2.1")
}

sourceSets {
    test {
        java {
            // TurboQuant, AWQ, AutoRound, and BnB adapters moved to Tafkir integration
            exclude("tech/kayys/aljabr/safetensor/quantization/QuantizationRoundTripTest.java")
        }
    }
}

tasks.test {
    useJUnitPlatform()
}
