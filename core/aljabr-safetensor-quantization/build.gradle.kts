plugins {
    java
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


    testImplementation("org.awaitility:awaitility:4.2.1")
    testImplementation("io.quarkus:quarkus-junit5")
    testImplementation("io.rest-assured:rest-assured:5.4.0")
}

tasks.test {
    useJUnitPlatform()
}
