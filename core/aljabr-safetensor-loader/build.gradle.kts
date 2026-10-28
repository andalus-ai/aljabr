plugins {
    java
}

dependencies {
    implementation(project(":core:aljabr-safetensor-spi"))
    implementation(project(":core:aljabr-spi-inference"))
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("com.fasterxml.jackson.core:jackson-core:2.16.1")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("io.smallrye.config:smallrye-config:3.10.1")
    implementation("io.quarkus:quarkus-arc")
    implementation("io.quarkus:quarkus-jackson")
    implementation("io.quarkus:quarkus-micrometer")
    implementation("io.micrometer:micrometer-core:1.16.3")
    implementation("jakarta.annotation:jakarta.annotation-api:2.1.1")
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")
}


