plugins {
    `java-library`
}

dependencies {
   // api(project(":spi:aljabr-spi"))

    implementation(project(":core:aljabr-tensor"))
    api(project(":core:aljabr-error-code"))
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("jakarta.validation:jakarta.validation-api:3.0.2")
    implementation("org.jboss.logging:jboss-logging:3.5.3.Final")
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
