plugins {
    `java-library`
}

dependencies {
    implementation(project(":engine:gguf:aljabr-gguf-converter"))
    implementation(project(":engine:gguf:aljabr-gguf-core"))
    implementation(project(":core:aljabr-core"))
    api(project(":engine:safetensor:aljabr-safetensor-loader"))
    api(project(":spi:aljabr-spi-model"))
    implementation(project(":engine:safetensor:aljabr-safetensor-core"))
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("com.google.code.gson:gson:2.11.0")
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api")
    implementation("jakarta.inject:jakarta.inject-api")
    implementation("org.slf4j:slf4j-api:2.0.13")

    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("org.assertj:assertj-core:3.26.3")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
