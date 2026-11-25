plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

dependencies {
    api(project(":models:aljabr-model-chronos2"))
    api(project(":engine:safetensor:aljabr-safetensor-loader"))
    api(project(":engine:safetensor:aljabr-safetensor-tensor-storage"))
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api")
    implementation("jakarta.inject:jakarta.inject-api")
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.10.2")
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
        }
    }
    repositories {
        mavenLocal()
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
