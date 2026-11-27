plugins {
    `java-library`
    `maven-publish`
}

dependencies {
    api(project(":models:diffusion:aljabr-diffusion-execution-api"))
    api(project(":models:diffusion:aljabr-diffusion-api"))
    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.10.2")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
    withSourcesJar()
    withJavadocJar()
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
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
