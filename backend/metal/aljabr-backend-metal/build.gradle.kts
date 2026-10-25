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
sourceSets {
    main {
        java {
            setSrcDirs(listOf("src/main/java"))
            include("tech/kayys/aljabr/backend/metal/binding/MetalBinding.java")
            include("tech/kayys/aljabr/backend/metal/binding/MetalCpuFallback.java")
            include("tech/kayys/aljabr/backend/metal/binding/MetalFlashAttentionBinding.java")
            include("tech/kayys/aljabr/backend/metal/binding/MetalFlashAttentionCpuFallback.java")
            include("tech/kayys/aljabr/backend/metal/binding/MetalLibraryDiscovery.java")
            include("tech/kayys/aljabr/backend/metal/MetalComputeBackend.java")
        }
        resources {
            srcDir("src/main/cpp/resources")
        }
    }
}


dependencies {
    implementation(project(":core:aljabr-tensor"))
    implementation(project(":backend:cpu:aljabr-backend-cpu"))
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.1.0")
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

// Show CSV output only for the opt-in native Metal benchmark. Regular test
// runs keep Gradle's default output capture to avoid noisy successful tests.
val metalBenchmarkEnabled = System.getenv("ALJABR_METAL_BENCH")?.let {
    it == "1" || it.equals("true", ignoreCase = true)
} ?: false

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    testLogging {
        // Show detailed assertion messages and the failing source line in CI.
        // This is especially important for native numerical-parity failures,
        // where a test name alone does not identify the bad output or row.
        events("failed")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
        showExceptions = true
        showCauses = true
        showStackTraces = true
        showStandardStreams = metalBenchmarkEnabled
    }
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
