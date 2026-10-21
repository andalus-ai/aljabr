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

repositories {
    mavenCentral()
    mavenLocal()
}

sourceSets {
    named("main") {
        java {
            setSrcDirs(listOf("src/main/java"))
            include("tech/kayys/aljabr/cuda/binding/CudaBinding.java")
            include("tech/kayys/aljabr/cuda/binding/CudaCpuFallback.java")
            include("tech/kayys/aljabr/cuda/detection/CudaDetector.java")
            include("tech/kayys/aljabr/cuda/detection/CudaCapabilities.java")
            include("tech/kayys/aljabr/cuda/gpu/GPUMemoryPool.java")
            include("tech/kayys/aljabr/cuda/gpu/GPUAccelerator.java")
            include("tech/kayys/aljabr/cuda/gpu/CUDAStreamManager.java")
            include("tech/kayys/aljabr/cuda/config/CudaRunnerMode.java")
        }
    }
}

dependencies {
    implementation(project(":core:aljabr-tensor"))
    implementation(group = "io.quarkus", name = "quarkus-arc")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(group = "org.assertj", name = "assertj-core")
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
