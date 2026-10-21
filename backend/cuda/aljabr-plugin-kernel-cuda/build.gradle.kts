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

dependencies {
   // implementation(project(":core:plugin:aljabr-plugin-kernel-core"))
    implementation(project(":core:aljabr-tensor"))
    implementation(project(":backend:cuda:aljabr-kernel-cuda"))
    compileOnly(group = "org.jboss.logging", name = "jboss-logging")
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

tasks.jar {
    manifest {
        attributes(
            mapOf(
                "Plugin-Author" to "Aljabr Team",
                "Plugin-Capabilities" to "cuda-acceleration, flash-attention-2, flash-attention-3, paged-attention, gpu-inference",
                "Plugin-Deployment" to "microservice,hybrid",
                "Plugin-GPU-Requirement" to "NVIDIA GPU, CUDA 11.0+",
                "Plugin-Id" to "cuda-kernel",
                "Plugin-License" to "MIT",
                "Plugin-Minimum-Compute-Capability" to "6.0",
                "Plugin-Minimum-Memory" to "4GB",
                "Plugin-Name" to "CUDA Kernel",
                "Plugin-Performance-Memory-Overhead" to "50-100MB",
                "Plugin-Performance-Speedup" to "5-10x (vs CPU)",
                "Plugin-Provider" to "tech.kayys.aljabr.plugin.kernel.cuda.CudaKernelPlugin",
                "Plugin-Type" to "kernel",
                "Plugin-Vendor" to "Kayys.tech",
                "Plugin-Version" to "0.1.0-SNAPSHOT",
                "Supported-Architectures" to "volta,turing,ampere,ada,hopper,blackwell",
                "Supported-Compute-Capabilities" to "6.0,6.1,7.0,7.5,8.0,8.6,8.9,9.0,10.0"
            )
        )
    }
}
