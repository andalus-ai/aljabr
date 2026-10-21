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
