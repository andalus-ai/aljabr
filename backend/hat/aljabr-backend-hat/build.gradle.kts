plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(26))
    }
}

// This module is included only with -PwithHat=true and a validated Babylon JDK.
val babylonJdkPath = providers.gradleProperty("babylonJdkPath")
    .orElse(providers.environmentVariable("BABYLON_JDK_HOME"))
    .get()

tasks.withType<JavaCompile> {
    options.isFork = true
    options.forkOptions.executable = file("$babylonJdkPath/bin/javac").absolutePath
    options.compilerArgs.addAll(listOf("--enable-preview", "--add-modules=jdk.incubator.code", "--add-modules=jdk.incubator.vector", "-source", "26", "-target", "26"))
}

tasks.withType<Javadoc> {
    enabled = false
}

sourceSets {
    main {
        java {
            setSrcDirs(listOf("src/main/java"))
        }
    }
}

dependencies {
    implementation(project(":core:aljabr-tensor"))
    implementation(project(":backend:cpu:aljabr-backend-cpu"))
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")
    implementation("oracle.code:hat-core:1.0")
    implementation("oracle.code:hat-backend-java-seq:1.0")
    implementation("oracle.code:hat-backend-java-mt:1.0")
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    executable = file("$babylonJdkPath/bin/java").absolutePath
    jvmArgs("--enable-preview", "--add-modules=jdk.incubator.code", "--add-modules=jdk.incubator.vector")
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
