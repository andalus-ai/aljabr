import org.gradle.api.tasks.SourceSetContainer
import org.gradle.api.publish.PublishingExtension
import org.gradle.api.publish.maven.MavenPublication
import org.gradle.plugins.signing.SigningExtension

buildscript {
    repositories {
        google()
        mavenCentral()
        mavenLocal()
    }
    dependencies {
        classpath("io.smallrye:jandex:3.5.3")
    }
}

plugins {
    java
    `maven-publish`
    id("io.quarkus") version "3.32.2" apply false
}

extra["aljabrVersion"] = "0.1.0-SNAPSHOT"
extra["quarkusVersion"] = "3.32.2"

allprojects {
    group = "tech.kayys.aljabr"
    version = rootProject.extra["aljabrVersion"] as String

    repositories {
        google()
        mavenCentral()
        mavenLocal()
    }
}

subprojects {
    apply(plugin = "java")
    apply(plugin = "maven-publish")

    val quarkusVersion = rootProject.extra["quarkusVersion"] as String
    val mutinyVersion = "2.5.5"
    val smallryeMutinyVertxVersion = "3.15.1"
    val caffeineVersion = "3.1.8"
    val commonsCollectionsVersion = "4.4"
    val jakartaValidationVersion = "3.0.2"
    val jakartaEnterpriseVersion = "4.0.1"
    val jakartaInjectVersion = "2.0.1"
    val reactiveStreamsVersion = "1.0.4"
    val jbossLoggingVersion = "3.6.1.Final"
    val jacksonVersion = "2.16.1"
    val junitJupiterVersion = "5.10.2"
    val junitPlatformVersion = "1.10.2"
    val mockitoJupiterVersion = "5.14.2"

    java {
        toolchain {
            languageVersion.set(JavaLanguageVersion.of(25))
        }
        withSourcesJar()
        withJavadocJar()
    }

    // Configure Javadoc task
    tasks.withType<Javadoc> {
        options {
            this as StandardJavadocDocletOptions
            addBooleanOption("Xdoclint:none", true)
            addStringOption("-add-modules", "jdk.incubator.vector")
        }
    }

    dependencies {
        // Common logging and util dependencies can go here
        add("testRuntimeOnly", "org.junit.platform:junit-platform-launcher:$junitPlatformVersion")
        // Make JUnit Jupiter + AssertJ available to every subproject test classpath
        add("testImplementation", platform("org.junit:junit-bom:$junitJupiterVersion"))
        add("testImplementation", "org.junit.jupiter:junit-jupiter")
        add("testImplementation", "org.assertj:assertj-core:3.25.3")
    }

    tasks.withType<JavaCompile> {
        options.encoding = "UTF-8"
        // Enable preview features if needed for Java 21 FFM
        options.compilerArgs.add("--enable-preview")
        options.compilerArgs.add("--add-modules=jdk.incubator.vector")
    }

    tasks.withType<Test>().configureEach {
        dependsOn(tasks.named("jar"))
        jvmArgs(
            "--enable-preview",
            "--add-modules=jdk.incubator.vector",
            "--enable-native-access=ALL-UNNAMED",
        )
        useJUnitPlatform()
    }

    tasks.withType<JavaExec>().configureEach {
        jvmArgs(
            "--enable-preview",
            "--add-modules=jdk.incubator.vector",
            "--enable-native-access=ALL-UNNAMED",
        )
    }

    val sourceSets = extensions.getByType(SourceSetContainer::class.java)
    val mainSourceSet = sourceSets.named("main")
    val jandexOutputDir = layout.buildDirectory.dir("generated/jandex/main")

    val generateJandexIndex = tasks.register("generateJandexIndex") {
        dependsOn(tasks.named("compileJava"), tasks.named("processResources"))

        val classesDirs = mainSourceSet.map { it.output.classesDirs }
        val indexFile = jandexOutputDir.map { it.file("META-INF/jandex.idx") }

        inputs.files(classesDirs)
        outputs.file(indexFile)

        doLast {
            val indexer = org.jboss.jandex.Indexer()

            classesDirs.get().files
                .filter { it.exists() }
                .forEach { classesDir ->
                    classesDir.resolve("META-INF/jandex.idx").let { staleIndex ->
                        if (staleIndex.exists()) {
                            staleIndex.delete()
                        }
                    }
                    classesDir.walkTopDown()
                        .filter { it.isFile && it.extension == "class" }
                        .forEach { classFile ->
                            classFile.inputStream().use(indexer::index)
                        }
                }

            val indexOutput = indexFile.get().asFile
            indexOutput.parentFile.mkdirs()
            indexOutput.outputStream().use { output ->
                org.jboss.jandex.IndexWriter(output).write(indexer.complete())
            }
        }
    }

    mainSourceSet.configure {
        output.dir(mapOf("builtBy" to generateJandexIndex), jandexOutputDir)
    }

    tasks.named("classes") {
        dependsOn(generateJandexIndex)
    }

    configurations.configureEach {
        resolutionStrategy.dependencySubstitution {
            fun safeSubstitute(moduleNotation: String, projectPath: String) {
                if (findProject(projectPath) != null) {
                    substitute(module(moduleNotation)).using(project(projectPath))
                }
            }
            safeSubstitute("tech.kayys.aljabr:aljabr-spi-model", ":core:aljabr-spi-model")
            safeSubstitute("tech.kayys.aljabr:aljabr-tensor", ":core:aljabr-tensor")
            safeSubstitute("tech.kayys.aljabr:aljabr-error-code", ":core:aljabr-error-code")
            safeSubstitute("tech.kayys.aljabr:aljabr-tokenizer-core", ":core:aljabr-tokenizer-core")
            safeSubstitute("tech.kayys.aljabr:aljabr-safetensor-api", ":core:aljabr-safetensor-api")
            safeSubstitute("tech.kayys.aljabr:aljabr-safetensor-core", ":core:aljabr-safetensor-core")
            safeSubstitute("tech.kayys.aljabr:aljabr-safetensor-loader", ":core:aljabr-safetensor-loader")
            safeSubstitute("tech.kayys.aljabr:aljabr-safetensor-spi", ":core:aljabr-safetensor-spi")
            safeSubstitute("tech.kayys.aljabr:aljabr-nn", ":core:aljabr-nn")
            safeSubstitute("tech.kayys.aljabr:aljabr-3d", ":core:aljabr-3d")
            safeSubstitute("tech.kayys.aljabr:aljabr-core", ":core:aljabr-core")
            safeSubstitute("tech.kayys.aljabr:aljabr-tensor", ":core:aljabr-tensor")
            substitute(module("tech.kayys.aljabr:tafkir-ml-core")).using(module("tech.kayys.tafkir:tafkir-ml-core:0.1.0-SNAPSHOT"))
        }
        resolutionStrategy.eachDependency {
            if (requested.group == "io.quarkus" && requested.version.isNullOrBlank()) {
                useVersion(quarkusVersion)
                because("Align Quarkus dependency versions across the Gradle migration")
            }
            if (requested.group == "io.smallrye.reactive"
                && requested.name == "mutiny"
                && requested.version.isNullOrBlank()) {
                useVersion(mutinyVersion)
                because("Align Mutiny versions across mixed legacy Gradle modules")
            }
            if (requested.group == "io.smallrye.reactive"
                && requested.name == "smallrye-mutiny-vertx-web-client"
                && requested.version.isNullOrBlank()) {
                useVersion(smallryeMutinyVertxVersion)
                because("Align SmallRye Vert.x client versions across Gradle modules")
            }
            if (requested.group == "com.github.ben-manes.caffeine"
                && requested.name == "caffeine"
                && requested.version.isNullOrBlank()) {
                useVersion(caffeineVersion)
                because("Align Caffeine versions across mixed legacy Gradle modules")
            }
            if (requested.group == "org.apache.commons"
                && requested.name == "commons-collections4"
                && requested.version.isNullOrBlank()) {
                useVersion(commonsCollectionsVersion)
                because("Align Commons Collections versions across mixed legacy Gradle modules")
            }
            if (requested.group == "jakarta.validation"
                && requested.name == "jakarta.validation-api"
                && requested.version.isNullOrBlank()) {
                useVersion(jakartaValidationVersion)
                because("Align Jakarta Validation versions across mixed legacy Gradle modules")
            }
            if (requested.group == "jakarta.enterprise"
                && requested.name == "jakarta.enterprise.cdi-api"
                && requested.version.isNullOrBlank()) {
                useVersion(jakartaEnterpriseVersion)
                because("Align Jakarta CDI versions across mixed legacy Gradle modules")
            }
            if (requested.group == "jakarta.inject"
                && requested.name == "jakarta.inject-api"
                && requested.version.isNullOrBlank()) {
                useVersion(jakartaInjectVersion)
                because("Align Jakarta Inject versions across mixed legacy Gradle modules")
            }
            if (requested.group == "org.reactivestreams"
                && requested.name == "reactive-streams"
                && requested.version.isNullOrBlank()) {
                useVersion(reactiveStreamsVersion)
                because("Align Reactive Streams versions across mixed legacy Gradle modules")
            }
            if (requested.group == "org.jboss.logging"
                && requested.name == "jboss-logging"
                && requested.version.isNullOrBlank()) {
                useVersion(jbossLoggingVersion)
                because("Align JBoss Logging versions across mixed legacy Gradle modules")
            }
            if (requested.group == "com.fasterxml.jackson.core"
                && (requested.name == "jackson-core" || requested.name == "jackson-databind")
                && requested.version.isNullOrBlank()) {
                useVersion(jacksonVersion)
                because("Align Jackson core versions across mixed legacy Gradle modules")
            }
            if (requested.group == "com.fasterxml.jackson.datatype"
                && requested.name == "jackson-datatype-jsr310"
                && requested.version.isNullOrBlank()) {
                useVersion(jacksonVersion)
                because("Align Jackson datatype versions across mixed legacy Gradle modules")
            }
            if (requested.group == "org.junit.jupiter"
                && requested.name.startsWith("junit-jupiter")
                && requested.version.isNullOrBlank()) {
                useVersion(junitJupiterVersion)
                because("Align JUnit Jupiter versions across mixed legacy Gradle modules")
            }
            if (requested.group == "org.mockito"
                && requested.name == "mockito-junit-jupiter"
                && requested.version.isNullOrBlank()) {
                useVersion(mockitoJupiterVersion)
                because("Align Mockito JUnit integration versions across mixed legacy Gradle modules")
            }
        }
    }

    afterEvaluate {
        extensions.configure<PublishingExtension>("publishing") {
            repositories {
                // ── GitHub Packages ───────────────────────────────────────────
                // Credentials: set GITHUB_ACTOR and GITHUB_TOKEN as repository secrets.
                // GitHub Packages Maven registry requires: https://maven.pkg.github.com/OWNER/REPOSITORY
                val repoUrlProperty = providers.gradleProperty("deployment.repo.url").orNull
                val githubRepo = System.getenv("GITHUB_REPOSITORY")
                    ?: providers.gradleProperty("github.repository").orNull
                    ?: run {
                        val ghOwner = System.getenv("GITHUB_REPOSITORY_OWNER")
                            ?: System.getenv("GITHUB_ACTOR")
                            ?: "bhangun"
                        "$ghOwner/aljabr"
                    }
                val ghActor  = System.getenv("GITHUB_ACTOR")  ?: providers.gradleProperty("gpr.user").orNull
                val ghToken  = System.getenv("GITHUB_TOKEN")  ?: providers.gradleProperty("gpr.key").orNull
                if (!ghToken.isNullOrBlank() && !ghActor.isNullOrBlank()) {
                    maven {
                        name = "GitHubPackages"
                        url  = uri(repoUrlProperty ?: "https://maven.pkg.github.com/$githubRepo")
                        credentials {
                            username = ghActor
                            password = ghToken
                        }
                    }
                }
            }
            if (publications.findByName("mavenJava") == null) {
                publications.create<MavenPublication>("mavenJava") {
                    from(components["java"])
                    pom {
                        name.set(project.name)
                        description.set("Aljabr ML Framework Module")
                        url.set("https://github.com/wayang-platform/wayang-platform")
                        licenses {
                            license {
                                name.set("The Apache License, Version 2.0")
                                url.set("http://www.apache.org/licenses/LICENSE-2.0.txt")
                            }
                        }
                        developers {
                            developer {
                                id.set("wayang")
                                name.set("Wayang Platform")
                                email.set("dev@wayang.tech")
                            }
                        }
                        scm {
                            connection.set("scm:git:git://github.com/wayang-platform/wayang-platform.git")
                            developerConnection.set("scm:git:ssh://github.com/wayang-platform/wayang-platform.git")
                            url.set("https://github.com/wayang-platform/wayang-platform")
                        }
                    }
                }
            }
        }
        
        val signingKey = System.getenv("ORG_GRADLE_PROJECT_signingKey")
        if (!signingKey.isNullOrBlank()) {
            apply(plugin = "signing")
            extensions.configure<SigningExtension>("signing") {
                val signingPassword = System.getenv("ORG_GRADLE_PROJECT_signingPassword")
                useInMemoryPgpKeys(signingKey, signingPassword)
                sign(extensions.getByType<PublishingExtension>().publications["mavenJava"])
            }
        }
    }
}

