import org.gradle.api.GradleException

/**
 * Enforces the one-way dependency boundary: Aljabr is foundational and must
 * never compile against Tafkir implementation or workflow code.
 */
val verifyNoTafkirDependency = tasks.register("verifyNoTafkirDependency") {
    group = "verification"
    description = "Fails if Aljabr production sources or Gradle configuration depend on Tafkir."

    doLast {
        val root = rootProject.projectDir
        val ignoredDirectories = setOf(".git", ".gradle", "build", "out")
        val productionSource = Regex(""".*/src/main/.*\\.(java|kt|kts)$""")
        val gradleConfiguration = Regex(""".*/(build|settings)\\.gradle(\\.kts)?$""")
        val forbidden = Regex(
            """tech\\.kayys\\.tafkir|tafkir-quantizer-(gptq|awq|autoround|turboquant)|withTafkir"""
        )
        val violations = mutableListOf<String>()

        root.walkTopDown()
            .onEnter { directory -> directory.name !in ignoredDirectories }
            .filter { it.isFile }
            .filter { file ->
                val relative = file.relativeTo(root).invariantSeparatorsPath
                productionSource.matches("/" + relative) ||
                    gradleConfiguration.matches("/" + relative)
            }
            .forEach { file ->
                val relative = file.relativeTo(root).invariantSeparatorsPath
                val content = file.readText()
                if (forbidden.containsMatchIn(content)) {
                    violations += relative
                }
                if (relative.endsWith("settings.gradle.kts") || relative.endsWith("settings.gradle")) {
                    if (Regex("""includeBuild""").containsMatchIn(content) &&
                        Regex("""tafkir""", RegexOption.IGNORE_CASE).containsMatchIn(content)) {
                        violations += "$relative (Tafkir composite-build reference)"
                    }
                }
            }

        if (violations.isNotEmpty()) {
            throw GradleException(
                "Aljabr must not depend on Tafkir. Remove Tafkir references from:\n" +
                    violations.distinct().sorted().joinToString("\n") { " - $it" }
            )
        }
        logger.lifecycle("Aljabr/Tafkir dependency boundary verified: no reverse dependency found.")
    }
}

allprojects {
    tasks.matching { it.name == "check" }.configureEach {
        dependsOn(verifyNoTafkirDependency)
    }
}
