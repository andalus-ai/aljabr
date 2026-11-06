plugins {
    application
}

val aljabrVersion = providers.gradleProperty("aljabrVersion").get()

dependencies {
    implementation("tech.kayys.aljabr:aljabr-spi-core:$aljabrVersion")
    runtimeOnly(project(":provider"))
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

application {
    mainClass.set("example.host.ProviderDiscoveryDemo")
}
