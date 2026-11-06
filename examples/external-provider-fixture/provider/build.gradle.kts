plugins {
    `java-library`
}

val aljabrVersion = providers.gradleProperty("aljabrVersion").get()

dependencies {
    // The external provider depends only on Aljabr's published provider SPI.
    api("tech.kayys.aljabr:aljabr-spi-core:$aljabrVersion")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}
