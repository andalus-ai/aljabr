plugins {
    application
}

dependencies {
    implementation(project(":engine:gguf:aljabr-gguf-converter-java"))
    implementation(project(":spi:aljabr-spi-model"))

    testImplementation("org.junit.jupiter:junit-jupiter")
}

application {
    mainClass.set("tech.kayys.aljabr.tools.convert.ConversionCli")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
