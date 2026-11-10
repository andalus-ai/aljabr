plugins {
    `java-library`
}

dependencies {
    implementation(project(":plugins:ml:aljabr-tensor"))
    api(project(":spi:aljabr-spi-compute"))
    testImplementation(project(":spi:aljabr-spi-testkit"))
}

val jmh by sourceSets.creating {
    java.srcDir("src/jmh/java")
    compileClasspath += sourceSets.main.get().output
    runtimeClasspath += output + compileClasspath
}

configurations[jmh.implementationConfigurationName]
    .extendsFrom(configurations.implementation.get())
configurations[jmh.runtimeOnlyConfigurationName]
    .extendsFrom(configurations.runtimeOnly.get())

dependencies {
    add(jmh.implementationConfigurationName, "org.openjdk.jmh:jmh-core:1.37")
    add(jmh.annotationProcessorConfigurationName, "org.openjdk.jmh:jmh-generator-annprocess:1.37")
}

tasks.register<JavaExec>("jmh") {
    group = "benchmark"
    description = "Runs CPU attention JMH benchmarks."
    dependsOn(jmh.classesTaskName)
    classpath = jmh.runtimeClasspath
    mainClass.set("org.openjdk.jmh.Main")
    args("-f", "2", "-wi", "5", "-i", "8", "-w", "1s", "-r", "1s")
}

tasks.named("check") {
    dependsOn(jmh.classesTaskName)
}
