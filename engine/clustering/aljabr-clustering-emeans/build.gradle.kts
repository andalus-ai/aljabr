plugins {
    `java-library`
}

dependencies {
    api(project(":engine:clustering:aljabr-clustering-api"))
    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
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
    add(jmh.implementationConfigurationName, project(":engine:clustering:aljabr-clustering-kmeans"))
    add(jmh.implementationConfigurationName, project(":backend:cpu:aljabr-backend-cpu"))
    add(jmh.annotationProcessorConfigurationName, "org.openjdk.jmh:jmh-generator-annprocess:1.37")
}

tasks.register<JavaExec>("jmh") {
    group = "benchmark"
    description = "Runs e-means versus Lloyd CPU clustering benchmarks."
    dependsOn(jmh.classesTaskName)
    classpath = jmh.runtimeClasspath
    mainClass.set("org.openjdk.jmh.Main")
    args("-f", "1", "-wi", "3", "-i", "5", "-w", "1s", "-r", "1s")
}
