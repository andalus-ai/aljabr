plugins {
    `java-library`
}

dependencies {
    api(project(":core:aljabr-core"))
    testImplementation(testFixtures(project(":core:aljabr-core")))
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test { useJUnitPlatform() }
