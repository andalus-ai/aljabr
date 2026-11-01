plugins {
    java
}

dependencies {
    implementation(project(":core:aljabr-numerics"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
