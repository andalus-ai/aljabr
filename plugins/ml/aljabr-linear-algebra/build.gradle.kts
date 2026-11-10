plugins {
    java
}

dependencies {
    implementation(project(":plugins:ml:aljabr-numerics"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test { useJUnitPlatform() }
