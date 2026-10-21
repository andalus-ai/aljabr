plugins {
    `java-library`
}

dependencies {
    api(project(":core:aljabr-core"))
    implementation("org.rocksdb:rocksdbjni:9.2.1")
}
