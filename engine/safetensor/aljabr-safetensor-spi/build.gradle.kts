plugins {
    java
}

dependencies {
    implementation(project(":spi:aljabr-spi-tokenizer"))
    implementation(project(":spi:aljabr-spi-inference"))
    implementation(project(":spi:aljabr-spi-model"))
}
