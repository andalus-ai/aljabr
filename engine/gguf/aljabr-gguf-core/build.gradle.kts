plugins {
    `java-library`
}

sourceSets {
    main {
        java {
            // Keep the actively used GGUF metadata/parser/tokenizer surface.
            // The module also contains older exporter/runner/training trees that
            // do not match today's package layout and pull in stale dependencies.
            include(
                "tech/kayys/aljabr/gguf/core/GGUFConstants.java",
                "tech/kayys/aljabr/gguf/core/GGUFTensorInfo.java",
                "tech/kayys/aljabr/gguf/core/GgmlType.java",
                "tech/kayys/aljabr/gguf/core/GgufMetaType.java",
                "tech/kayys/aljabr/gguf/core/GgufMetaValue.java",
                "tech/kayys/aljabr/gguf/core/GgufExporter.java",
                "tech/kayys/aljabr/gguf/core/GgufModel.java",
                "tech/kayys/aljabr/gguf/loader/GGUFModel.java",
                "tech/kayys/aljabr/gguf/loader/GGUFTensorInfo.java",
                "tech/kayys/aljabr/gguf/loader/GGUFParser.java",
                "tech/kayys/aljabr/gguf/loader/GGUFReader.java",
                "tech/kayys/aljabr/gguf/loader/gguf/*.java",
                "tech/kayys/aljabr/gguf/loader/quant/*.java",
                "tech/kayys/aljabr/gguf/loader/sampler/*.java",
                "tech/kayys/aljabr/gguf/loader/inference/*.java",
                "tech/kayys/aljabr/gguf/loader/tensor/*.java",
                "tech/kayys/aljabr/gguf/runtime/*.java",
                "tech/kayys/aljabr/gguf/tokenizer/GGUFTokenizer.java",
                "tech/kayys/aljabr/gguf/writer/GGUFWriter.java",
                "tech/kayys/aljabr/gguf/model/aljabr/*.java",
                "tech/kayys/aljabr/gguf/model/ModelConfig.java",
                "tech/kayys/aljabr/gguf/runner/AljabrWeightAdapter.java",
            )
        }
    }
}

dependencies {
    api(project(":core:aljabr-nn"))
    api(project(":core:aljabr-tensor"))
    implementation(project(":core:aljabr-spi-model"))
    implementation("com.google.code.gson:gson:2.10.1")
    api(project(":core:aljabr-spi-tokenizer"))
    implementation("io.quarkus:quarkus-core:3.32.2")
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
