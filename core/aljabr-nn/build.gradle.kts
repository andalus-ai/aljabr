plugins {
    java
}

dependencies {
    implementation(project(":core:aljabr-tensor"))

    // NN tests construct tensors through TensorFactory; make the CPU provider available at test runtime.
    testImplementation(project(":backend:cpu:aljabr-backend-cpu"))
}
