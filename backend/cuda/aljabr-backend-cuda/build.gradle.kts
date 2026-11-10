plugins {
    java
}

dependencies {
    implementation(project(":plugins:ml:aljabr-tensor"))
    implementation(project(":backend:cpu:aljabr-backend-cpu"))
    implementation(project(":backend:cuda:aljabr-kernel-cuda"))
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")
}
