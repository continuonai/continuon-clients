plugins {
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.25" apply false
    id("com.google.protobuf") version "0.9.4" apply false
    id("org.jetbrains.kotlin.plugin.serialization") version "1.9.25" apply false
}

val bufVersion = "1.34.0"
val bufBinary = layout.buildDirectory.file("buf/bin/buf").get().asFile

tasks.register("installBuf") {
    group = "build setup"
    description = "Download a local buf binary for proto linting and codegen"
    outputs.file(bufBinary)

    doLast {
        if (!bufBinary.exists()) {
            bufBinary.parentFile.mkdirs()
            val url = java.net.URL("https://github.com/bufbuild/buf/releases/download/v$bufVersion/buf-Linux-x86_64")
            url.openStream().use { input ->
                bufBinary.outputStream().use { output -> input.copyTo(output) }
            }
            bufBinary.setExecutable(true)
        }
    }
}

tasks.register<Exec>("validateProtoSchemas") {
    group = "verification"
    description = "Run buf lint against proto definitions"
    dependsOn("installBuf")
    inputs.files(file("proto/buf.yaml"), file("buf.gen.yaml"))
    inputs.dir("proto")
    workingDir = file("proto")
    commandLine(bufBinary.absolutePath, "lint")
}

tasks.register("generateProtoKotlin") {
    group = "code generation"
    description = "Alias for generating Kotlin/JVM stubs for the XR app"
    dependsOn(":apps:continuonxr:generateDebugProto", ":apps:continuonxr:generateReleaseProto")
}

tasks.matching { it.name == "check" }.configureEach {
    dependsOn("validateProtoSchemas")
}
