plugins {
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.25" apply false
    id("com.google.protobuf") version "0.9.4" apply false
    id("org.jetbrains.kotlin.plugin.serialization") version "1.9.25" apply false
}

val bufVersion = "1.34.0"
val bufBinary = layout.buildDirectory.file("buf/bin/buf").get().asFile

val installMockServerNodeModules = tasks.register<Exec>("installMockServerNodeModules") {
    group = "build setup"
    description = "Install npm dependencies for the mock ContinuonBrain server"
    workingDir = file("apps/mock-continuonbrain")
    inputs.files(file("apps/mock-continuonbrain/package.json"), file("apps/mock-continuonbrain/package-lock.json"))
    outputs.dir(file("apps/mock-continuonbrain/node_modules"))
    commandLine("npm", "install")
}

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

tasks.register<Exec>("generateProtoTypescript") {
    group = "code generation"
    description = "Generate TypeScript gRPC/ts-proto stubs via buf"
    dependsOn("installBuf", installMockServerNodeModules)
    inputs.files(file("proto/buf.yaml"), file("buf.gen.yaml"))
    inputs.dir("proto")
    outputs.dir(file("apps/mock-continuonbrain/src/generated"))
    workingDir = file("proto")
    commandLine(
        bufBinary.absolutePath,
        "generate",
        "--template",
        "../buf.gen.yaml"
    )
}

tasks.register("generateProtoKotlin") {
    group = "code generation"
    description = "Alias for generating Kotlin/JVM stubs for the XR app"
    dependsOn(":apps:continuonxr:generateDebugProto", ":apps:continuonxr:generateReleaseProto")
}

tasks.matching { it.name == "check" }.configureEach {
    dependsOn("validateProtoSchemas")
}
