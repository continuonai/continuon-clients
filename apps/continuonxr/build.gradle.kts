import com.google.protobuf.gradle.*
import com.android.build.api.dsl.AndroidSourceSet
import org.gradle.api.file.SourceDirectorySet
import org.gradle.api.plugins.ExtensionAware

fun AndroidSourceSet.proto(action: SourceDirectorySet.() -> Unit) {
    (this as ExtensionAware)
        .extensions
        .getByName("proto")
        .let { it as SourceDirectorySet }
        .action()
}

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.protobuf")
    id("org.jetbrains.kotlin.plugin.serialization")
}

android {
    namespace = "com.continuonxr.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.continuonxr.app"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.0.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.15"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    sourceSets {
        getByName("main") {
            proto {
                srcDir("../../proto")
            }
        }
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.10.01")
    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3:1.3.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")

    implementation("com.google.protobuf:protobuf-javalite:3.25.3")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("io.grpc:grpc-okhttp:1.64.0")
    implementation("io.grpc:grpc-protobuf-lite:1.64.0")
    implementation("io.grpc:grpc-stub:1.64.0")
    implementation("javax.annotation:javax.annotation-api:1.3.2")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // Jetpack XR / SceneCore (update versions when newer drops)
    implementation("androidx.xr:runtime:1.0.0-alpha04")
    implementation("androidx.xr.compose:compose-runtime:1.0.0-alpha04")
    implementation("androidx.xr:scenecore:1.0.0-alpha04")
    implementation("org.webrtc:google-webrtc:1.0.32006")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}

protobuf {
    protoc {
        artifact = "com.google.protobuf:protoc:3.25.3"
    }
    plugins {
        id("grpc") {
            artifact = "io.grpc:protoc-gen-grpc-java:1.64.0"
        }
    }
    generateProtoTasks {
        all().configureEach {
            builtins {
                maybeCreate("java").option("lite")
            }
            plugins {
                id("grpc") {
                    option("lite")
                }
            }
        }
    }
}

tasks.matching { it.name == "preBuild" }.configureEach {
    dependsOn(rootProject.tasks.named("validateProtoSchemas"))
}

tasks.withType<Test>().configureEach {
    dependsOn(rootProject.tasks.named("generateProtoKotlin"))
}
