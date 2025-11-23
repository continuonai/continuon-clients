package com.continuonxr.app.logging

import com.continuonxr.app.config.LoggingConfig
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class HttpRldsUploader(
    private val config: LoggingConfig,
    private val client: OkHttpClient = OkHttpClient(),
) : RldsUploader {
    override fun upload(episodeDir: File) {
        val endpoint = config.uploadEndpoint ?: return
        val zipFile = File.createTempFile("rlds-episode", ".zip", episodeDir.parentFile)
        zipDirectory(episodeDir, zipFile)
        val requestBuilder = Request.Builder()
            .url(endpoint)
            .post(zipFile.asRequestBody("application/zip".toMediaType()))
        config.uploadAuthToken?.let { token ->
            requestBuilder.addHeader("Authorization", "Bearer $token")
        }
        client.newCall(requestBuilder.build()).execute().use {
            // Best-effort; ignore body in stub.
        }
        zipFile.delete()
    }

    private fun zipDirectory(sourceDir: File, zipFile: File) {
        ZipOutputStream(zipFile.outputStream()).use { zos ->
            sourceDir.walkTopDown()
                .filter { it.isFile }
                .forEach { file ->
                    val entryName = sourceDir.toURI().relativize(file.toURI()).path
                    zos.putNextEntry(ZipEntry(entryName))
                    file.inputStream().use { input -> input.copyTo(zos) }
                    zos.closeEntry()
                }
        }
    }
}

