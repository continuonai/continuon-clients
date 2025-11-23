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
) : RldsUploadTransport {
    override fun upload(episodeDir: File, metadata: EpisodeMetadata): Boolean {
        val endpoint = config.uploadEndpoint ?: return false
        val zipFile = File.createTempFile("rlds-episode", ".zip", episodeDir.parentFile)
        return try {
            zipDirectory(episodeDir, zipFile)
            val requestBuilder = Request.Builder()
                .url(endpoint)
                .post(zipFile.asRequestBody("application/zip".toMediaType()))
                .addHeader("X-Continuon-Tag-xr_mode", metadata.xrMode)
                .addHeader("X-Continuon-Tag-control_role", metadata.controlRole)
                .addHeader("X-Continuon-Tag-environment_id", metadata.environmentId)
            if (metadata.tags.isNotEmpty()) {
                requestBuilder.addHeader("X-Continuon-Tag-list", metadata.tags.joinToString(","))
            }
            config.uploadAuthToken?.let { token ->
                requestBuilder.addHeader("Authorization", "Bearer $token")
            }
            client.newCall(requestBuilder.build()).execute().use { response ->
                response.isSuccessful
            }
        } catch (e: Exception) {
            false
        } finally {
            zipFile.delete()
        }
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

