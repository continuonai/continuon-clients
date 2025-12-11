package com.continuonxr.app.logging

import com.continuonxr.app.glove.GloveDiagnostics

fun Observation.withVideoDepthTimestamps(
    videoTimestampNanos: Long? = null,
    depthTimestampNanos: Long? = null,
): Observation = copy(
    videoTimestampNanos = this.videoTimestampNanos ?: videoTimestampNanos,
    depthTimestampNanos = this.depthTimestampNanos ?: depthTimestampNanos,
)

fun Observation.withGloveDiagnostics(glove: GloveDiagnostics?): Observation {
    if (glove == null) return this
    return copy(diagnostics = diagnostics.withGloveDiagnostics(glove))
}

