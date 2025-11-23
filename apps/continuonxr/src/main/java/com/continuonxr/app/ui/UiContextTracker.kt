package com.continuonxr.app.ui

import com.continuonxr.app.logging.UiContext

/**
 * Tracks workstation UI context (active panel, layout, focus) for RLDS logging.
 * In production, connect this to XR UI shell events.
 */
class UiContextTracker {
    @Volatile
    private var latest: UiContext = UiContext()

    fun update(context: UiContext) {
        latest = context
    }

    fun current(): UiContext = latest
}

