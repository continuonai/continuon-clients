package com.continuonxr.app.navigation

enum class ModeRoute {
    ModeA,
    ModeB,
    ModeC,
}

data class ModeScreen(
    val route: ModeRoute,
    val title: String,
    val description: String,
)

val modeScreens: List<ModeScreen> = listOf(
    ModeScreen(
        route = ModeRoute.ModeA,
        title = "Mode A",
        description = "Foundation shell for spatial mapping and discovery.",
    ),
    ModeScreen(
        route = ModeRoute.ModeB,
        title = "Mode B",
        description = "Collaboration-first shell for co-located XR sessions.",
    ),
    ModeScreen(
        route = ModeRoute.ModeC,
        title = "Mode C",
        description = "Teleop shell for mixed-reality robot control.",
    ),
)
