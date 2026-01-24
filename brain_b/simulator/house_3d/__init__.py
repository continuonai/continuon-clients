"""
High-Fidelity 3D House POV Training System

A photorealistic 3D training environment that shows the robot's point-of-view
inside a house, with split-view display (robot POV + overhead map + sensor panels).

Usage:
    from brain_b.simulator.house_3d import HouseRenderer, HouseScene, HouseAssets

    # Create a house scene
    scene = HouseScene.from_template('studio_apartment')

    # Render robot POV
    renderer = HouseRenderer(width=640, height=480)
    frame = renderer.render_pov(scene, robot_position, robot_heading)

    # Render overhead map
    overhead = renderer.render_overhead(scene, robot_position)
"""

from .assets import (
    HouseAssets,
    Room,
    Wall,
    Floor,
    Furniture,
    RoomType,
    HOUSE_TEMPLATES,
)

from .materials import (
    Material,
    PBRMaterial,
    MATERIAL_LIBRARY,
    TextureType,
)

from .camera import (
    Camera,
    PerspectiveCamera,
    OrthographicCamera,
    RobotCamera,
)

from .renderer import (
    HouseRenderer,
    RenderSettings,
    LightSource,
)

from .scene import (
    HouseScene,
    SceneObject,
    Transform,
)

from .training_integration import (
    Visual3DTrainingEnvironment,
    Visual3DFrame,
    Visual3DEpisodeStep,
    House3DPerceptionAdapter,
    create_visual_training_env,
    run_visual_training_episode,
    generate_visual_training_batch,
)

__all__ = [
    # Assets
    'HouseAssets',
    'Room',
    'Wall',
    'Floor',
    'Furniture',
    'RoomType',
    'HOUSE_TEMPLATES',
    # Materials
    'Material',
    'PBRMaterial',
    'MATERIAL_LIBRARY',
    'TextureType',
    # Camera
    'Camera',
    'PerspectiveCamera',
    'OrthographicCamera',
    'RobotCamera',
    # Renderer
    'HouseRenderer',
    'RenderSettings',
    'LightSource',
    # Scene
    'HouseScene',
    'SceneObject',
    'Transform',
    # Training Integration
    'Visual3DTrainingEnvironment',
    'Visual3DFrame',
    'Visual3DEpisodeStep',
    'House3DPerceptionAdapter',
    'create_visual_training_env',
    'run_visual_training_episode',
    'generate_visual_training_batch',
]

__version__ = '1.0.0'
