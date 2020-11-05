import brainrender
brainrender.SHADER_STYLE = 'cartoon'
from brainrender.scene import Scene


# Create a scene
scene = Scene()

# Add the whole EC in gray
scene.add_brain_regions(['ENT'], alpha=0.7)

# Add VAL nucleus in wireframe style with the allen color
scene.add_optic_cannula(target_region='ENT', radius=100, z_offset=700)

scene.render()

