"""Build the tank step by step: 1) glass, 2) + ocean water, 3) + spray/bubbles/steam/tracers."""
import sys
from PIL import Image
from oceantank import Tank, Camera, Timeline, Scene, over

size = (1280, 720)
tank = Tank(length=30, width=8, depth=3.2, glass_height=4.3)
tl = Timeline(tank)                  # cyclone "already formed", like at the start of the animation
cam = Camera(az=0.0, el=0.62)

scene = Scene(tank).build_glass()                 # 1) the rectangular glass tank
Image.fromarray(over(scene.render(tl, cam, size))).save("1_glass.png")

scene.add_ocean()                                  # 2) the ocean water
Image.fromarray(over(scene.render(tl, cam, size))).save("2_glass_water.png")

scene.add_particles(tl)                            # 3) particles (optional)
Image.fromarray(over(scene.render(tl, cam, size))).save("3_full.png")
print("ok")
