import carla
import random
from carla import *
from time import sleep

client = carla.Client('localhost', 2000)
client.set_timeout(20.0) # seconds
world = client.get_world()
blueprint_library = world.get_blueprint_library()

model_3 = random.choice(blueprint_library.filter('vehicle.*.*'))
transform = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(model_3, transform)

camera_bp = blueprint_library.find('sensor.camera.rgb')
relative_transform = carla.Transform(Location(x=0, y=0, z=3), Rotation(yaw=180))
camera = world.spawn_actor(camera_bp, relative_transform,attach_to=vehicle)
camera.listen(lambda image: image.save_to_disk('./output/%06d.png' % image.frame))
sleep(3)