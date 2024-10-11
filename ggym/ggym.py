import math
import warnings
from typing import TYPE_CHECKING, Optional, List, Tuple

import numpy as np

import gym
from Box2D.examples.simple.simple_01 import vertices
from gym import error, spaces
from gym.core import ObsType, ActType
from gym.error import DependencyNotInstalled
from gym.utils import EzPickle, colorize
from gym.utils.step_api_compatibility import step_api_compatibility
from numpy.distutils.fcompiler.none import NoneFCompiler

from group_24_gym.group_24_gym.envs.environment_2 import init_x

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")


if TYPE_CHECKING:
    import pygame
"""global value define"""
INITIAL_RANDOM = 5

# render parameters
FPS = 50
VIEWPORT_W = 600
VIEWPORT_H = 400
SCALE = 30.0

# physical parameters for the robot
HEIGHT_HULL = +4
WIDTH_HULL = +15
HULL_POLY = [(-WIDTH_HULL, +HEIGHT_HULL), (+WIDTH_HULL, +HEIGHT_HULL), (+WIDTH_HULL, -HEIGHT_HULL), (-WIDTH_HULL, -HEIGHT_HULL)]
wheel_radius = 2

# engine and motor
MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

MOTORS_TORQUE = 80

# terrain parameters
TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

# lidar parameters
LIDAR_RANGE = 160 / SCALE

"""physical simulation function"""
# fixture of the parts of the robot
HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0,
)  # 0.99 bouncy

WHEEL_FD = fixtureDef(
    shape=circleShape(radius=wheel_radius / SCALE, pos=(0, 0)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)


# physical contact
class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.hull == contact.fixtureA.body
            or self.env.hull == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.wheels[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.wheels[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.wheels[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.wheels[i].ground_contact = False


class Group24env(gym.Env,EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }
    def __init__(
            self,
            render_mode: Optional[str] = None,
            continuous: bool = False,
            gravity: float = -9.8,
            enable_wind: bool = False,
            wind_power: float = 15.0,
            turbulence_power: float = 1.5, # 0
            hardcore:bool = False,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            continuous,
            gravity,
            enable_wind,
            wind_power,
            turbulence_power,
            hardcore
        )
        self.isopen = True

        # world init
        assert (
                -9.81 <= gravity < 0.0
        ), f"gravity (current value: {gravity}) must be between -9.81 and 0"
        self.gravity = gravity

        self.world = Box2D.b2World()

        if 0.0 > wind_power or wind_power > 20.0:
            warnings.warn(
                colorize(
                    f"WARN: wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})",
                    "yellow",
                ),
            )
        self.wind_power = wind_power

        if 0.0 > turbulence_power or turbulence_power > 2.0:
            warnings.warn(
                colorize(
                    f"WARN: turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})",
                    "yellow",
                ),
            )
        self.turbulence_power = turbulence_power

        self.enable_wind = enable_wind
        self.wind_idx = np.random.randint(-9999, 9999)
        self.torque_idx = np.random.randint(-9999, 9999)

        # terrain init
        self.hardcore = hardcore
        self.terrain: List[Box2D.b2Body] = []

        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=FRICTION,
        )
        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )
        self.fd_triangle = fixtureDef(
            shape=polygonShape(vertices=[(0,0),(1,0),(0,1)]),
            firction=FRICTION,
        )

        # robot init
        self.hull: Optional[Box2D.b2Body] = None
        self.prev_shaping = None
        self.particles = []

        # action space
        self.action_space = spaces.Box(
            # two for engines, one for wheels
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            np.array([-1, -1, -1]).astype(np.float32),
            np.array([1, -1, 1]).astype(np.float32),
        )
        # observation space
        # we have a hull(main body), two wheels, three engines, ground contact
        low = np.array([
            # the parameters of the hull
            -math.pi, # minimum angular (orientation) of the robot
            -5.0, # minimum x-direction velocity the hull
            -5.0, # minimum y-direction velocity the hull
            -5.0, # minimum angular velocity of the hull
            -1.5, # minimum normalized x-position of the hull
            -1.5, # minimum normalized y-position of the hull

            # the parameters of the wheels
            -5.0, # minimum angular velocity of the wheels' joints

            # the ground contact listener
            -0.0, # minimum ground contact indicator for the left wheel (0 = no contact)
            -0.0 # minimum ground contact indicator for the right wheel (0 = no contact)
            ]+ [-1.0] * 10).astype(np.float32)

        high = np.array([
            # the parameters of the hull
            math.pi, # maximum angular (orientation) of the robot
            5.0, # maximum x-direction velocity the hull
            5.0, # maximum y-direction velocity the hull
            5.0, # maximum angular velocity of the hull
            1.5, # maximum normalized x-position of the hull
            1.5, # maximum normalized y-position of the hull

            # the parameters of the wheels
            5.0, # maximum angular velocity of the wheels' joints

            # the ground contact listener
            1.0, # maximum ground contact indicator for the left wheel (1 = contact)
            1.0 # maximum ground contact indicator for the right wheel (1 = contact)
            ]+ [1.0] * 10).astype(np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # reward init
        self.reward = None
        self.game_over = None
        # lidar init
        self.lidar_render = None

        self.render_mode = render_mode
        self.screen: Optional[pygame.Surface] = None
        self.clock = None

    # def destroy

    def _destroy(self):
        if not self.terrain:
            return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []

        self.world.DestroyBody(self.hull)
        self.hull = None
        for wheel in self.wheels:
            self.world.DestroyBody(wheel)
        for joint in self.joints:
            self.world.DestroyBody(joint)

        self._clean_particles(True)



    # terrain generation
    def _generate_terrain(self, hardcore):
        GRASS, STUMP, SLOPE, PIT, _STATES_ = range(5)
        state = GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []

        original_y = 0
        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD:
                    velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                y += velocity

            elif state == PIT and oneshot:
                counter = self.np_random.integers(3, 5)
                poly = [
                    (x, y),
                    (x + TERRAIN_STEP, y),
                    (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                    (x, y - 4 * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [
                    (p[0] + TERRAIN_STEP * counter, p[1]) for p in poly
                ]
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * TERRAIN_STEP

            elif state == STUMP and oneshot:
                counter = self.np_random.integers(1, 3)
                # counter = 1
                Y = self.np_random.integers(VIEWPORT_H/10,VIEWPORT_W/5)
                poly = [
                    (x, y),
                    (x + counter * TERRAIN_STEP, y),
                    (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP),
                    (x, y + counter * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

            elif state == SLOPE and oneshot:
                counter = self.np_random.integers(1, 3)
                counter = 1
                y = original_y
                up_or_down = self.np_random.integers(0,1)
                if up_or_down:
                    # up
                    poly = [
                        (x, y),
                        (x + counter * TERRAIN_STEP, y),
                        (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP),
                    ]
                else:
                    # down
                    poly = [
                        (x, y),
                        (x, y + counter * TERRAIN_STEP),
                        (x + counter * TERRAIN_STEP,y),
                    ]
                self.fd_triangle.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_triangle)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)


            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if state == GRASS and hardcore:
                    state = self.np_random.integers(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1]),
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            color = (76, 255 if i % 2 == 0 else 204, 76)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (102, 153, 76)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()


    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)
        self._destroy()

        # contact listener
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround

        # system reset
        self.game_over = False
        self.reward = 0

        # terrain resets
        self._generate_terrain(self.hardcore)

        # robot resets
        self.prev_shaping = None
        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * wheel_radius
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y), fixtures=HULL_FD
        )
        self.hull.color1 = (127, 51, 229)
        self.hull.color2 = (76, 76, 127)
        # initial force to wake the robot up
        self.hull.ApplyForceToCenter(
            (self.np_random.uniform(0, INITIAL_RANDOM), 0), True
        )

        self.wheels:List[Box2D.b2Body] = []
        self.joints:List[Box2D.b2RevoluteJoint] = []

        for i in [-1,1]:
            wheel = self.world.CreateDynamicBody(
                position = ((init_x+i*(WIDTH_HULL/(2*SCALE))),(init_y - (HEIGHT_HULL/(2*SCALE))-(wheel_radius/SCALE))),
                fixtures = WHEEL_FD,
            )
            wheel.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            wheel.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA = self.hull,
                bodyB = wheel,
                enableMotor = True,
                maxMotorTorque = 100,
                motorSpeed = 1,
                
            )

        # lidar resets
        self.lidar_render = 0
        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction

        self.lidar = [LidarCallback() for _ in range(10)]

    # generate particle
    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    # clean particle
    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action: ActType):

    def render(self, screen):

    # def close pygame

if __name__=="__main__":


