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
        rayCastCallback,
        weldJointDef,
        wheelJointDef,
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

HALF_HEIGHT_HULL = 10
HALF_WIDTH_HULL = 40
HALF_HEIGHT_LANDER = 8
HALF_WIDTH_LANDER = 20
WHEEL_RADIUS = 34 / 2   # half of the LEG_H

HULL_POLY = [(-HALF_WIDTH_HULL, HALF_HEIGHT_HULL), (HALF_WIDTH_HULL, HALF_HEIGHT_HULL), (HALF_WIDTH_HULL, -HALF_HEIGHT_HULL), (-HALF_WIDTH_HULL, -HALF_HEIGHT_HULL)]
LANDER_POLY = [(-HALF_WIDTH_LANDER, HALF_HEIGHT_LANDER), (HALF_WIDTH_LANDER, HALF_HEIGHT_LANDER), (HALF_WIDTH_LANDER, -HALF_HEIGHT_LANDER), (-HALF_WIDTH_LANDER, -HALF_HEIGHT_LANDER)]

wheel_radius = 2
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

# engine and motor
MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6

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
    shape=circleShape(radius=WHEEL_RADIUS / SCALE, pos=(0, 0)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

LEG_FD = fixtureDef(
    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

LANDER_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
    density=1.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0,
)  # 0.99 bouncy


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
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True

    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False


class Group24env(gym.Env,EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }
    def __init__(
            self,
            render_mode: Optional[str] = None,
            continuous: bool = True,
            gravity: float = -9.8,
            enable_wind: bool = False,
            wind_power: float = 15.0,
            turbulence_power: float = 0, # 0
            hardcore:bool = True,
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
        self.continuous = continuous
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
        self.scroll = None

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
        self.particles = []

        # action space
        self.action_space = spaces.Box(
            # two for engines, one for wheels, one for legs
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            np.array([-1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 0]).astype(np.float32),
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

            # the parameters of the wheels and legs
            -5.0, # minimum angular velocity of the wheels' joints
            -math.pi/2, # minimum angle position of the leg

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
            0,  # maximum angle position of the leg

            # the ground contact listener
            1.0, # maximum ground contact indicator for the left wheel (1 = contact)
            1.0 # maximum ground contact indicator for the right wheel (1 = contact)
            ]+ [1.0] * 10).astype(np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)


        # reward init
        self.game_over = None
        self.prev_shaping = None


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

        for lander in self.lander:
            self.world.DestroyBody(lander)
        self.lander = []
        self.lander_joints = []

        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []

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

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly = []
        for i in range(TERRAIN_LENGTH // 20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH) * TERRAIN_STEP
            y = VIEWPORT_H / SCALE * 3 / 4
            poly = [
                (
                    x
                    + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5)
                    + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                    y
                    + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5)
                    + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                )
                for a in range(5)
            ]
            x1 = min(p[0] for p in poly)
            x2 = max(p[0] for p in poly)
            self.cloud_poly.append((poly, x1, x2))

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
        self.scroll = 0
        self.prev_shaping = None

        # terrain resets
        self._generate_terrain(self.hardcore)
        # cloud reset
        self._generate_clouds()

        # robot resets
        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y), fixtures=HULL_FD
        )
        self.hull.color1 = (127, 51, 229)
        self.hull.color2 = (76, 76, 127)

        # initial force to wake the robot up
        self.hull.ApplyForceToCenter(
            (self.np_random.uniform(0, INITIAL_RANDOM), 0), True
        )

        self.lander:List[Box2D.b2Body] = []
        self.lander_joints:List[Box2D.b2WeldJoint] = []
        lander = self.world.CreateDynamicBody(
            position=(init_x, init_y - (HALF_HEIGHT_HULL - HALF_HEIGHT_LANDER) / SCALE), fixtures=LANDER_FD
        )
        lander.color1 = (98, 149, 132)
        lander.color2 = (36, 54, 66)
        weldjd = weldJointDef(
            bodyA=self.hull,
            bodyB=lander,
            localAnchorA=(0, -(HALF_HEIGHT_HULL) / SCALE),
            localAnchorB=(0, (HALF_HEIGHT_LANDER / SCALE)),
            referenceAngle=0.0,
            frequencyHz=0.0,
            dampingRatio=0.0
        )
        lander.ground_contact = False
        self.lander.append(lander)
        self.lander_joints.append(self.world.CreateJoint(weldjd))


        self.legs:List[Box2D.b2Body] = []
        self.joints:List[Box2D.b2RevoluteJoint] = []
        for i in [-1,1]:
            leg = self.world.CreateDynamicBody(
                position=(init_x + i * (HALF_WIDTH_HULL / SCALE), init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LEG_FD,
            )
            leg.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            leg.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA = self.hull,
                bodyB = leg,
                localAnchorA=(i * (HALF_WIDTH_HULL / SCALE), LEG_DOWN),
                localAnchorB = (0,LEG_H / 2),
                enableMotor = True,
                enableLimit = True,
                maxMotorTorque = MOTORS_TORQUE,
                motorSpeed = i, # the direction of the rotation
                upperAngle = 1.57,
                lowerAngle = -1.57
            )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            wheel = self.world.CreateDynamicBody(
                position=(init_x + i * (HALF_WIDTH_HULL / SCALE), init_y - WHEEL_RADIUS / SCALE - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=WHEEL_FD,
            )
            wheel.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            wheel.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            wheeljd = wheelJointDef(
                bodyA=leg,
                bodyB=wheel,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, 0),
                # localAxisA=(1, 0),  # horizontal slide
                enableMotor=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=-1, # clockwise rotation, robot moves forwards
                frequencyHz=4.0,
                dampingRatio=0.7,
            )  # frequencyHz & dampingRatio 可能要调整
            wheel.ground_contact = False
            self.legs.append(wheel)
            self.joints.append(self.world.CreateJoint(wheeljd))

        self.drawlist = self.terrain + self.legs + [self.hull] + self.lander

        # lidar resets
        self.lidar_render = 0

        class LidarCallback(rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction

        self.lidar = [LidarCallback() for _ in range(10)]

        # render reset
        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0, 0, 0]))[0], {}

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

    def step(self, action): # the action matches with action_space
        # ensure the robot was created successfully in reset()
        assert self.hull and self.lander is not None

        # winding and turbulence simulation
        if self.enable_wind and not (
                self.legs[1].ground_contact or self.legs[3].ground_contact
        ):
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi*0.01*self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.hull.ApplyForceToCenter((wind_mag, 0.0 ),True)
            torque_mag = math.tanh(
                math.sin(0.02 * self.torque_idx)
                + (math.sin(math.pi * 0.01 * self.torque_idx))
            ) * (self.turbulence_power)
            self.torque_idx += 1
            self.hull.ApplyTorque(
                torque_mag,True
            )

        # action processing
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "


        # engine control
        tip = (math.sin(self.hull.angle), math.cos(self.hull.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        # main engines
        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
                not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            # 4 is move a bit downwards, +-2 for randomness
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.hull.position[0] + ox, self.hull.position[1] + oy)

            # main engine particles create
            p = self._create_particle(
                3.5,  # 3.5 is here to make particle speed adequate
                impulse_pos[0],
                impulse_pos[1],
                m_power,
            )  # particles are just a decoration
            p.ApplyLinearImpulse(
                (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

            self.hull.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        # side engines
        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
                not self.continuous and action in [1, 3]
        ):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0

            # side engines particles create
            ox = tip[0] * dispersion[0] + side[0] * (
                    3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                    3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            impulse_pos = (
                self.hull.position[0] + ox - tip[0] * 17 / SCALE,
                self.hull.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )

            # create particles
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(
                (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

            self.hull.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        # wheel control
        self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[2]))
        self.joints[1].maxMotorTorque = float(
            MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1)
        )

        self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[2]))
        self.joints[3].maxMotorTorque = float(
            MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1)
        )
        # leg control
        self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[3]))
        self.joints[0].maxMotorTorque = float(
            MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1)
        )

        self.joints[2].moterSpeed = float(SPEED_HIP * np.sign(action[3]))
        self.joints[2].maxMotorTorque = float(
            MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1)
        )
        # lidar
        pos = self.hull.position
        vel = self.hull.linearVelocity
        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        # state
        state = [
            # the angle of the hull
            self.hull.angle,
            # normalized x-direction linear velocity of the hull
            vel.x * (VIEWPORT_W/SCALE)/FPS,
            # normalized y-direction linear velocity of the hull
            vel.y * (VIEWPORT_H/SCALE)/FPS,
            # the angular velocity of the hull
            20.0 * self.hull.angularVelocity/FPS,
            # the normalized x-position of the hull
            (pos.x - VIEWPORT_W/SCALE/2)/(VIEWPORT_W/SCALE/2),
            # the normalized y-position of the hull
            (pos.y - VIEWPORT_H/SCALE/2)/(VIEWPORT_H/SCALE/2),
            # wheels' speed and rotation direction
            self.joints[0].motorSpeed,
            # the angle of the legs
            self.joints[1].angle,
            # wheels ground contact
            1.0 if self.legs[1].ground_contact else 0.0,
            1.0 if self.legs[3].ground_contact else 0.0,
        ]
        # add the lidar readings in state
        state += [l.fraction for l in self.lidar]
        # reward
        reward = 0
        # get reward for the forward movement
        shaping = (
                130 * pos[0] / SCALE
        )  # moving forward is a way to receive reward (normalized to get 300 on completion)
        # when flying, the robot should be balanced as soon as possible
        if self.joints[1].ground_contact == 0 and self.joints[3].ground_contact == 0:
            if np.abs(state[0]) <= 1.57 / 3:
                shaping += (
                        + 10 * state[8]
                        + 10 * state[9]
                        + 5 * (1.57 / 3 - np.abs(state[0]))
                )
            else:
                # unbalance control
                shaping += (
                        + 10 * state[8]
                        + 10 * state[9]
                        - 5 * (np.abs(state[0])-1.57/3)
                )
        else:
            shaping -= 0

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        # penalty
        # engine fuel use
        reward -= (
                m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03
        # motor use
        reward -= 2 * 0.00035 * MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1)

        # termination condition
        terminated = False
        if self.game_over or pos[0] < 0:
            reward -= -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP: # reach the target
            terminated = True

        # rendering and return
        self.scroll = pos.x - VIEWPORT_W/SCALE/5
        if self.render_mode == "human":
            self.render()
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def render(self, screen):


    # def close pygame

if __name__=="__main__":


