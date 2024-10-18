__credits__ = ["Group24"]

import math
import warnings
from typing import TYPE_CHECKING, List, Optional

import numpy as np

import gym
from gym import error, spaces
from gym.error import DependencyNotInstalled
from gym.utils import EzPickle, colorize

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
        weldJointDef,
    )
except ImportError:
    raise DependencyNotInstalled("box2D is not installed, run `pip install gym[box2d]`")

if TYPE_CHECKING:
    import pygame

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 600
SPEED_HIP = 10
SPEED_KNEE = 10
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

# physical parameters for the robot
HALF_HEIGHT_HULL = 10
HALF_WIDTH_HULL = 40
HALF_HEIGHT_LANDER = 6
HALF_WIDTH_LANDER = 20
WHEEL_RADIUS = 34 / 2

HULL_POLY = [(-HALF_WIDTH_HULL, HALF_HEIGHT_HULL), (HALF_WIDTH_HULL, HALF_HEIGHT_HULL),
             (HALF_WIDTH_HULL, -HALF_HEIGHT_HULL), (-HALF_WIDTH_HULL, -HALF_HEIGHT_HULL)]
LANDER_POLY = [(-HALF_WIDTH_LANDER, HALF_HEIGHT_LANDER), (HALF_WIDTH_LANDER, HALF_HEIGHT_LANDER),
               (HALF_WIDTH_LANDER, -HALF_HEIGHT_LANDER), (-HALF_WIDTH_LANDER, -HALF_HEIGHT_LANDER)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 40 / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 25  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

# engine physical parameters
LITATION_COEFFICIENT = 243
QUANTITY_OUTLINE_ADAPTOR = 0.1  # cm -> m
ENGINE_POWER_FACTOR = 53526
throttle_increase_rate = 0.01
RPS = 2000

HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0,
)  # 0.99 bouncy

LEG_FD = fixtureDef(
    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

WHEEL_FD = fixtureDef(
    shape=circleShape(radius=WHEEL_RADIUS / SCALE, pos=(0, WHEEL_RADIUS / SCALE)),
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
        for lander in [self.env.lander[0]]:
            if lander in [contact.fixtureA.body, contact.fixtureB.body]:
                lander.ground_contact = True

    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False
        for lander in [self.env.lander[0]]:
            if lander in [contact.fixtureA.body, contact.fixtureB.body]:
                lander.ground_contact = False


class Group24(gym.Env, EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self,
                 render_mode: Optional[str] = None,
                 continuous: bool = True,
                 gravity: float = -9.8,
                 enable_wind: bool = False,
                 wind_power: float = 15.0,
                 turbulence_power: float = 0,
                 hardcore: bool = True,
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

        self.world = Box2D.b2World()
        self.terrain: List[Box2D.b2Body] = []
        self.hull: Optional[Box2D.b2Body] = None

        self.prev_shaping = None

        self.hardcore = hardcore

        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=FRICTION,
        )

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

        # robot init
        self.hull: Optional[Box2D.b2Body] = None
        self.prev_shaping = None
        self.particles = []

        # engine init
        self.throttle_left = 0.0
        self.throttle_right = 0.0

        # observation space (v1: modify by zewen) (v2: modify by ziyue, not sure horizontal position and vertical position are needed, and need (-99999, 99999))
        # hull: angle, angular velocity horizontal speed, horizontal speed, vertical speed, horizontal position, vertical position
        # two legs: angle, joints angular speed
        # two wheels: joints angular speed, contact with ground
        # lander: contact with ground
        # lidar: 20 rangefinder measurements
        low = np.array([
                           -math.pi, -5.0, -5.0, -5.0, -99999, -99999,
                           -math.pi, -5.0,
                           -5.0, 0,
                           -math.pi, -5.0,
                           -5.0, 0,
                           0
                       ] + [-1.0] * 20).astype(np.float32)

        high = np.array([
                            math.pi, 5.0, 5.0, 5.0, 99999, 99999,
                            math.pi, 5.0,
                            5.0, 1,
                            math.pi, 5.0,
                            5.0, 1,
                            1
                        ] + [1.0] * 20).astype(np.float32)

        # action space (v1: modify by zewen) (v2: modify by ziyue)
        # rear leg joint motor speed, front wheel motor speed, back wheel motor speed
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 1]).astype(np.float32),
        )
        self.observation_space = spaces.Box(low, high)

        # reward init
        self.reward = None
        self.game_over = None

        # lidar init
        self.lidar_render = None

        self.render_mode = render_mode
        self.screen: Optional[pygame.Surface] = None
        self.clock = None

        self.render_mode = render_mode
        self.screen: Optional[pygame.Surface] = None
        self.clock = None

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
        GRASS, TOWER, STAIRS, SLOPE, HOLE, _STATES_ = range(6)
        state = GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []

        stair_steps, stair_width, stair_height = 0, 0, 0
        original_y = 0

        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD:
                    velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                y += velocity

            elif state == TOWER and oneshot:
                robot_width = (HALF_WIDTH_HULL + WHEEL_RADIUS) * 2 / SCALE + LEG_H * 2
                robot_height = ((HALF_HEIGHT_HULL * 2 + WHEEL_RADIUS) / SCALE + LEG_H)
                counterx = self.np_random.integers(robot_width, robot_width * 2.5)
                countery = self.np_random.integers(VIEWPORT_H / SCALE - robot_height * 2 + HALF_HEIGHT_HULL / SCALE,
                                                   VIEWPORT_H / SCALE - robot_height)
                poly = [
                    (x, y),
                    (x + counterx * TERRAIN_STEP, y),
                    (x + counterx * TERRAIN_STEP, y + countery * TERRAIN_STEP),
                    (x, y + countery * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                counter = counterx

            elif state == STAIRS and oneshot:
                stair_height = +2 if self.np_random.random() > 0.5 else -2
                stair_width = self.np_random.integers(4, 5)
                stair_steps = self.np_random.integers(2, 4)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (
                            x + (s * stair_width) * TERRAIN_STEP,
                            y + (s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + ((1 + s) * stair_width) * TERRAIN_STEP,
                            y + (s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + ((1 + s) * stair_width) * TERRAIN_STEP,
                            y + (-2 + s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + (s * stair_width) * TERRAIN_STEP,
                            y + (-2 + s * stair_height) * TERRAIN_STEP,
                        ),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                    t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                    self.terrain.append(t)
                counter = stair_steps * stair_width

            elif state == STAIRS and not oneshot:
                s = stair_steps * stair_width - counter - stair_height
                n = s / stair_width
                y = original_y + (n * stair_height) * TERRAIN_STEP

            elif state == SLOPE and oneshot:
                robot_width = (HALF_WIDTH_HULL + WHEEL_RADIUS) * 2 / SCALE + LEG_H * 2
                slope_width = self.np_random.integers(robot_width, robot_width * 2.5)
                slope_angle = self.np_random.integers(10, 35)
                slope_height = slope_width * np.tan(np.deg2rad(slope_angle))
                poly = [
                    (x, y),
                    (x + slope_width * TERRAIN_STEP, y + slope_height * TERRAIN_STEP),
                    (x + (slope_width + robot_width * 1.5) * TERRAIN_STEP, y + slope_height * TERRAIN_STEP),
                    (x + (2 * slope_width + robot_width * 1.5) * TERRAIN_STEP, y),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                counter = np.ceil((2 * slope_width + robot_width * 1.5))

            elif state == HOLE and oneshot:
                robot_width = (HALF_WIDTH_HULL + WHEEL_RADIUS) * 2 / SCALE + LEG_H * 2
                robot_height = ((HALF_HEIGHT_HULL * 2 + WHEEL_RADIUS) / SCALE + LEG_H)
                counter = self.np_random.integers(robot_width, 3 * robot_width)
                hole_height = self.np_random.integers((VIEWPORT_H / SCALE - y) - robot_height,
                                                      (VIEWPORT_H / SCALE - y) - robot_height / 3)
                poly = [
                    (x, VIEWPORT_H / SCALE),
                    (x + counter * TERRAIN_STEP, VIEWPORT_H / SCALE),
                    (x + counter * TERRAIN_STEP, VIEWPORT_H / SCALE - hole_height),
                    (x, VIEWPORT_H / SCALE - hole_height),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
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
        super().reset(seed=seed)
        self._destroy()

        # contact listener
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround

        # system reset
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        # terrain resets
        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        # engine init
        self.throttle_left = 0.0
        self.throttle_right = 0.0

        # robot resets
        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y), fixtures=HULL_FD
        )
        self.hull.color1 = (127, 51, 229)
        self.hull.color2 = (76, 76, 127)
        # apply a random initial force to the hull in the physics simulation
        # self.hull.ApplyForceToCenter(
        #     (self.np_random.uniform(0, 0), 0), True
        # )

        # lander creation
        self.lander: List[Box2D.b2Body] = []
        self.lander_joints: List[Box2D.b2WeldJoint] = []
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

        self.legs: List[Box2D.b2Body] = []
        self.joints: List[Box2D.b2RevoluteJoint] = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(init_x + i * (HALF_WIDTH_HULL / SCALE), init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LEG_FD,
            )
            leg.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            leg.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            if i == -1:
                rjd_lowerAngle = -np.pi
                rjd_upperAngle = 0
            else:
                rjd_lowerAngle = 0
                rjd_upperAngle = np.pi
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(i * (HALF_WIDTH_HULL / SCALE), LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=i,
                lowerAngle=rjd_lowerAngle,
                upperAngle=rjd_upperAngle,
            )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            wheel = self.world.CreateDynamicBody(
                position=(init_x + i * (HALF_WIDTH_HULL / SCALE),
                          init_y - (HALF_HEIGHT_HULL / SCALE) - LEG_H - WHEEL_RADIUS / SCALE),
                angle=(i * 0.05),
                fixtures=WHEEL_FD,
            )
            wheel.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            wheel.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=wheel,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, WHEEL_RADIUS / SCALE),
                enableMotor=True,
                enableLimit=False,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=1,
            )
            wheel.ground_contact = False
            self.legs.append(wheel)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.legs + [self.hull] + self.lander

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction

        self.lidar = [LidarCallback() for _ in range(20)]
        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0, 0, 0]))[0], {}

    # generate particle (ttl: Time To Live)
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

    def step(self, action: np.ndarray):
        assert self.hull is not None
        fly_check = (self.legs[2].angle - self.hull.angle) > 3 and (self.legs[0].angle - self.hull.angle) < -3
        # UAS mode
        if fly_check:
            print("fly enable")
            # Rear leg control
            self.joints[0].motorSpeed = float(
                SPEED_HIP * np.clip(action[0], -1, -1)
            )
            self.joints[0].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1)
            )
            # Front leg control
            self.joints[2].motorSpeed = float(
                SPEED_HIP * np.clip(action[2], 1, 1)
            )
            self.joints[2].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1)
            )
            # engine parameters setting
            # direction of engines' impulse
            tip = (-math.sin(self.hull.angle), math.cos(self.hull.angle))

            # particle power
            p_power_left = (np.clip(action[1], 0, 1) + 1.0) * 0.5
            p_power_right = (np.clip(action[3], 0, 1) + 1.0) * 0.5

            # left wheel's engine
            blade_angular_speed_left = -float(SPEED_KNEE * np.clip(action[1], -1, 0))
            impulse_magnitude_left = LITATION_COEFFICIENT * pow(blade_angular_speed_left, 2) * pow(
                QUANTITY_OUTLINE_ADAPTOR, 3)

            print("blade_angular_speed_left", blade_angular_speed_left)
            print("impulse_magnitude_left",impulse_magnitude_left)

            # impulse pos left
            impulse_pos_left = \
                (self.legs[0].position[0],
                 self.legs[0].position[1])

            # impulse left
            impulse_left = (impulse_magnitude_left * tip[0], impulse_magnitude_left * tip[1])

            # right wheel's engine
            blade_angular_speed_right = -float(SPEED_KNEE * np.clip(action[3], -1, 0))
            impulse_magnitude_right = LITATION_COEFFICIENT * pow(blade_angular_speed_right, 2) * pow(
                QUANTITY_OUTLINE_ADAPTOR, 3)

            print("blade_angular_speed_right", blade_angular_speed_right)
            print("impulse_magnitude_right", impulse_magnitude_right)


            # impulse pos right
            impulse_pos_right = \
                (self.legs[2].position[0],
                 self.legs[2].position[1])

            # impulse right
            impulse_right = (impulse_magnitude_right * tip[0], impulse_magnitude_right * tip[1])

            # During each step, gradually increase the throttle
            self.throttle_left = min(self.throttle_left + throttle_increase_rate, 1.0)
            self.throttle_right = min(self.throttle_right + throttle_increase_rate, 1.0)

            # Calculate the scaled impulse using the throttle levels
            scaled_impulse_left = (
                impulse_left[0] * self.throttle_left,
                impulse_left[1] * self.throttle_left
            )
            scaled_impulse_right = (
                impulse_right[0] * self.throttle_right,
                impulse_right[1] * self.throttle_right
            )

            # engine activates
            # create particles left
            p_left = self._create_particle(
                3.5,
                impulse_pos_left[0] - LEG_H*tip[0],
                impulse_pos_left[1] - LEG_H*tip[1],
                p_power_left,
            )

            # apply impulse to particle
            p_left.ApplyLinearImpulse(
                (-scaled_impulse_left[0], -scaled_impulse_left[1]),
                impulse_pos_left,
                True
            )

            # apply impulse to the left leg
            self.legs[0].ApplyLinearImpulse(
                scaled_impulse_left,
                impulse_pos_left,
                True
            )

            # create particles right
            p_right = self._create_particle(
                3.5,
                impulse_pos_right[0] - LEG_H * tip[0],
                impulse_pos_right[1] - LEG_H * tip[1],
                p_power_right,
            )

            # apply impulse to particle
            p_right.ApplyLinearImpulse(
                (-scaled_impulse_right[0], -scaled_impulse_right[1]),
                impulse_pos_right,
                True
            )

            # apply impulse to the right leg
            self.legs[2].ApplyLinearImpulse(
                scaled_impulse_right,
                impulse_pos_right,
                True
            )

        # Other modes
        else:
            rear_leg_speed = self.joints[0].speed / SPEED_HIP
            front_leg_speed = self.joints[2].speed / SPEED_HIP
            leg_damping = 0.15

            # Rear leg control
            self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]) - leg_damping * rear_leg_speed)
            self.joints[0].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1)
            )
            # Rear wheel control
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1)
            )
            # Front leg control
            self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]) - leg_damping * front_leg_speed)
            self.joints[2].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1)
            )
            # Front wheel control
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1)
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(20):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(np.pi * i / 20.0) * LIDAR_RANGE,
                pos[1] - math.cos(np.pi * i / 20.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        
        state = [
            self.hull.angle,
            2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            0.3 * pos.x * (VIEWPORT_W / SCALE) / FPS,
            0.3 * pos.y * (VIEWPORT_H / SCALE) / FPS,
            self.joints[0].angle,
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0,
            1.0 if self.lander[0].ground_contact else 0.0,
        ]
        state += [l.fraction for l in self.lidar]
        assert len(state) == 35

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5

        shaping = (
                130 * pos[0] / SCALE
        )  # moving forward is a way to receive reward
        shaping -= 5.0 * abs(
            state[0]
        )  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            if self.legs[3].ApplyLinearImpulse == True and self.legs[1].ApplyLinearImpulse == True:
                # flying energy consumption
                reward -= 0.035 * ENGINE_POWER_FACTOR * pow(blade_angular_speed_left, 2) * pow(QUANTITY_OUTLINE_ADAPTOR,
                                                                                               2)
                # the balance the hull when flying
                reward -= 10 * (np.abs(self.hull.angle) - 30)

        terminated = False
        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = True
        if self.game_over or pos[1] > (VIEWPORT_H / SCALE + LEG_H):
            reward = -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True

        if self.render_mode == "human":
            self.render()
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def render(self):
        # Check the render mode
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        # Import the pygame library
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[box2d]`"
            )

        # Initialize the screen and clock
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Create a Rendering Surface
        self.surf = pygame.Surface(
            (VIEWPORT_W + max(0.0, self.scroll) * SCALE, VIEWPORT_H)
        )

        pygame.transform.scale(self.surf, (SCALE, SCALE))

        # Draw a light blue background
        pygame.draw.polygon(
            self.surf,
            color=(215, 215, 255),
            points=[
                (self.scroll * SCALE, 0),
                (self.scroll * SCALE + VIEWPORT_W, 0),
                (self.scroll * SCALE + VIEWPORT_W, VIEWPORT_H),
                (self.scroll * SCALE, VIEWPORT_H),
            ],
        )

        # Draw the cloud
        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2:
                continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE:
                continue
            pygame.draw.polygon(
                self.surf,
                color=(255, 255, 255),
                points=[
                    (p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly
                ],
            )
            gfxdraw.aapolygon(
                self.surf,
                [(p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly],
                (255, 255, 255),
            )

        # Draw the terrain
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE:
                continue
            scaled_poly = []
            for coord in poly:
                scaled_poly.append([coord[0] * SCALE, coord[1] * SCALE])
            pygame.draw.polygon(self.surf, color=color, points=scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, color)

        # Plotting LIDAR sensor data
        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if i < 2 * len(self.lidar):
            single_lidar = (
                self.lidar[i]
                if i < len(self.lidar)
                else self.lidar[len(self.lidar) - i - 1]
            )
            if hasattr(single_lidar, "p1") and hasattr(single_lidar, "p2"):
                pygame.draw.line(
                    self.surf,
                    color=(255, 0, 0),
                    start_pos=(single_lidar.p1[0] * SCALE, single_lidar.p1[1] * SCALE),
                    end_pos=(single_lidar.p2[0] * SCALE, single_lidar.p2[1] * SCALE),
                    width=1,
                )

        # Update the color and lifespan of particles
        for obj in self.particles:
            obj.ttl -= 0.5
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )

        self._clean_particles(False)

        # Draw robots, particles and other objects
        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    if len(path) > 2:
                        pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                        gfxdraw.aapolygon(self.surf, path, obj.color1)
                        path.append(path[0])
                        pygame.draw.polygon(
                            self.surf, color=obj.color2, points=path, width=1
                        )
                        gfxdraw.aapolygon(self.surf, path, obj.color2)
                    else:
                        pygame.draw.aaline(
                            self.surf,
                            start_pos=path[0],
                            end_pos=path[1],
                            color=obj.color1,
                        )

        # draw target flag
        flagy1 = TERRAIN_HEIGHT * SCALE
        flagy2 = flagy1 + 50
        x = TERRAIN_STEP * 3 * SCALE
        pygame.draw.aaline(
            self.surf, color=(0, 0, 0), start_pos=(x, flagy1), end_pos=(x, flagy2)
        )
        f = [
            (x, flagy2),
            (x, flagy2 - 10),
            (x + 25, flagy2 - 5),
        ]
        pygame.draw.polygon(self.surf, color=(230, 51, 0), points=f)
        pygame.draw.lines(
            self.surf, color=(0, 0, 0), points=f + [f[0]], width=1, closed=False
        )

        # flip and display the image
        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (-self.scroll * SCALE, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )[:, -VIEWPORT_W:]

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class BipedalWalkerHardcore:
    def __init__(self):
        raise error.Error(
            "Error initializing BipedalWalkerHardcore Environment.\n"
            "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
            "To use this environment, instead create it by specifying the hardcore keyword in gym.make, i.e.\n"
            'gym.make("BipedalWalker-v3", hardcore=True)'
        )


if __name__ == "__main__":
    # Initialize the environment with render_mode set to 'human' for real-time visualization
    env = Group24(render_mode="human")
    env.reset()

    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])  # Initial action
    UGV, CROUCHING, STOP, UAS = 1, 2, 3, 4
    state = UGV

    while True:
        # Step through the environment and render each step
        s, r, terminated, truncated, info = env.step(a)
        total_reward += r

        # Output the step details every 20 steps
        if steps % 20 == 0 or terminated or truncated:
            print(f"\naction {a}")
            print(f"step {steps} total_reward {total_reward:+0.2f}")
            print(f"hull {s[0:6]}")
            print(f"back leg {s[6:8]}")
            print(f"back wheel {s[8:10]}")
            print(f"front leg {s[10:12]}")
            print(f"front wheel {s[12:14]}")
            print(f"lander {s[14]}")

        steps += 1

        # Update the logic for controlling the walker
        # (control logic remains unchanged from the provided script)

        leg_targ = [None, None]
        wheel_targ = [None, None]
        leg_todo = [0.0, 0.0]
        wheel_todo = [0.0, 0.0]

        if state == UGV:
            wheel_targ[0] = -1.0
            wheel_targ[1] = -1.0
            leg_targ[0] = np.pi / 4
            leg_targ[1] = -np.pi / 4
        if state == CROUCHING:
            wheel_targ[0] = -0.01
            wheel_targ[1] = -0.01
            leg_targ[0] = -np.pi / 4
            leg_targ[1] = np.pi / 4
        if state == STOP:
            wheel_targ[0] = -0.001
            wheel_targ[1] = 0.001
            leg_targ[0] = np.pi / 4
            leg_targ[1] = -np.pi / 4
        if state == UAS:
            # make sure the lander is on the ground
            if steps < 150:
                wheel_targ[0] = 0
                wheel_targ[1] = 0
            else:
                if steps % 20 == 0 or terminated or truncated:
                    print("UAS mode")
                wheel_targ[0] = -0.01
                wheel_targ[1] = -0.01
            leg_targ[0] = -np.pi
            leg_targ[1] = np.pi

        if leg_targ[0]:
            leg_todo[0] = 0.9 * (leg_targ[0] - s[6]) - 0.25 * s[7]
        if leg_targ[1]:
            leg_todo[1] = 0.9 * (leg_targ[1] - s[10]) - 0.25 * s[11]
        if wheel_targ[0]:
            wheel_todo[0] = 4.0 * (wheel_targ[0] - s[8])
        if wheel_targ[1]:
            wheel_todo[1] = 4.0 * (wheel_targ[1] - s[12])

        # PID control to keep head straight and reduce oscillations
        leg_todo[0] -= 3 * (0 - s[0]) - 1.5 * s[1]
        leg_todo[1] -= 3 * (0 - s[0]) - 1.5 * s[1]
        wheel_todo[0] -= 10 * s[3]
        wheel_todo[1] -= 10 * s[3]

        # Update the action
        a[0] = leg_todo[0]
        a[1] = wheel_todo[0]
        a[2] = leg_todo[1]
        a[3] = wheel_todo[1]
        a = np.clip(0.5 * a, -1.0, 1.0)
        if steps % 20 == 0 or terminated or truncated:
            print("action_main",a)
        if terminated or truncated:
            break

    env.close()

