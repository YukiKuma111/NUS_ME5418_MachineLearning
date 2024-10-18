# ME5418_ML

## Steps to Set Up

### Step1: Environment configuration

```
mkdir your_workspace
cd your_workspace/
git clone https://github.com/YukiKuma111/ME5418_ML.git
cd ME5418_ML/
conda env create -f environment.yaml

# test env
python m4_env.py
```

<!-- ## zewen:


## rui:


## ziyue:

Question:

    - 应该一个类似于时间限制的控制器？如果小车一段时间内都没有产生移动，就会kill掉这个episode

### 2024.10.17:

- environment.yaml

    1. 发现缺少gym库，目前已添加其和其依赖，但是还需等孙老师进行测试

- class ContactDetector：

    1. run时发现lander的触底检测无效，现已修复

- initial part:

    1. 修改了几个参数，但是不确定是否正确
        MOTORS_TORQUE = 400 可能还要改，但是不可以再小了，算出来最小360;而且轮子速度不能很快，不然站不起来
        SPEED_HIP = 10
        TERRAIN_GRASS = 25  增长obstacles之间的缓冲地面，防止没有地方变形 （孙老师提出的问题）
        LEG_W, LEG_H = 8 / SCALE, 40 / SCALE 缩短了 LEG_H
    2. 加入了孙老师定义的UAS相关参数

- __init__:

    1. 多增加了几各文件，分别为
    “主控后退”  m4_env_1-rearleg.py
    “主控前腿”  m4_env_1-frontleg.py
    “主控两腿无飞行”    m4_env_wo-fly.py
    “主控两腿有飞行但需debug”   m4_env.py
    “孙老师写的源码”    g24_env.py
    2. 我把hardcore关掉了，先用于测试！记得上传前开开！

- _generate_terrain：

    1. 增加了楼梯可视化的高度（之前只改变了位置高度，没有调整poly的高）
    2. 发现存在obstacles重叠的情况。现大概已修复？

- step:

    1. 增加了阻尼，但是不知道是否有效果。。。
    2. 加入了孙老师写的UAS部分，但是有需要调整的地方。比如气体会随着轮子转动，力的大小等。
    3. 修改了孙老师的原版的逻辑判断，但是有可能还需要修改

- __name__ == "__main__":

    1. 注释了源码的state判断，直接写轮子转速
    2. 写了4个运动形态：STOP，UGV，UAS，CROUCHING，目前除了UAS都可以运行

    - if state == UGV:  任意给一个大于0的leg_targ就可以
        wheel_targ[0] = -1.0
        wheel_targ[1] = -1.0
        leg_targ[0] = np.pi / 4
        leg_targ[1] = np.pi / 4
    action [ 0.44878279 -0.00321004  0.00504127]
    step 720 total_reward +179.51
    hull [-1.6950037e-03  1.8976265e-06  4.0706563e-01  2.1003921e-06
    5.9721370e+00  4.6707901e-01]
    back leg [3.498923e-03 3.394467e-05]
    back wheel [-0.99838305  0.        ]
    front leg [-1.4989525e-03 -7.5649587e-07]
    front wheel [-1.0023984  1.       ]
    lander 0.0

    - if state == STOP: 同上，任意给一个大于0的leg_targ就可以，设置一个较小的wheel_targ
        wheel_targ[0] = -0.01
        wheel_targ[1] = -0.01
        leg_targ[0] = np.pi / 4
        leg_targ[1] = np.pi / 4
    action [ 0.45284462 -0.02000004 -0.02000004]
    step 1220 total_reward -18.03
    hull [ 3.3692137e-04 -7.8427037e-10 -5.0731908e-09 -1.1295026e-09
    5.5201554e-01  4.6696785e-01]
    back leg [4.7217458e-03 5.7731597e-15]
    back wheel [2.7022175e-08 1.0000000e+00]
    front leg [-1.6690657e-02 -4.4408921e-16]
    front wheel [2.7417768e-08 0.0000000e+00]
    lander 0.0

    - if state == UAS:  暂时还没有转轮子产生风；后腿leg_targ在-np.pi
        wheel_targ[0] = -0.01
        wheel_targ[1] = -0.01
        leg_targ[0] = -np.pi
        leg_targ[1] = -np.pi
    action [-1. -1. -1.]
    step 3440 total_reward -290.66
    hull [ 2.2823205e-04  2.1673422e-03 -3.8735073e-03 -1.7998373e-05
    5.0355148e-01  3.2691330e-01]
    back leg [-3.1713054e+00  1.8356368e-06]
    back wheel [-0.7392487  0.       ]
    front leg [3.1714251e+00 1.2107193e-07]
    front wheel [-0.7392414  0.       ]
    lander 1.0

    - 完整跑完一个episode：
    step 1296 total_reward +340.49
    hull [-4.9899105e-04 -1.1690515e-03  4.0942398e-01  3.3036969e-03
    1.0647765e+01  4.6736881e-01]
    back leg [ 3.497310e-03 -5.736947e-07]
    back wheel [-0.9824801  1.       ]
    front leg [-2.2177882e-03  1.3364479e-07]
    front wheel [-0.9820424  0.       ]
    lander 0.0

### 2024.10.16:

- README.md:

    1. 修改了一下格式
    2. 把我们做记录的地方进行了注释，这样就只会在编辑模式下看到记录
    3. 增加了Steps to Set Up & Step1: Environment configuration

- 增加environment.yaml

- __init__部分

    1. 修改了action space和observation space
    - 发现运行后小车倒着跑

### 2024.10.13:
- reset部分

    1. 把wheel的wheel转动换回了revolute，但删除角度限制（自由转动）
    2. 添加了leg的lowerAngle和upperAngle的判断
    3. 调整了尺寸大小 LEG_H，HALF_HEIGHT_LANDER
    
    备份在了local里

    ***

- render部分

    1. 加入了zewen写的 _create_particle & _clean_particles
    2. 复制了rui写的render
    - 目前可以跑通，但是没有particle的显示（或许和没有用到_create_particle有关）

    ***

- __init__部分

    1. 复制了zewen的 self.fd_triangle
    2. 修改了激光雷达的数量10->20 (obervation 10 -> 20)

    ***

- _destory部分

    1. 参考zewen的修改进行了更新

    ***

- _generate_terrain部分

    1. 删除PIT陷阱
    2. 修改 STUMP -> TOWER
    3. 保留了stairs
    4. 增加了slope
    5. 调高了step的间距 1 -> 2
    6. 增加了hole

    ***

- step部分

    1. 修改了机关雷达的扫描范围（0~pi，接近半圆形但似乎不是垂直于地面？）
    2. state有增加：assert len(state) == 24 -> 34

    ***

before：
- reset部分
    1. 增加了lander
    2. 让lander与hull之间是焊接连接weld
    3. 将wheel设置成了wheel的转动
    4. 更换了wedth和height的变量名 -> half
    5. 在self.drawlist中添加了self.lander -->
