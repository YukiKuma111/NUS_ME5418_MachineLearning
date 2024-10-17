# ME5418_ML

## Steps to Set Up

### Step1: Environment configuration

```
mkdir your_workspace
cd your_workspace/
git clone https://github.com/YukiKuma111/ME5418_ML.git
cd ME5418_ML/
conda env create -f environment.yaml
```

<!-- ## zewen:


## rui:


## ziyue:

### 2024.10.17:

- class ContactDetector：

    1. run时发现lander的触底检测无效，现已修复
    2. 

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
