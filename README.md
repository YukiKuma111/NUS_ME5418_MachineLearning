# ME5418_ML

## zewen:


## rui:


## ziyue:

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
    3. ε＝ε＝ε＝(#>д<)ﾉ

    ***

- __init__部分

    1. 复制了zewen的 self.fd_triangle
    2. 修改了激光雷达的数量10->20 (obervation 10 -> 20)

- step部分

    1. 修改了机关雷达的扫描范围（0~pi，接近半圆形但似乎不是垂直于地面？）
    2. state有增加：assert len(state) == 24 -> 34

before：
- reset部分
    1. 增加了lander
    2. 让lander与hull之间是焊接连接weld
    3. 将wheel设置成了wheel的转动
    4. 更换了wedth和height的变量名 -> half
    5. 在self.drawlist中添加了self.lander
