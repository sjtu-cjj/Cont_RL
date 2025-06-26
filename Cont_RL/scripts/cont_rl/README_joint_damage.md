# 关节损伤模块使用指南

## 概述
本模块提供了用于机器人关节故障模拟的环境包装器，支持永久性关节损伤，用于研究机器人在故障条件下的适应能力。

## 模块结构

### 文件组织
```
Cont_RL/scripts/cont_rl/
├── joint_damage.py           # 关节损伤模块（新模块）
├── streaming_train.py        # 简化的训练脚本
└── README_joint_damage.md    # 使用指南（本文件）
```

### 主要组件

#### 1. `joint_damage.py` - 关节损伤模块
包含以下主要类和功能：

- **JointNameMapper**: 处理关节名称到索引的转换
- **DamageApplier**: 负责对动作应用各种类型的损伤
- **JointDamageWrapper**: 主要的环境包装器
- **工具函数**: 配置创建、统计打印等便利函数

#### 2. `streaming_train.py` - 简化的训练脚本
从原始的758行代码减少到约180行，移除了冗长的关节损伤代码，通过导入模块来使用关节损伤功能。

## 使用方法

### 基本使用
```python
from joint_damage import JointDamageWrapper, create_damage_config

# 配置损伤参数
damage_config = create_damage_config(args_cli)

# 包装环境
env = JointDamageWrapper(base_env, damage_config)
```

### 指定损伤关节
通过环境变量指定要损伤的关节：
```bash
export DAMAGED_JOINT="FL_hip,RR_thigh"
python streaming_train.py --task=... --enable_joint_damage
```

### 支持的关节名称
**UnitreeGo2机器人关节映射：**
- **前左腿 (FL)**: FL_hip (0), FL_thigh (1), FL_calf (2)
- **前右腿 (FR)**: FR_hip (3), FR_thigh (4), FR_calf (5)
- **后左腿 (RL)**: RL_hip (6), RL_thigh (7), RL_calf (8)
- **后右腿 (RR)**: RR_hip (9), RR_thigh (10), RR_calf (11)

**支持的关节名称格式：**
- 简化格式：`FL_hip`, `RR_thigh` 等
- 完整格式：`FL_hip_joint`, `RR_thigh_joint` 等

### 损伤类型
1. **'zero'**: 将受损关节的力矩设为零
2. **'partial'**: 按比例降低受损关节的力矩
3. **'random'**: 对受损关节应用随机噪声

### 命令行参数详解
```bash
python streaming_train.py \
    --enable_joint_damage \               # 启用关节损伤
    --damage_probability 0.001 \          # 损伤概率（每步）
    --max_damaged_joints 2 \              # 最大损伤关节数
    --damage_type zero \                  # 损伤类型
    --damage_severity 0.7 \               # 损伤严重程度（partial类型）
    --output_damage_info                  # 输出详细信息
```

## 运行示例

### 1. 使用默认配置训练
```bash
python streaming_train.py \
    --task=Cont-RL-Velocity-Flat-Unitree-Go2-ContRL-v0 \
    --num_envs=4096 \
    --enable_joint_damage
```

### 2. 指定损伤特定关节
```bash
export DAMAGED_JOINT="FL_hip"
python streaming_train.py \
    --task=Cont-RL-Velocity-Flat-Unitree-Go2-ContRL-v0 \
    --num_envs=4096 \
    --enable_joint_damage \
    --damage_probability 0.002 \
    --output_damage_info
```

### 3. 使用partial损伤类型
```bash
python streaming_train.py \
    --task=Cont-RL-Velocity-Flat-Unitree-Go2-ContRL-v0 \
    --num_envs=4096 \
    --enable_joint_damage \
    --damage_type partial \
    --damage_severity 0.7 \
    --max_damaged_joints 2
```

### 4. 多关节损伤
```bash
export DAMAGED_JOINT="FL_hip,RR_hip,RL_thigh"
python streaming_train.py \
    --task=Cont-RL-Velocity-Flat-Unitree-Go2-ContRL-v0 \
    --num_envs=4096 \
    --enable_joint_damage \
    --damage_type zero \
    --max_damaged_joints 3
```

## API 参考

### JointDamageWrapper 主要方法

#### `get_damage_statistics()`
获取损伤统计信息：
```python
stats = env.get_damage_statistics()
# 返回: {
#     'total_damage_events': int,      # 总损伤事件数
#     'current_damaged_joints': int,   # 当前损伤关节数
#     'damage_rate': float,            # 损伤率（事件/步）
#     'step_count': int                # 总步数
# }
```

#### `get_damage_report()`
获取详细的损伤报告：
```python
report = env.get_damage_report()
print(report)  # 打印详细的损伤状态
```

#### `reset_damage_states()`
重置所有损伤状态：
```python
env.reset_damage_states()  # 清除所有损伤
```

### 工具函数

#### `create_damage_config(args_cli)`
根据命令行参数创建损伤配置：
```python
damage_config = create_damage_config(args_cli)
```

#### `print_damage_config_info(damage_config)`
打印损伤配置信息：
```python
print_damage_config_info(damage_config)
```

#### `find_damage_wrapper(env)`
从环境链中查找损伤包装器：
```python
damage_wrapper = find_damage_wrapper(env)
```

#### `print_damage_statistics(damage_wrapper, detailed=False)`
打印损伤统计信息：
```python
print_damage_statistics(damage_wrapper, output_detailed_report=True)
```

## 模块化优势

### 代码重构效果
| 指标 | 重构前 | 重构后 | 改进 |
|-----|--------|--------|------|
| 总行数 | 758行 | ~180行 | 减少76% |
| 关节损伤代码 | 内嵌 | 独立模块 | 完全分离 |
| 类数量 | 1个大类 | 3个专门类 | 责任分离 |
| 可维护性 | 低 | 高 | 大幅提升 |

### 设计优势
1. **模块化设计**: 关节损伤功能独立于训练脚本
2. **职责分离**: 每个类负责特定功能
3. **易于扩展**: 可以轻松添加新的损伤类型
4. **重用性**: 可用于其他项目
5. **可测试性**: 独立模块易于单元测试

### 性能优化
1. **懒初始化**: 基于实际使用动态初始化
2. **高效张量操作**: 使用PyTorch内置函数
3. **内存优化**: 避免不必要的数据复制
4. **计算优化**: 最小化损伤检查开销

## 技术特性

### 永久性损伤机制
- 一旦关节受损，将持续到训练结束
- 适合研究长期故障下的适应能力
- 避免了临时损伤的复杂性

### 观测值一致性
- 自动修正观测值中的`last_action`部分
- 确保观测与实际执行动作的一致性
- 支持字典和张量两种观测格式

### 多环境支持
- 支持并行多环境训练
- 每个环境独立管理损伤状态
- 自动检测环境数量

### 设备兼容性
- 自动处理GPU/CPU设备转换
- 支持CUDA加速计算
- 兼容不同硬件配置

## 扩展指南

### 添加新的损伤类型
在`DamageApplier`类中添加新的静态方法：
```python
@staticmethod
def apply_custom_damage(damaged_actions: torch.Tensor, env_idx: int, damaged_joints: torch.Tensor) -> None:
    # 自定义损伤逻辑
    pass
```

然后在`JointDamageWrapper._apply_damage_masks`中添加相应的调用。

### 支持新的机器人
在`JointNameMapper`中扩展关节映射：
```python
def __init__(self, robot_type="unitree_go2"):
    if robot_type == "new_robot":
        self.joint_name_to_index = {
            # 新机器人的关节映射
        }
    else:
        # 默认UnitreeGo2映射
        pass
```

### 添加临时性损伤
可以扩展`JointDamageWrapper`以支持临时性损伤：
```python
def _apply_temporary_damage(self, env_idx: int, damaged_joints: torch.Tensor, duration: int):
    """应用临时性损伤"""
    for joint_idx in damaged_joints:
        self.damage_states[env_idx, joint_idx] = duration  # 使用持续时间而非1
```

## 故障排除

### 常见问题

1. **关节名称不识别**
   - 检查拼写和格式
   - 参考支持的关节名称列表

2. **环境包装器未找到**
   - 确保在正确位置创建了损伤包装器
   - 检查环境链的创建顺序

3. **观测值维度不匹配**
   - 检查观测值格式（dict vs tensor）
   - 确认last_action的位置

### 调试技巧

1. **启用详细输出**：
   ```bash
   --output_damage_info
   ```

2. **检查损伤状态**：
   ```python
   damage_wrapper = find_damage_wrapper(env)
   print(damage_wrapper.get_damage_report())
   ```

3. **验证关节映射**：
   ```python
   from joint_damage import JointNameMapper
   mapper = JointNameMapper()
   print(mapper.joint_name_to_index)
   ```

## 总结

这个模块化的关节损伤系统提供了：
- 🔧 **易用性**: 简单的配置和使用
- 🚀 **性能**: 高效的损伤计算
- 🔄 **灵活性**: 支持多种损伤类型
- 📊 **监控**: 详细的损伤统计
- 🔧 **可扩展**: 易于添加新功能

通过将关节损伤功能模块化，我们大大提高了代码的可维护性和重用性，同时保持了功能的完整性和性能。 