#!/bin/bash

# 自动化训练脚本 - 依次训练不同关节的损伤情况
# 作者: 生成于 $(date)

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 训练配置
TASK="Cont-RL-Velocity-Flat-Unitree-Go2-ContRL-v0"
PRETRAINED_CHECKPOINT="/home/chenjunjie/workspace/Incremental_learning/Cont_RL/logs/rsl_rl/unitree_go2_flat/2025-05-07_13-31-24/model_299.pt"
FINETUNE_MODE="critic-only"
DAMAGE_PROBABILITY="1.0"
DAMAGE_TYPE="zero"

# 要训练的关节列表
JOINTS=("FL_hip" "FL_thigh" "FL_calf" "RR_hip" "RR_thigh" "RR_calf")

# 记录开始时间
SCRIPT_START_TIME=$(date)
SCRIPT_START_TIMESTAMP=$(date +%s)

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}🚀 开始自动化关节损伤训练${NC}"
echo -e "${BLUE}📅 开始时间: ${SCRIPT_START_TIME}${NC}"
echo -e "${BLUE}🎯 训练任务: ${TASK}${NC}"
echo -e "${BLUE}🧠 预训练模型: ${PRETRAINED_CHECKPOINT}${NC}"
echo -e "${BLUE}🔧 微调模式: ${FINETUNE_MODE}${NC}"
echo -e "${BLUE}⚙️ 损伤类型: ${DAMAGE_TYPE}${NC}"
echo -e "${BLUE}📊 损伤概率: ${DAMAGE_PROBABILITY}${NC}"
echo -e "${BLUE}🦴 关节列表: ${JOINTS[*]}${NC}"
echo -e "${BLUE}================================================================================================${NC}"

# 初始化计数器
TOTAL_JOINTS=${#JOINTS[@]}
CURRENT_COUNT=0
SUCCESS_COUNT=0
FAILED_JOINTS=()

# 遍历每个关节进行训练
for JOINT in "${JOINTS[@]}"; do
    CURRENT_COUNT=$((CURRENT_COUNT + 1))
    
    echo ""
    echo -e "${YELLOW}[${CURRENT_COUNT}/${TOTAL_JOINTS}] 开始训练关节: ${JOINT}${NC}"
    echo -e "${YELLOW}================================================================${NC}"
    
    # 记录单个训练开始时间
    TRAIN_START_TIME=$(date)
    TRAIN_START_TIMESTAMP=$(date +%s)
    
    echo -e "${BLUE}⏰ 开始时间: ${TRAIN_START_TIME}${NC}"
    echo -e "${BLUE}🦴 损伤关节: ${JOINT}${NC}"
    
    # 执行训练命令
    DAMAGED_JOINT=$JOINT python ./streaming_train.py \
        --task=$TASK \
        --pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
        --finetune_mode=$FINETUNE_MODE \
        --enable_joint_damage \
        --damage_probability=$DAMAGE_PROBABILITY \
        --damage_type=$DAMAGE_TYPE \
        --headless \
        --video
    
    # 检查训练是否成功
    EXIT_CODE=$?
    TRAIN_END_TIME=$(date)
    TRAIN_END_TIMESTAMP=$(date +%s)
    TRAIN_DURATION=$((TRAIN_END_TIMESTAMP - TRAIN_START_TIMESTAMP))
    
    # 格式化训练时间
    HOURS=$((TRAIN_DURATION / 3600))
    MINUTES=$(((TRAIN_DURATION % 3600) / 60))
    SECONDS=$((TRAIN_DURATION % 60))
    
    if [ $HOURS -gt 0 ]; then
        DURATION_STR="${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    elif [ $MINUTES -gt 0 ]; then
        DURATION_STR="${MINUTES}分钟 ${SECONDS}秒"
    else
        DURATION_STR="${SECONDS}秒"
    fi
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ 关节 ${JOINT} 训练完成！${NC}"
        echo -e "${GREEN}⏱️ 训练时长: ${DURATION_STR}${NC}"
        echo -e "${GREEN}🏁 结束时间: ${TRAIN_END_TIME}${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}❌ 关节 ${JOINT} 训练失败！退出码: ${EXIT_CODE}${NC}"
        echo -e "${RED}⏱️ 失败前时长: ${DURATION_STR}${NC}"
        echo -e "${RED}🏁 失败时间: ${TRAIN_END_TIME}${NC}"
        FAILED_JOINTS+=($JOINT)
    fi
    
    echo -e "${YELLOW}================================================================${NC}"
    
    # 如果不是最后一个关节，等待一段时间再继续
    if [ $CURRENT_COUNT -lt $TOTAL_JOINTS ]; then
        echo -e "${BLUE}⏳ 等待5秒后继续下一个关节训练...${NC}"
        sleep 5
    fi
done

# 计算总时间
SCRIPT_END_TIME=$(date)
SCRIPT_END_TIMESTAMP=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END_TIMESTAMP - SCRIPT_START_TIMESTAMP))

# 格式化总时间
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

if [ $TOTAL_HOURS -gt 0 ]; then
    TOTAL_DURATION_STR="${TOTAL_HOURS}小时 ${TOTAL_MINUTES}分钟 ${TOTAL_SECONDS}秒"
elif [ $TOTAL_MINUTES -gt 0 ]; then
    TOTAL_DURATION_STR="${TOTAL_MINUTES}分钟 ${TOTAL_SECONDS}秒"
else
    TOTAL_DURATION_STR="${TOTAL_SECONDS}秒"
fi

# 输出最终总结
echo ""
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}🎉 所有关节训练完成！${NC}"
echo -e "${BLUE}📅 开始时间: ${SCRIPT_START_TIME}${NC}"
echo -e "${BLUE}🏁 结束时间: ${SCRIPT_END_TIME}${NC}"
echo -e "${BLUE}⏱️ 总时长: ${TOTAL_DURATION_STR}${NC}"
echo -e "${BLUE}📊 训练统计:${NC}"
echo -e "${GREEN}  ✅ 成功: ${SUCCESS_COUNT}/${TOTAL_JOINTS}${NC}"

if [ ${#FAILED_JOINTS[@]} -gt 0 ]; then
    echo -e "${RED}  ❌ 失败: ${#FAILED_JOINTS[@]}/${TOTAL_JOINTS}${NC}"
    echo -e "${RED}  💥 失败关节: ${FAILED_JOINTS[*]}${NC}"
else
    echo -e "${GREEN}  🎊 所有关节训练成功！${NC}"
fi

echo -e "${BLUE}================================================================================================${NC}"

# 如果有失败的关节，以非零退出码结束
if [ ${#FAILED_JOINTS[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi
