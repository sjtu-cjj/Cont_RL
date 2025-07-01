#!/bin/bash

# 检查是否提供了 commit message
if [ $# -lt 1 ]; then
  echo "用法: $0 \"提交注释内容\""
  exit 1
fi

# 拼接所有参数作为 commit message
COMMIT_MSG="$*"

# 获取当前路径信息
REPO_DIR=$(pwd)
REPO_NAME=$(basename "$REPO_DIR")
WORKSPACE_DIR=$(dirname "$REPO_DIR")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 设置备份路径和文件名
BACKUP_DIR="${WORKSPACE_DIR}/backups"
BACKUP_FILE="${BACKUP_DIR}/${REPO_NAME}_backup_${TIMESTAMP}.zip"

# 创建备份目录（如果不存在）
mkdir -p "$BACKUP_DIR"

# 创建备份（排除 .git）
echo "📦 正在创建备份：$BACKUP_FILE"
cd "$WORKSPACE_DIR"
zip -r "$BACKUP_FILE" "$REPO_NAME" -x "${REPO_NAME}/.git/*" > /dev/null
cd "$REPO_DIR"

# Git 操作
git add .
git commit -m "$COMMIT_MSG"
git push

# 完成提示
echo "✅ Git 推送完成：$COMMIT_MSG"
echo "🗂️ 备份文件已保存：$BACKUP_FILE"

