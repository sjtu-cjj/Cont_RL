#!/bin/bash

# 检查是否提供了 commit message
if [ $# -lt 1 ]; then
  echo "用法: $0 \"提交注释内容\""
  exit 1
fi

# 拼接所有参数作为 commit message
COMMIT_MSG="$*"

# 获取当前目录信息
REPO_DIR=$(pwd)
REPO_NAME=$(basename "$REPO_DIR")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建备份目录
BACKUP_DIR="$REPO_DIR/backups"
mkdir -p "$BACKUP_DIR"

# 设置备份文件名
BACKUP_FILE="${BACKUP_DIR}/${REPO_NAME}_backup_${TIMESTAMP}.zip"

# 打包整个项目目录（不含 .git 和备份目录本身）
echo "📦 正在创建备份：$BACKUP_FILE"
zip -r "$BACKUP_FILE" . -x ".git/*" "backups/*" > /dev/null

# Git 操作
git add .
git commit -m "$COMMIT_MSG"
git push

# 提示完成
echo "✅ Git 推送完成：$COMMIT_MSG"
echo "🗂️ 备份文件已保存：$BACKUP_FILE"

