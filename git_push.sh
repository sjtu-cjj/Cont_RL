#!/bin/bash

# 检查是否提供了 commit message
if [ $# -lt 1 ]; then
  echo "用法: $0 \"提交注释内容\""
  exit 1
fi

# 拼接所有参数作为 commit message
COMMIT_MSG="$*"

# 执行 Git 操作
git add .
git commit -m "$COMMIT_MSG"
git push

# 提示完成
echo "✅ Git 推送完成：$COMMIT_MSG"

