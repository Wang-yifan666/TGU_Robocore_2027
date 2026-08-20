#!/bin/bash
# 用法: ./git-push.sh "提交信息"

cd "$(dirname "$0")/.."  

if [ -z "$1" ]; then
    echo "错误: 请提供提交信息"
    echo "用法: $0 \"提交信息\""
    exit 1
fi

git add .
git commit -m "$1"
git push