#!/bin/bash

# 定义软链接所在的目录
LINK_DIR="dataset/water_val_16_256"

# 定义数据源目录
DATA_SRC="/home/siat-video2024/zjh/UnderwaterVideoDataset/UOT32_selected_pure_img"

# 获取所有子文件夹名称
folders=($(ls -1 "$DATA_SRC"))

# 遍历所有子文件夹
for folder in "${folders[@]}"; do
    echo "正在处理文件夹: $folder"

    # 软链接的目标路径
    TARGET_PATH="$DATA_SRC/$folder"

    # 修改软链接
    ln -sfn "$TARGET_PATH" "$LINK_DIR/hr_256"
    ln -sfn "$TARGET_PATH" "$LINK_DIR/sr_16_256"

    echo "软链接更新为: $TARGET_PATH"

    # 运行 Python 脚本
    echo "开始执行 infer.py..."
    python infer.py

    echo "infer.py 执行完成，继续下一个文件夹..."
done

echo "所有文件夹处理完成！"
