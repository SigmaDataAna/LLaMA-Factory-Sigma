#!/bin/bash

# 1. 设置变量
# AZCOPY_VERSION="10.19.0"  # 指定azcopy版本
DOWNLOAD_URL="https://aka.ms/downloadazcopy-v10-linux"
INSTALL_DIR="/usr/local/bin"
TEMP_DIR="/tmp/azcopy_download"

# 2. 创建临时目录
mkdir -p $TEMP_DIR

# 3. 下载 azcopy
echo "Downloading azcopy..."
wget -O $TEMP_DIR/azcopy.tar.gz $DOWNLOAD_URL

# 4. 解压文件
echo "Extracting azcopy..."
tar -xf $TEMP_DIR/azcopy.tar.gz -C $TEMP_DIR

# 5. 移动 azcopy 到系统路径
if command -v sudo &> /dev/null
then
    echo "sudo command found, using sudo for installation."
    SUDO="sudo"
else
    echo "sudo command not found, proceeding without sudo."
    SUDO=""
fi
echo "Installing azcopy to $INSTALL_DIR..."
$SUDO mv $TEMP_DIR/azcopy_linux_amd64_*/azcopy $INSTALL_DIR

# 6. 设置执行权限
$SUDO chmod +x $INSTALL_DIR/azcopy

# 7. 检查是否成功安装
if command -v azcopy &> /dev/null
then
    echo "AzCopy installed successfully!"
    azcopy --version
else
    echo "Error: AzCopy installation failed!"
fi

# 8. 清理临时文件
echo "Cleaning up..."
rm -rf $TEMP_DIR

echo "AzCopy setup completed."