# 数字图像处理工具 - 安装与使用指南

## 📖 目录

- [简介](#简介)
- [系统要求](#系统要求)
- [安装步骤](#安装步骤)
  - [Windows 安装](#windows-安装)
  - [Mac OS X 安装](#mac-os-x-安装)
  - [Linux 安装](#linux-安装)
- [快速开始](#快速开始)
- [功能详解](#功能详解)
- [键盘快捷键](#键盘快捷键)
- [常见问题](#常见问题)
- [技术支持](#技术支持)

---

## 简介

这是一个基于 Web 技术开发的数字图像处理工具，提供了与 Matlab 图像处理工具箱类似的功能。该工具采用现代化的图形用户界面（GUI），支持键盘和鼠标交互操作，包含了丰富的数字图像处理基本算法。

### 主要特点

- ✅ **完整的GUI界面** - 现代化设计，操作直观
- ✅ **33种图像处理算法** - 覆盖基本操作、滤波、边缘检测、形态学等
- ✅ **实时预览** - 所有操作即时显示效果
- ✅ **撤销/重做** - 支持最多50步历史记录
- ✅ **跨平台支持** - Windows、Mac OS X、Linux 全平台兼容
- ✅ **无需安装** - 基于浏览器运行，无需额外依赖

---

## 系统要求

### 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 双核处理器 | 四核及以上处理器 |
| 内存 | 4GB RAM | 8GB RAM 或以上 |
| 硬盘 | 500MB 可用空间 | 1GB 可用空间 |
| 显示器 | 1280x720 分辨率 | 1920x1080 分辨率或更高 |

### 软件要求

| 操作系统 | 支持版本 |
|----------|----------|
| Windows | Windows 10 / Windows 11 |
| Mac OS X | macOS 10.15 (Catalina) 及以上 |
| Linux | Ubuntu 18.04+, Debian 10+, Fedora 30+ |

### 浏览器要求

| 浏览器 | 最低版本 |
|--------|----------|
| Google Chrome | 80+ |
| Mozilla Firefox | 75+ |
| Microsoft Edge | 80+ |
| Safari | 13+ |

> **注意**: 推荐使用 Google Chrome 或 Mozilla Firefox 以获得最佳体验。

---

## 安装步骤

### 方式一：使用已编译版本（推荐）

#### Windows 安装

1. **下载程序包**
   - 从发布页面下载 `image-processor-windows.zip`
   - 解压到任意目录（如 `C:\ImageProcessor`）

2. **安装 Node.js（如未安装）**
   ```powershell
   # 方法一：通过官网下载安装包
   # 访问 https://nodejs.org/ 下载 LTS 版本安装包
   
   # 方法二：使用 Chocolatey 包管理器（推荐）
   choco install nodejs-lts
   
   # 方法三：使用 Scoop 包管理器
   scoop install nodejs-lts
   ```

3. **验证安装**
   ```powershell
   # 打开 PowerShell 或命令提示符
   node --version   # 应显示 v18.x.x 或更高
   npm --version    # 应显示 9.x.x 或更高
   ```

4. **启动程序**
   ```powershell
   # 进入程序目录
   cd C:\ImageProcessor
   
   # 安装依赖
   npm install
   
   # 启动开发服务器
   npm run dev
   ```

5. **访问应用**
   - 打开浏览器访问 `http://localhost:3000`

#### Mac OS X 安装

1. **下载程序包**
   - 从发布页面下载 `image-processor-macos.tar.gz`
   - 双击解压或使用终端：
   ```bash
   tar -xzf image-processor-macos.tar.gz
   ```

2. **安装 Node.js（如未安装）**
   ```bash
   # 方法一：通过官网下载安装包
   # 访问 https://nodejs.org/ 下载 LTS 版本安装包
   
   # 方法二：使用 Homebrew 包管理器（推荐）
   brew install node@18
   
   # 方法三：使用 nvm（Node Version Manager）
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
   source ~/.zshrc
   nvm install 18
   nvm use 18
   ```

3. **验证安装**
   ```bash
   # 打开终端
   node --version   # 应显示 v18.x.x 或更高
   npm --version    # 应显示 9.x.x 或更高
   ```

4. **启动程序**
   ```bash
   # 进入程序目录
   cd ImageProcessor
   
   # 安装依赖
   npm install
   
   # 启动开发服务器
   npm run dev
   ```

5. **访问应用**
   - 打开浏览器访问 `http://localhost:3000`

#### Linux 安装

1. **下载程序包**
   ```bash
   # 使用 wget
   wget https://releases.example.com/image-processor-linux.tar.gz
   
   # 或使用 curl
   curl -LO https://releases.example.com/image-processor-linux.tar.gz
   
   # 解压
   tar -xzf image-processor-linux.tar.gz
   ```

2. **安装 Node.js（如未安装）**
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
   
   # Fedora
   sudo dnf install nodejs
   
   # Arch Linux
   sudo pacman -S nodejs npm
   ```

3. **启动程序**
   ```bash
   cd ImageProcessor
   npm install
   npm run dev
   ```

---

### 方式二：从源码构建

#### 前提条件

- Git 版本控制工具
- Node.js 18.x 或更高版本
- npm 或 bun 包管理器

#### Windows 构建步骤

```powershell
# 1. 克隆仓库
git clone https://github.com/your-repo/image-processor.git
cd image-processor

# 2. 安装依赖
npm install

# 3. 开发模式运行
npm run dev

# 4. 构建生产版本（可选）
npm run build
npm run start
```

#### Mac OS X 构建步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-repo/image-processor.git
cd image-processor

# 2. 安装依赖
npm install

# 3. 开发模式运行
npm run dev

# 4. 构建生产版本（可选）
npm run build
npm run start
```

---

## 快速开始

### 1. 打开图片

有三种方式打开图片：

- **点击按钮**: 点击"打开图片"按钮，选择本地图片文件
- **拖放上传**: 直接将图片文件拖放到画布区域
- **示例图片**: 点击示例图片按钮快速加载预设图片

支持格式: JPG, JPEG, PNG, GIF, BMP, WebP

### 2. 选择工具

从左侧工具栏选择需要的工具：

| 工具 | 快捷键 | 功能 |
|------|--------|------|
| 选择工具 | V | 基本指针操作 |
| 移动工具 | M | 移动画布（开发中） |
| 裁剪工具 | C | 框选区域裁剪 |
| 画笔工具 | B | 自由绘制 |
| 橡皮擦 | E | 擦除恢复原图 |
| 吸管工具 | I | 拾取图像颜色 |

### 3. 应用滤镜

1. 在右侧面板选择"滤镜"标签
2. 找到需要的滤镜效果
3. 调整参数（如有）
4. 点击按钮应用效果

### 4. 调整图像

1. 在右侧面板选择"调整"标签
2. 调整亮度、对比度、伽马、饱和度等参数
3. 点击"应用调整"按钮

### 5. 保存图片

- 点击顶部工具栏的保存按钮
- 或使用快捷键 `Ctrl+S` (Windows/Linux) / `Cmd+S` (Mac)
- 图片将以 PNG 格式下载

---

## 功能详解

### 一、基本图像操作

#### 1. 灰度化 (Grayscale)
将彩色图像转换为灰度图像，使用标准加权公式：
```
Gray = 0.299*R + 0.587*G + 0.114*B
```

#### 2. 二值化 (Binarization)
将灰度图像转换为黑白二值图像，可调整阈值（0-255）：
- 像素值 ≥ 阈值 → 白色 (255)
- 像素值 < 阈值 → 黑色 (0)

#### 3. 反色 (Invert)
对图像颜色取反：
```
R' = 255 - R
G' = 255 - G
B' = 255 - B
```

#### 4. 亮度调整 (Brightness)
整体增加或减少图像亮度，范围：-100 到 +100

#### 5. 对比度调整 (Contrast)
调整图像明暗对比程度，范围：0.0 到 4.0

#### 6. 伽马校正 (Gamma Correction)
非线性亮度调整，用于校正图像亮度分布

#### 7. 直方图均衡化 (Histogram Equalization)
自动调整图像对比度，使直方图分布更均匀

---

### 二、滤波算法

#### 1. 高斯模糊 (Gaussian Blur)
使用高斯核进行图像平滑，可调整σ值（模糊程度）

**参数**: σ (sigma) - 0.1 到 5.0
- 较小值：轻微模糊
- 较大值：强烈模糊

#### 2. 均值滤波 (Mean Filter)
使用矩形邻域平均值替代中心像素，有效去除随机噪声

#### 3. 中值滤波 (Median Filter)
使用邻域中值替代中心像素，有效去除椒盐噪声，保留边缘

#### 4. 锐化 (Sharpening)
增强图像边缘和细节

**参数**: 强度 - 0.1 到 5.0

---

### 三、边缘检测

#### 1. Sobel 边缘检测
使用 3x3 Sobel 算子检测边缘，对噪声有一定抑制能力

#### 2. Prewitt 边缘检测
使用 3x3 Prewitt 算子检测边缘，计算简单

#### 3. Laplacian 边缘检测
使用二阶导数检测边缘，对噪声敏感

#### 4. Canny 边缘检测
多阶段边缘检测算法，效果最佳：
1. 高斯模糊降噪
2. 计算梯度幅值和方向
3. 非极大值抑制
4. 双阈值检测

---

### 四、形态学运算

#### 1. 腐蚀 (Erosion)
缩小图像中的白色区域，去除小的白色噪点

#### 2. 膨胀 (Dilation)
扩大图像中的白色区域，填充小的黑色空洞

#### 3. 开运算 (Opening)
先腐蚀后膨胀，去除小的白色噪点，保持整体形状

#### 4. 闭运算 (Closing)
先膨胀后腐蚀，填充小的黑色空洞，保持整体形状

**参数**: 核大小 - 3x3 到 15x15

---

### 五、几何变换

#### 1. 旋转 (Rotation)
- 预设：顺时针90°、逆时针90°
- 自定义：0° - 360° 任意角度

#### 2. 缩放 (Scaling)
- 范围：10% - 200%
- 双线性插值保证图像质量

#### 3. 翻转 (Flip)
- 水平翻转：左右镜像
- 垂直翻转：上下镜像

---

### 六、特效处理

#### 1. 浮雕效果 (Emboss)
模拟浮雕艺术效果，突出图像轮廓

#### 2. 油画效果 (Oil Painting)
模拟油画艺术风格

**参数**:
- 半径：1 - 10
- 强度：5 - 30

#### 3. 马赛克效果 (Mosaic)
像素化处理，模糊图像细节

**参数**: 块大小：2 - 30 像素

#### 4. 边缘增强 (Edge Enhance)
强化图像边缘，提高清晰度

---

### 七、噪声处理

#### 1. 椒盐噪声 (Salt & Pepper Noise)
模拟图像传输过程中的随机噪声

**参数**: 密度 - 1% - 50%

#### 2. 高斯噪声 (Gaussian Noise)
模拟传感器噪声

**参数**: 标准差 - 5 - 100

---

## 键盘快捷键

### 文件操作

| 快捷键 | Windows/Linux | Mac OS X | 功能 |
|--------|---------------|----------|------|
| 打开图片 | Ctrl+O | ⌘+O | 打开图片文件 |
| 保存图片 | Ctrl+S | ⌘+S | 保存当前图片 |
| 撤销 | Ctrl+Z | ⌘+Z | 撤销上一步操作 |
| 重做 | Ctrl+Y | ⌘+Y | 重做已撤销操作 |

### 工具切换

| 快捷键 | 功能 |
|--------|------|
| V | 选择工具 |
| M | 移动工具 |
| C | 裁剪工具 |
| B | 画笔工具 |
| E | 橡皮擦 |
| I | 吸管工具 |

### 缩放控制

| 快捷键 | 功能 |
|--------|------|
| + | 放大 25% |
| - | 缩小 25% |
| 0 | 实际大小 (100%) |

### 滤镜快捷键

| 快捷键 | Windows/Linux | Mac OS X | 功能 |
|--------|---------------|----------|------|
| 顺时针旋转 | Ctrl+R | ⌘+R | 顺时针旋转90° |
| 逆时针旋转 | Ctrl+Shift+R | ⌘+Shift+R | 逆时针旋转90° |
| 水平翻转 | Ctrl+H | ⌘+H | 水平翻转 |
| 垂直翻转 | Ctrl+Shift+H | ⌘+Shift+H | 垂直翻转 |
| 灰度化 | Ctrl+G | ⌘+G | 灰度化处理 |
| 反色 | Ctrl+I | ⌘+I | 颜色取反 |
| 重置 | Delete | Delete | 恢复原始图像 |

---

## 常见问题

### Q1: 程序无法启动怎么办？

**A**: 请检查以下项目：
1. 确认 Node.js 已正确安装（运行 `node --version`）
2. 确认端口 3000 未被占用
3. 尝试删除 `node_modules` 目录后重新运行 `npm install`
4. 检查防火墙设置是否阻止了程序运行

### Q2: 图片加载失败怎么办？

**A**: 可能的原因和解决方法：
1. **文件格式不支持**: 确保图片是 JPG/PNG/GIF/BMP/WebP 格式
2. **文件过大**: 尝试压缩图片或使用较小的图片
3. **文件损坏**: 尝试用其他图片查看器打开确认

### Q3: 滤镜处理很慢怎么办？

**A**: 处理速度取决于：
1. 图片尺寸：大图片需要更多处理时间
2. CPU性能：复杂算法（如Canny边缘检测）需要较多计算
3. 建议：先用小尺寸图片测试效果

### Q4: 如何处理超大图片？

**A**: 建议：
1. 先使用缩放功能将图片缩小
2. 或使用外部工具预先调整图片尺寸
3. 推荐处理 2000x2000 像素以下的图片

### Q5: 导出的图片质量如何？

**A**: 
1. 导出格式为 PNG，无损压缩
2. 保持原始图片的色彩深度
3. 处理过程中的数值运算可能导致轻微精度损失

### Q6: 为什么有些滤镜效果不明显？

**A**: 
1. 检查参数设置是否合适
2. 某些滤镜对特定类型的图片效果更明显
3. 例如：边缘检测对高对比度图片效果更好

### Q7: 可以批量处理图片吗？

**A**: 当前版本暂不支持批量处理，需要逐张处理图片。

### Q8: 程序支持哪些语言？

**A**: 当前版本支持简体中文界面，后续版本将支持多语言。

### Q9: 如何报告 Bug 或提出建议？

**A**: 
1. 在 GitHub 仓库提交 Issue
2. 或发送邮件至技术支持邮箱
3. 请附带详细描述和复现步骤

### Q10: 程序是免费的吗？

**A**: 是的，本程序完全免费，开源使用。

---

## 技术支持

### 获取帮助

- 📧 **邮箱**: support@example.com
- 💬 **GitHub Issues**: https://github.com/your-repo/image-processor/issues
- 📚 **文档**: https://docs.example.com/image-processor

### 版本信息

- 当前版本: 1.0.0
- 更新日期: 2024年
- 开发框架: Next.js 16 + React 19 + TypeScript

### 许可证

本项目采用 MIT 许可证，可自由使用、修改和分发。

---

## 附录：算法公式参考

### 灰度化公式
```
Gray = 0.299×R + 0.587×G + 0.114×B
```

### 对比度调整
```
Pixel' = Pixel × factor + 128 × (1 - factor)
```

### 伽马校正
```
Pixel' = (Pixel/255)^(1/γ) × 255
```

### Sobel 算子
```
Gx = [-1  0  1]    Gy = [-1 -2 -1]
     [-2  0  2]         [ 0  0  0]
     [-1  0  1]         [ 1  2  1]

G = √(Gx² + Gy²)
```

### 高斯函数
```
G(x,y) = (1/2πσ²) × e^(-(x²+y²)/2σ²)
```

---

**感谢您使用数字图像处理工具！** 🎨
