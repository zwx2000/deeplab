/**
 * 数字图像处理核心算法库
 * 类似于Matlab图像处理工具箱的功能
 */

export interface ImageData {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export interface Point {
  x: number;
  y: number;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface HistogramData {
  red: number[];
  green: number[];
  blue: number[];
  gray: number[];
}

// 创建空白图像
export function createImage(width: number, height: number, fill: number = 255): ImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  data.fill(fill);
  return { width, height, data };
}

// 复制图像
export function cloneImage(image: ImageData): ImageData {
  return {
    width: image.width,
    height: image.height,
    data: new Uint8ClampedArray(image.data)
  };
}

// 从Canvas获取图像数据
export function getImageFromCanvas(canvas: HTMLCanvasElement): ImageData {
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return {
    width: imageData.width,
    height: imageData.height,
    data: imageData.data
  };
}

// 将图像绘制到Canvas
export function drawImageToCanvas(image: ImageData, canvas: HTMLCanvasElement): void {
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d')!;
  const imageData = new ImageData(image.data, image.width, image.height);
  ctx.putImageData(imageData, 0, 0);
}

// 获取像素值
export function getPixel(image: ImageData, x: number, y: number): [number, number, number, number] {
  const idx = (y * image.width + x) * 4;
  return [
    image.data[idx],     // R
    image.data[idx + 1], // G
    image.data[idx + 2], // B
    image.data[idx + 3]  // A
  ];
}

// 设置像素值
export function setPixel(image: ImageData, x: number, y: number, r: number, g: number, b: number, a: number = 255): void {
  const idx = (y * image.width + x) * 4;
  image.data[idx] = r;
  image.data[idx + 1] = g;
  image.data[idx + 2] = b;
  image.data[idx + 3] = a;
}

// ==================== 基本图像处理算法 ====================

// 1. 灰度化 (Grayscale)
export function toGrayscale(image: ImageData): ImageData {
  const result = cloneImage(image);
  for (let i = 0; i < result.data.length; i += 4) {
    const gray = Math.round(
      0.299 * result.data[i] + 
      0.587 * result.data[i + 1] + 
      0.114 * result.data[i + 2]
    );
    result.data[i] = gray;
    result.data[i + 1] = gray;
    result.data[i + 2] = gray;
  }
  return result;
}

// 2. 二值化 (Binarization/Thresholding)
export function toBinary(image: ImageData, threshold: number = 128): ImageData {
  const result = cloneImage(image);
  for (let i = 0; i < result.data.length; i += 4) {
    const gray = Math.round(
      0.299 * result.data[i] + 
      0.587 * result.data[i + 1] + 
      0.114 * result.data[i + 2]
    );
    const binary = gray >= threshold ? 255 : 0;
    result.data[i] = binary;
    result.data[i + 1] = binary;
    result.data[i + 2] = binary;
  }
  return result;
}

// 3. 反色 (Invert/Negative)
export function invert(image: ImageData): ImageData {
  const result = cloneImage(image);
  for (let i = 0; i < result.data.length; i += 4) {
    result.data[i] = 255 - result.data[i];
    result.data[i + 1] = 255 - result.data[i + 1];
    result.data[i + 2] = 255 - result.data[i + 2];
  }
  return result;
}

// 4. 亮度调整 (Brightness)
export function adjustBrightness(image: ImageData, value: number): ImageData {
  const result = cloneImage(image);
  for (let i = 0; i < result.data.length; i += 4) {
    result.data[i] = Math.max(0, Math.min(255, result.data[i] + value));
    result.data[i + 1] = Math.max(0, Math.min(255, result.data[i + 1] + value));
    result.data[i + 2] = Math.max(0, Math.min(255, result.data[i + 2] + value));
  }
  return result;
}

// 5. 对比度调整 (Contrast)
export function adjustContrast(image: ImageData, factor: number): ImageData {
  const result = cloneImage(image);
  const intercept = 128 * (1 - factor);
  for (let i = 0; i < result.data.length; i += 4) {
    result.data[i] = Math.max(0, Math.min(255, result.data[i] * factor + intercept));
    result.data[i + 1] = Math.max(0, Math.min(255, result.data[i + 1] * factor + intercept));
    result.data[i + 2] = Math.max(0, Math.min(255, result.data[i + 2] * factor + intercept));
  }
  return result;
}

// 6. 伽马校正 (Gamma Correction)
export function gammaCorrection(image: ImageData, gamma: number): ImageData {
  const result = cloneImage(image);
  const lookupTable = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    lookupTable[i] = Math.round(Math.pow(i / 255, 1 / gamma) * 255);
  }
  for (let i = 0; i < result.data.length; i += 4) {
    result.data[i] = lookupTable[result.data[i]];
    result.data[i + 1] = lookupTable[result.data[i + 1]];
    result.data[i + 2] = lookupTable[result.data[i + 2]];
  }
  return result;
}

// 7. 直方图均衡化 (Histogram Equalization)
export function histogramEqualization(image: ImageData): ImageData {
  const result = cloneImage(image);
  
  // 计算直方图
  const histogram = new Array(256).fill(0);
  for (let i = 0; i < result.data.length; i += 4) {
    const gray = Math.round(0.299 * result.data[i] + 0.587 * result.data[i + 1] + 0.114 * result.data[i + 2]);
    histogram[gray]++;
  }
  
  // 计算累积分布函数
  const cdf = new Array(256);
  cdf[0] = histogram[0];
  for (let i = 1; i < 256; i++) {
    cdf[i] = cdf[i - 1] + histogram[i];
  }
  
  // 找到最小的非零CDF值
  let cdfMin = 0;
  for (let i = 0; i < 256; i++) {
    if (cdf[i] > 0) {
      cdfMin = cdf[i];
      break;
    }
  }
  
  // 创建查找表
  const totalPixels = result.width * result.height;
  const lookupTable = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    if (cdf[i] > 0) {
      lookupTable[i] = Math.round(((cdf[i] - cdfMin) / (totalPixels - cdfMin)) * 255);
    }
  }
  
  // 应用均衡化
  for (let i = 0; i < result.data.length; i += 4) {
    const gray = Math.round(0.299 * result.data[i] + 0.587 * result.data[i + 1] + 0.114 * result.data[i + 2]);
    const newGray = lookupTable[gray];
    result.data[i] = newGray;
    result.data[i + 1] = newGray;
    result.data[i + 2] = newGray;
  }
  
  return result;
}

// ==================== 滤波算法 ====================

// 创建高斯核
function createGaussianKernel(size: number, sigma: number): number[][] {
  const kernel: number[][] = [];
  const center = Math.floor(size / 2);
  let sum = 0;
  
  for (let y = 0; y < size; y++) {
    kernel[y] = [];
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const value = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      kernel[y][x] = value;
      sum += value;
    }
  }
  
  // 归一化
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      kernel[y][x] /= sum;
    }
  }
  
  return kernel;
}

// 通用卷积函数
function convolve(image: ImageData, kernel: number[][]): ImageData {
  const result = cloneImage(image);
  const kSize = kernel.length;
  const kHalf = Math.floor(kSize / 2);
  
  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      let r = 0, g = 0, b = 0;
      
      for (let ky = 0; ky < kSize; ky++) {
        for (let kx = 0; kx < kSize; kx++) {
          const px = Math.max(0, Math.min(image.width - 1, x + kx - kHalf));
          const py = Math.max(0, Math.min(image.height - 1, y + ky - kHalf));
          const idx = (py * image.width + px) * 4;
          const weight = kernel[ky][kx];
          
          r += image.data[idx] * weight;
          g += image.data[idx + 1] * weight;
          b += image.data[idx + 2] * weight;
        }
      }
      
      const idx = (y * image.width + x) * 4;
      result.data[idx] = Math.max(0, Math.min(255, Math.round(r)));
      result.data[idx + 1] = Math.max(0, Math.min(255, Math.round(g)));
      result.data[idx + 2] = Math.max(0, Math.min(255, Math.round(b)));
    }
  }
  
  return result;
}

// 8. 高斯模糊 (Gaussian Blur)
export function gaussianBlur(image: ImageData, sigma: number = 1.5, kernelSize?: number): ImageData {
  const size = kernelSize || Math.ceil(sigma * 3) * 2 + 1;
  const kernel = createGaussianKernel(size, sigma);
  return convolve(image, kernel);
}

// 9. 均值滤波 (Mean/Average Filter)
export function meanFilter(image: ImageData, size: number = 3): ImageData {
  const kernel: number[][] = [];
  const weight = 1 / (size * size);
  for (let y = 0; y < size; y++) {
    kernel[y] = [];
    for (let x = 0; x < size; x++) {
      kernel[y][x] = weight;
    }
  }
  return convolve(image, kernel);
}

// 10. 中值滤波 (Median Filter)
export function medianFilter(image: ImageData, size: number = 3): ImageData {
  const result = cloneImage(image);
  const kHalf = Math.floor(size / 2);
  
  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      const rValues: number[] = [];
      const gValues: number[] = [];
      const bValues: number[] = [];
      
      for (let ky = -kHalf; ky <= kHalf; ky++) {
        for (let kx = -kHalf; kx <= kHalf; kx++) {
          const px = Math.max(0, Math.min(image.width - 1, x + kx));
          const py = Math.max(0, Math.min(image.height - 1, y + ky));
          const idx = (py * image.width + px) * 4;
          
          rValues.push(image.data[idx]);
          gValues.push(image.data[idx + 1]);
          bValues.push(image.data[idx + 2]);
        }
      }
      
      rValues.sort((a, b) => a - b);
      gValues.sort((a, b) => a - b);
      bValues.sort((a, b) => a - b);
      
      const mid = Math.floor(rValues.length / 2);
      const idx = (y * image.width + x) * 4;
      result.data[idx] = rValues[mid];
      result.data[idx + 1] = gValues[mid];
      result.data[idx + 2] = bValues[mid];
    }
  }
  
  return result;
}

// 11. 锐化 (Sharpening)
export function sharpen(image: ImageData, amount: number = 1): ImageData {
  const kernel = [
    [0, -amount, 0],
    [-amount, 1 + 4 * amount, -amount],
    [0, -amount, 0]
  ];
  return convolve(image, kernel);
}

// 12. 自定义卷积核滤波
export function customKernelFilter(image: ImageData, kernel: number[][]): ImageData {
  return convolve(image, kernel);
}

// ==================== 边缘检测算法 ====================

// 13. Sobel边缘检测
export function sobelEdgeDetection(image: ImageData): ImageData {
  const gray = toGrayscale(image);
  const result = createImage(image.width, image.height, 0);
  
  const sobelX = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
  ];
  
  const sobelY = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
  ];
  
  for (let y = 1; y < image.height - 1; y++) {
    for (let x = 1; x < image.width - 1; x++) {
      let gx = 0, gy = 0;
      
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          const px = x + kx - 1;
          const py = y + ky - 1;
          const idx = (py * image.width + px) * 4;
          const pixel = gray.data[idx];
          
          gx += pixel * sobelX[ky][kx];
          gy += pixel * sobelY[ky][kx];
        }
      }
      
      const magnitude = Math.min(255, Math.sqrt(gx * gx + gy * gy));
      const idx = (y * image.width + x) * 4;
      result.data[idx] = magnitude;
      result.data[idx + 1] = magnitude;
      result.data[idx + 2] = magnitude;
    }
  }
  
  return result;
}

// 14. Prewitt边缘检测
export function prewittEdgeDetection(image: ImageData): ImageData {
  const gray = toGrayscale(image);
  const result = createImage(image.width, image.height, 0);
  
  const prewittX = [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
  ];
  
  const prewittY = [
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
  ];
  
  for (let y = 1; y < image.height - 1; y++) {
    for (let x = 1; x < image.width - 1; x++) {
      let gx = 0, gy = 0;
      
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          const px = x + kx - 1;
          const py = y + ky - 1;
          const idx = (py * image.width + px) * 4;
          const pixel = gray.data[idx];
          
          gx += pixel * prewittX[ky][kx];
          gy += pixel * prewittY[ky][kx];
        }
      }
      
      const magnitude = Math.min(255, Math.sqrt(gx * gx + gy * gy));
      const idx = (y * image.width + x) * 4;
      result.data[idx] = magnitude;
      result.data[idx + 1] = magnitude;
      result.data[idx + 2] = magnitude;
    }
  }
  
  return result;
}

// 15. Laplacian边缘检测
export function laplacianEdgeDetection(image: ImageData): ImageData {
  const gray = toGrayscale(image);
  
  const kernel = [
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
  ];
  
  const convolved = convolve(gray, kernel);
  
  // 取绝对值
  for (let i = 0; i < convolved.data.length; i += 4) {
    const val = Math.abs(convolved.data[i]);
    convolved.data[i] = Math.min(255, val);
    convolved.data[i + 1] = Math.min(255, val);
    convolved.data[i + 2] = Math.min(255, val);
  }
  
  return convolved;
}

// 16. Canny边缘检测 (简化版)
export function cannyEdgeDetection(image: ImageData, lowThreshold: number = 50, highThreshold: number = 150): ImageData {
  // Step 1: 高斯模糊
  const blurred = gaussianBlur(image, 1.4, 5);
  
  // Step 2: 计算梯度
  const gray = toGrayscale(blurred);
  const gradientX = createImage(image.width, image.height, 0);
  const gradientY = createImage(image.width, image.height, 0);
  const magnitude = createImage(image.width, image.height, 0);
  const direction = new Float32Array(image.width * image.height);
  
  const sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
  const sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
  
  for (let y = 1; y < image.height - 1; y++) {
    for (let x = 1; x < image.width - 1; x++) {
      let gx = 0, gy = 0;
      
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          const px = x + kx - 1;
          const py = y + ky - 1;
          const idx = (py * image.width + px) * 4;
          const pixel = gray.data[idx];
          
          gx += pixel * sobelX[ky][kx];
          gy += pixel * sobelY[ky][kx];
        }
      }
      
      const mag = Math.sqrt(gx * gx + gy * gy);
      const idx = (y * image.width + x) * 4;
      magnitude.data[idx] = mag;
      magnitude.data[idx + 1] = mag;
      magnitude.data[idx + 2] = mag;
      
      direction[y * image.width + x] = Math.atan2(gy, gx);
    }
  }
  
  // Step 3: 非极大值抑制
  const nms = createImage(image.width, image.height, 0);
  
  for (let y = 1; y < image.height - 1; y++) {
    for (let x = 1; x < image.width - 1; x++) {
      const angle = direction[y * image.width + x] * 180 / Math.PI;
      const idx = (y * image.width + x) * 4;
      const mag = magnitude.data[idx];
      
      let neighbor1 = 0, neighbor2 = 0;
      
      if ((angle >= -22.5 && angle < 22.5) || (angle >= 157.5 || angle < -157.5)) {
        neighbor1 = magnitude.data[((y) * image.width + x + 1) * 4];
        neighbor2 = magnitude.data[((y) * image.width + x - 1) * 4];
      } else if ((angle >= 22.5 && angle < 67.5) || (angle >= -157.5 && angle < -112.5)) {
        neighbor1 = magnitude.data[((y + 1) * image.width + x - 1) * 4];
        neighbor2 = magnitude.data[((y - 1) * image.width + x + 1) * 4];
      } else if ((angle >= 67.5 && angle < 112.5) || (angle >= -112.5 && angle < -67.5)) {
        neighbor1 = magnitude.data[((y + 1) * image.width + x) * 4];
        neighbor2 = magnitude.data[((y - 1) * image.width + x) * 4];
      } else {
        neighbor1 = magnitude.data[((y - 1) * image.width + x - 1) * 4];
        neighbor2 = magnitude.data[((y + 1) * image.width + x + 1) * 4];
      }
      
      if (mag >= neighbor1 && mag >= neighbor2) {
        nms.data[idx] = mag;
        nms.data[idx + 1] = mag;
        nms.data[idx + 2] = mag;
      }
    }
  }
  
  // Step 4: 双阈值检测
  const result = createImage(image.width, image.height, 0);
  
  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      const idx = (y * image.width + x) * 4;
      const mag = nms.data[idx];
      
      if (mag >= highThreshold) {
        result.data[idx] = 255;
        result.data[idx + 1] = 255;
        result.data[idx + 2] = 255;
      } else if (mag >= lowThreshold) {
        // 检查8邻域是否有强边缘
        let hasStrongEdge = false;
        for (let dy = -1; dy <= 1 && !hasStrongEdge; dy++) {
          for (let dx = -1; dx <= 1 && !hasStrongEdge; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx;
            const ny = y + dy;
            if (nx >= 0 && nx < image.width && ny >= 0 && ny < image.height) {
              const nIdx = (ny * image.width + nx) * 4;
              if (nms.data[nIdx] >= highThreshold) {
                hasStrongEdge = true;
              }
            }
          }
        }
        
        if (hasStrongEdge) {
          result.data[idx] = 255;
          result.data[idx + 1] = 255;
          result.data[idx + 2] = 255;
        }
      }
    }
  }
  
  return result;
}

// ==================== 形态学运算 ====================

// 17. 腐蚀 (Erosion)
export function erode(image: ImageData, kernelSize: number = 3): ImageData {
  const binary = toBinary(image, 128);
  const result = createImage(image.width, image.height, 255);
  const kHalf = Math.floor(kernelSize / 2);
  
  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      let minVal = 255;
      
      for (let ky = -kHalf; ky <= kHalf; ky++) {
        for (let kx = -kHalf; kx <= kHalf; kx++) {
          const px = Math.max(0, Math.min(image.width - 1, x + kx));
          const py = Math.max(0, Math.min(image.height - 1, y + ky));
          const idx = (py * image.width + px) * 4;
          minVal = Math.min(minVal, binary.data[idx]);
        }
      }
      
      const idx = (y * image.width + x) * 4;
      result.data[idx] = minVal;
      result.data[idx + 1] = minVal;
      result.data[idx + 2] = minVal;
    }
  }
  
  return result;
}

// 18. 膨胀 (Dilation)
export function dilate(image: ImageData, kernelSize: number = 3): ImageData {
  const binary = toBinary(image, 128);
  const result = createImage(image.width, image.height, 0);
  const kHalf = Math.floor(kernelSize / 2);
  
  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      let maxVal = 0;
      
      for (let ky = -kHalf; ky <= kHalf; ky++) {
        for (let kx = -kHalf; kx <= kHalf; kx++) {
          const px = Math.max(0, Math.min(image.width - 1, x + kx));
          const py = Math.max(0, Math.min(image.height - 1, y + ky));
          const idx = (py * image.width + px) * 4;
          maxVal = Math.max(maxVal, binary.data[idx]);
        }
      }
      
      const idx = (y * image.width + x) * 4;
      result.data[idx] = maxVal;
      result.data[idx + 1] = maxVal;
      result.data[idx + 2] = maxVal;
    }
  }
  
  return result;
}

// 19. 开运算 (Opening)
export function opening(image: ImageData, kernelSize: number = 3): ImageData {
  return dilate(erode(image, kernelSize), kernelSize);
}

// 20. 闭运算 (Closing)
export function closing(image: ImageData, kernelSize: number = 3): ImageData {
  return erode(dilate(image, kernelSize), kernelSize);
}

// ==================== 几何变换 ====================

// 21. 图像旋转
export function rotate(image: ImageData, angle: number): ImageData {
  const radians = angle * Math.PI / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  
  // 计算新图像尺寸
  const newWidth = Math.ceil(Math.abs(image.width * cos) + Math.abs(image.height * sin));
  const newHeight = Math.ceil(Math.abs(image.width * sin) + Math.abs(image.height * cos));
  
  const result = createImage(newWidth, newHeight, 255);
  const centerX = newWidth / 2;
  const centerY = newHeight / 2;
  const srcCenterX = image.width / 2;
  const srcCenterY = image.height / 2;
  
  for (let y = 0; y < newHeight; y++) {
    for (let x = 0; x < newWidth; x++) {
      const srcX = cos * (x - centerX) + sin * (y - centerY) + srcCenterX;
      const srcY = -sin * (x - centerX) + cos * (y - centerY) + srcCenterY;
      
      if (srcX >= 0 && srcX < image.width && srcY >= 0 && srcY < image.height) {
        // 双线性插值
        const x0 = Math.floor(srcX);
        const y0 = Math.floor(srcY);
        const x1 = Math.min(x0 + 1, image.width - 1);
        const y1 = Math.min(y0 + 1, image.height - 1);
        const fx = srcX - x0;
        const fy = srcY - y0;
        
        const idx00 = (y0 * image.width + x0) * 4;
        const idx01 = (y0 * image.width + x1) * 4;
        const idx10 = (y1 * image.width + x0) * 4;
        const idx11 = (y1 * image.width + x1) * 4;
        
        const idx = (y * newWidth + x) * 4;
        
        for (let c = 0; c < 3; c++) {
          const value = 
            image.data[idx00 + c] * (1 - fx) * (1 - fy) +
            image.data[idx01 + c] * fx * (1 - fy) +
            image.data[idx10 + c] * (1 - fx) * fy +
            image.data[idx11 + c] * fx * fy;
          result.data[idx + c] = Math.round(value);
        }
      }
    }
  }
  
  return result;
}

// 22. 图像缩放
export function scale(image: ImageData, scaleX: number, scaleY: number): ImageData {
  const newWidth = Math.round(image.width * scaleX);
  const newHeight = Math.round(image.height * scaleY);
  const result = createImage(newWidth, newHeight, 255);
  
  for (let y = 0; y < newHeight; y++) {
    for (let x = 0; x < newWidth; x++) {
      const srcX = x / scaleX;
      const srcY = y / scaleY;
      
      // 双线性插值
      const x0 = Math.floor(srcX);
      const y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, image.width - 1);
      const y1 = Math.min(y0 + 1, image.height - 1);
      const fx = srcX - x0;
      const fy = srcY - y0;
      
      const idx00 = (y0 * image.width + x0) * 4;
      const idx01 = (y0 * image.width + x1) * 4;
      const idx10 = (y1 * image.width + x0) * 4;
      const idx11 = (y1 * image.width + x1) * 4;
      
      const idx = (y * newWidth + x) * 4;
      
      for (let c = 0; c < 3; c++) {
        const value = 
          image.data[idx00 + c] * (1 - fx) * (1 - fy) +
          image.data[idx01 + c] * fx * (1 - fy) +
          image.data[idx10 + c] * (1 - fx) * fy +
          image.data[idx11 + c] * fx * fy;
        result.data[idx + c] = Math.round(value);
      }
    }
  }
  
  return result;
}

// 23. 水平翻转
export function flipHorizontal(image: ImageData): ImageData {
  const result = createImage(image.width, image.height);
  
  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      const srcIdx = (y * image.width + x) * 4;
      const dstIdx = (y * image.width + (image.width - 1 - x)) * 4;
      
      result.data[dstIdx] = image.data[srcIdx];
      result.data[dstIdx + 1] = image.data[srcIdx + 1];
      result.data[dstIdx + 2] = image.data[srcIdx + 2];
      result.data[dstIdx + 3] = image.data[srcIdx + 3];
    }
  }
  
  return result;
}

// 24. 垂直翻转
export function flipVertical(image: ImageData): ImageData {
  const result = createImage(image.width, image.height);
  
  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      const srcIdx = (y * image.width + x) * 4;
      const dstIdx = ((image.height - 1 - y) * image.width + x) * 4;
      
      result.data[dstIdx] = image.data[srcIdx];
      result.data[dstIdx + 1] = image.data[srcIdx + 1];
      result.data[dstIdx + 2] = image.data[srcIdx + 2];
      result.data[dstIdx + 3] = image.data[srcIdx + 3];
    }
  }
  
  return result;
}

// ==================== 噪声处理 ====================

// 25. 添加椒盐噪声
export function addSaltPepperNoise(image: ImageData, density: number = 0.05): ImageData {
  const result = cloneImage(image);
  const numPixels = Math.round(image.width * image.height * density);
  
  for (let i = 0; i < numPixels; i++) {
    const x = Math.floor(Math.random() * image.width);
    const y = Math.floor(Math.random() * image.height);
    const idx = (y * image.width + x) * 4;
    const value = Math.random() < 0.5 ? 0 : 255;
    
    result.data[idx] = value;
    result.data[idx + 1] = value;
    result.data[idx + 2] = value;
  }
  
  return result;
}

// 26. 添加高斯噪声
export function addGaussianNoise(image: ImageData, mean: number = 0, stdDev: number = 25): ImageData {
  const result = cloneImage(image);
  
  // Box-Muller变换生成高斯随机数
  const generateGaussian = () => {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  
  for (let i = 0; i < result.data.length; i += 4) {
    const noise = generateGaussian() * stdDev + mean;
    result.data[i] = Math.max(0, Math.min(255, result.data[i] + noise));
    result.data[i + 1] = Math.max(0, Math.min(255, result.data[i + 1] + noise));
    result.data[i + 2] = Math.max(0, Math.min(255, result.data[i + 2] + noise));
  }
  
  return result;
}

// ==================== 颜色空间转换 ====================

// RGB to HSL
export function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  r /= 255;
  g /= 255;
  b /= 255;
  
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h = 0, s = 0;
  const l = (max + min) / 2;
  
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    
    switch (max) {
      case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
      case g: h = ((b - r) / d + 2) / 6; break;
      case b: h = ((r - g) / d + 4) / 6; break;
    }
  }
  
  return [h * 360, s * 100, l * 100];
}

// HSL to RGB
export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  h /= 360;
  s /= 100;
  l /= 100;
  
  let r, g, b;
  
  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p: number, q: number, t: number) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };
    
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// 27. 色调调整
export function adjustHue(image: ImageData, degrees: number): ImageData {
  const result = cloneImage(image);
  
  for (let i = 0; i < result.data.length; i += 4) {
    const [h, s, l] = rgbToHsl(result.data[i], result.data[i + 1], result.data[i + 2]);
    const [r, g, b] = hslToRgb((h + degrees) % 360, s, l);
    
    result.data[i] = r;
    result.data[i + 1] = g;
    result.data[i + 2] = b;
  }
  
  return result;
}

// 28. 饱和度调整
export function adjustSaturation(image: ImageData, factor: number): ImageData {
  const result = cloneImage(image);
  
  for (let i = 0; i < result.data.length; i += 4) {
    const [h, s, l] = rgbToHsl(result.data[i], result.data[i + 1], result.data[i + 2]);
    const [r, g, b] = hslToRgb(h, Math.max(0, Math.min(100, s * factor)), l);
    
    result.data[i] = r;
    result.data[i + 1] = g;
    result.data[i + 2] = b;
  }
  
  return result;
}

// ==================== 直方图计算 ====================

// 计算直方图
export function calculateHistogram(image: ImageData): HistogramData {
  const red = new Array(256).fill(0);
  const green = new Array(256).fill(0);
  const blue = new Array(256).fill(0);
  const gray = new Array(256).fill(0);
  
  for (let i = 0; i < image.data.length; i += 4) {
    red[image.data[i]]++;
    green[image.data[i + 1]]++;
    blue[image.data[i + 2]]++;
    
    const g = Math.round(0.299 * image.data[i] + 0.587 * image.data[i + 1] + 0.114 * image.data[i + 2]);
    gray[g]++;
  }
  
  return { red, green, blue, gray };
}

// ==================== ROI裁剪 ====================

// 29. 图像裁剪
export function crop(image: ImageData, rect: Rect): ImageData {
  const result = createImage(rect.width, rect.height);
  
  for (let y = 0; y < rect.height; y++) {
    for (let x = 0; x < rect.width; x++) {
      const srcX = rect.x + x;
      const srcY = rect.y + y;
      
      if (srcX >= 0 && srcX < image.width && srcY >= 0 && srcY < image.height) {
        const srcIdx = (srcY * image.width + srcX) * 4;
        const dstIdx = (y * rect.width + x) * 4;
        
        result.data[dstIdx] = image.data[srcIdx];
        result.data[dstIdx + 1] = image.data[srcIdx + 1];
        result.data[dstIdx + 2] = image.data[srcIdx + 2];
        result.data[dstIdx + 3] = image.data[srcIdx + 3];
      }
    }
  }
  
  return result;
}

// ==================== 特效 ====================

// 30. 浮雕效果
export function emboss(image: ImageData): ImageData {
  const kernel = [
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
  ];
  const result = convolve(image, kernel);
  
  // 添加128偏移
  for (let i = 0; i < result.data.length; i += 4) {
    result.data[i] = Math.max(0, Math.min(255, result.data[i] + 128));
    result.data[i + 1] = Math.max(0, Math.min(255, result.data[i + 1] + 128));
    result.data[i + 2] = Math.max(0, Math.min(255, result.data[i + 2] + 128));
  }
  
  return result;
}

// 31. 油画效果
export function oilPainting(image: ImageData, radius: number = 3, intensity: number = 20): ImageData {
  const result = cloneImage(image);
  
  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      const intensityCount = new Array(intensity + 1).fill(0);
      const sumR = new Array(intensity + 1).fill(0);
      const sumG = new Array(intensity + 1).fill(0);
      const sumB = new Array(intensity + 1).fill(0);
      
      for (let ky = -radius; ky <= radius; ky++) {
        for (let kx = -radius; kx <= radius; kx++) {
          const px = Math.max(0, Math.min(image.width - 1, x + kx));
          const py = Math.max(0, Math.min(image.height - 1, y + ky));
          const idx = (py * image.width + px) * 4;
          
          const r = image.data[idx];
          const g = image.data[idx + 1];
          const b = image.data[idx + 2];
          
          const curIntensity = Math.floor(((r + g + b) / 3) * intensity / 255);
          intensityCount[curIntensity]++;
          sumR[curIntensity] += r;
          sumG[curIntensity] += g;
          sumB[curIntensity] += b;
        }
      }
      
      let maxCount = 0;
      let maxIndex = 0;
      for (let i = 0; i <= intensity; i++) {
        if (intensityCount[i] > maxCount) {
          maxCount = intensityCount[i];
          maxIndex = i;
        }
      }
      
      const idx = (y * image.width + x) * 4;
      result.data[idx] = Math.round(sumR[maxIndex] / maxCount);
      result.data[idx + 1] = Math.round(sumG[maxIndex] / maxCount);
      result.data[idx + 2] = Math.round(sumB[maxIndex] / maxCount);
    }
  }
  
  return result;
}

// 32. 马赛克效果
export function mosaic(image: ImageData, blockSize: number = 10): ImageData {
  const result = cloneImage(image);
  
  for (let y = 0; y < image.height; y += blockSize) {
    for (let x = 0; x < image.width; x += blockSize) {
      let sumR = 0, sumG = 0, sumB = 0, count = 0;
      
      for (let by = 0; by < blockSize && y + by < image.height; by++) {
        for (let bx = 0; bx < blockSize && x + bx < image.width; bx++) {
          const idx = ((y + by) * image.width + (x + bx)) * 4;
          sumR += image.data[idx];
          sumG += image.data[idx + 1];
          sumB += image.data[idx + 2];
          count++;
        }
      }
      
      const avgR = Math.round(sumR / count);
      const avgG = Math.round(sumG / count);
      const avgB = Math.round(sumB / count);
      
      for (let by = 0; by < blockSize && y + by < image.height; by++) {
        for (let bx = 0; bx < blockSize && x + bx < image.width; bx++) {
          const idx = ((y + by) * image.width + (x + bx)) * 4;
          result.data[idx] = avgR;
          result.data[idx + 1] = avgG;
          result.data[idx + 2] = avgB;
        }
      }
    }
  }
  
  return result;
}

// 33. 边缘增强
export function edgeEnhance(image: ImageData, strength: number = 1): ImageData {
  const edges = sobelEdgeDetection(image);
  const result = cloneImage(image);
  
  for (let i = 0; i < result.data.length; i += 4) {
    result.data[i] = Math.min(255, result.data[i] + edges.data[i] * strength);
    result.data[i + 1] = Math.min(255, result.data[i + 1] + edges.data[i + 1] * strength);
    result.data[i + 2] = Math.min(255, result.data[i + 2] + edges.data[i + 2] * strength);
  }
  
  return result;
}

// 获取图像类型定义
export type ImageProcessor = (image: ImageData, ...args: unknown[]) => ImageData;
