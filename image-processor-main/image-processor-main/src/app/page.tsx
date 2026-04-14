'use client';

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Progress } from '@/components/ui/progress';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuCheckboxItem,
} from '@/components/ui/dropdown-menu';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { useToast } from '@/hooks/use-toast';
import {
  Upload,
  Download,
  RotateCw,
  RotateCcw,
  FlipHorizontal,
  FlipVertical,
  Sun,
  Moon,
  Contrast,
  Palette,
  Circle,
  Square,
  Minus,
  Plus,
  Undo,
  Redo,
  ZoomIn,
  ZoomOut,
  Move,
  MousePointer,
  Crop,
  Paintbrush,
  Eraser,
  Save,
  FolderOpen,
  Image as ImageIcon,
  Settings,
  Info,
  Keyboard,
  HelpCircle,
  Layers,
  SlidersHorizontal,
  Filter,
  Sparkles,
  Grid3X3,
  Eye,
  EyeOff,
  RefreshCw,
  Copy,
  Clipboard,
  Scissors,
  Loader2,
  X,
} from 'lucide-react';

// 导入图像处理函数
import {
  ImageData as ImageDataType,
  createImage,
  cloneImage,
  toGrayscale,
  toBinary,
  invert,
  adjustBrightness,
  adjustContrast,
  gammaCorrection,
  histogramEqualization,
  gaussianBlur,
  meanFilter,
  medianFilter,
  sharpen,
  sobelEdgeDetection,
  prewittEdgeDetection,
  laplacianEdgeDetection,
  cannyEdgeDetection,
  erode,
  dilate,
  opening,
  closing,
  rotate,
  scale,
  flipHorizontal,
  flipVertical,
  addSaltPepperNoise,
  addGaussianNoise,
  adjustHue,
  adjustSaturation,
  calculateHistogram,
  crop,
  emboss,
  oilPainting,
  mosaic,
  edgeEnhance,
  Rect,
} from '@/lib/imageProcessing';

// 工具类型
type Tool = 'pointer' | 'move' | 'crop' | 'brush' | 'eraser' | 'picker';

// 历史记录项
interface HistoryItem {
  id: string;
  name: string;
  imageData: ImageDataType | null;
  timestamp: Date;
}

// 直方图数据
interface HistogramData {
  red: number[];
  green: number[];
  blue: number[];
  gray: number[];
}

// 快捷键定义
const SHORTCUTS = [
  { key: 'Ctrl+O', action: '打开图片' },
  { key: 'Ctrl+S', action: '保存图片' },
  { key: 'Ctrl+Z', action: '撤销' },
  { key: 'Ctrl+Y', action: '重做' },
  { key: 'Ctrl+C', action: '复制' },
  { key: 'Ctrl+V', action: '粘贴' },
  { key: 'V', action: '选择工具' },
  { key: 'M', action: '移动工具' },
  { key: 'C', action: '裁剪工具' },
  { key: 'B', action: '画笔工具' },
  { key: 'E', action: '橡皮擦' },
  { key: 'I', action: '吸管工具' },
  { key: '+', action: '放大' },
  { key: '-', action: '缩小' },
  { key: '0', action: '实际大小' },
  { key: 'Ctrl+R', action: '顺时针旋转90°' },
  { key: 'Ctrl+Shift+R', action: '逆时针旋转90°' },
  { key: 'Ctrl+H', action: '水平翻转' },
  { key: 'Ctrl+Shift+H', action: '垂直翻转' },
  { key: 'Ctrl+G', action: '灰度化' },
  { key: 'Ctrl+I', action: '反色' },
  { key: 'Delete', action: '重置图像' },
];

export default function ImageProcessorApp() {
  // 状态
  const [originalImage, setOriginalImage] = useState<ImageDataType | null>(null);
  const [currentImage, setCurrentImage] = useState<ImageDataType | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [zoom, setZoom] = useState(1);
  const [tool, setTool] = useState<Tool>('pointer');
  const [showHistogram, setShowHistogram] = useState(true);
  const [showGrid, setShowGrid] = useState(false);
  
  // 滤镜参数
  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(1);
  const [gamma, setGamma] = useState(1);
  const [threshold, setThreshold] = useState(128);
  const [blurSigma, setBlurSigma] = useState(1.5);
  const [sharpenAmount, setSharpenAmount] = useState(1);
  const [rotationAngle, setRotationAngle] = useState(90);
  const [scaleRatio, setScaleRatio] = useState(100);
  const [noiseDensity, setNoiseDensity] = useState(0.05);
  const [noiseStdDev, setNoiseStdDev] = useState(25);
  const [hueDegrees, setHueDegrees] = useState(0);
  const [saturationFactor, setSaturationFactor] = useState(1);
  const [oilRadius, setOilRadius] = useState(3);
  const [oilIntensity, setOilIntensity] = useState(20);
  const [mosaicSize, setMosaicSize] = useState(10);
  const [morphKernelSize, setMorphKernelSize] = useState(3);
  
  // ROI裁剪
  const [cropRect, setCropRect] = useState<Rect | null>(null);
  const [isCropping, setIsCropping] = useState(false);
  const [cropStart, setCropStart] = useState<{ x: number; y: number } | null>(null);
  
  // 画笔工具
  const [brushSize, setBrushSize] = useState(10);
  const [brushColor, setBrushColor] = useState('#ff0000');
  const [isDrawing, setIsDrawing] = useState(false);
  const [lastDrawPoint, setLastDrawPoint] = useState<{ x: number; y: number } | null>(null);
  
  // 吸管工具拾取的颜色
  const [pickedColor, setPickedColor] = useState<{ r: number; g: number; b: number } | null>(null);
  
  // 加载状态
  const [isLoading, setIsLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  
  // Refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const { toast } = useToast();

  // 计算直方图 - 使用useMemo避免在effect中设置状态
  const histogramData = useMemo(() => {
    if (currentImage) {
      return calculateHistogram(currentImage);
    }
    return null;
  }, [currentImage]);

  // 绘制图像到画布
  useEffect(() => {
    if (currentImage && canvasRef.current) {
      const canvas = canvasRef.current;
      canvas.width = currentImage.width;
      canvas.height = currentImage.height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        const imageData = new ImageData(currentImage.data, currentImage.width, currentImage.height);
        ctx.putImageData(imageData, 0, 0);
        
        // 绘制网格
        if (showGrid) {
          ctx.strokeStyle = 'rgba(128, 128, 128, 0.3)';
          ctx.lineWidth = 1;
          const gridSize = 50;
          for (let x = 0; x < currentImage.width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, currentImage.height);
            ctx.stroke();
          }
          for (let y = 0; y < currentImage.height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(currentImage.width, y);
            ctx.stroke();
          }
        }
      }
    }
  }, [currentImage, showGrid]);

  // 绘制覆盖层（ROI、画笔等）
  useEffect(() => {
    if (overlayCanvasRef.current && currentImage) {
      const canvas = overlayCanvasRef.current;
      canvas.width = currentImage.width;
      canvas.height = currentImage.height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // 绘制裁剪框
        if (cropRect && tool === 'crop') {
          ctx.strokeStyle = '#00ff00';
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 5]);
          ctx.strokeRect(cropRect.x, cropRect.y, cropRect.width, cropRect.height);
          
          // 绘制裁剪框角点
          ctx.setLineDash([]);
          ctx.fillStyle = '#00ff00';
          const handleSize = 8;
          ctx.fillRect(cropRect.x - handleSize/2, cropRect.y - handleSize/2, handleSize, handleSize);
          ctx.fillRect(cropRect.x + cropRect.width - handleSize/2, cropRect.y - handleSize/2, handleSize, handleSize);
          ctx.fillRect(cropRect.x - handleSize/2, cropRect.y + cropRect.height - handleSize/2, handleSize, handleSize);
          ctx.fillRect(cropRect.x + cropRect.width - handleSize/2, cropRect.y + cropRect.height - handleSize/2, handleSize, handleSize);
        }
      }
    }
  }, [cropRect, currentImage, tool]);

  // 添加历史记录
  const addHistory = useCallback((name: string, imageData: ImageDataType | null) => {
    const newItem: HistoryItem = {
      id: Date.now().toString(),
      name,
      imageData: imageData ? cloneImage(imageData) : null,
      timestamp: new Date(),
    };
    
    setHistory(prev => {
      const newHistory = [...prev.slice(0, historyIndex + 1), newItem];
      return newHistory.slice(-50); // 最多保留50条历史
    });
    setHistoryIndex(prev => Math.min(prev + 1, 49));
  }, [historyIndex]);

  // 撤销
  const undo = useCallback(() => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      setCurrentImage(history[newIndex]?.imageData ? cloneImage(history[newIndex].imageData!) : null);
    }
  }, [historyIndex, history]);

  // 重做
  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      setCurrentImage(history[newIndex]?.imageData ? cloneImage(history[newIndex].imageData!) : null);
    }
  }, [historyIndex, history]);

  // 应用滤镜
  const applyFilter = useCallback((filterName: string, filterFn: () => ImageDataType | null) => {
    if (!currentImage) {
      toast({
        title: '提示',
        description: '请先打开一张图片',
      });
      return;
    }
    
    setIsLoading(true);
    setLoadingProgress(50);
    
    // 使用setTimeout让UI有时间更新
    setTimeout(() => {
      const result = filterFn();
      setLoadingProgress(100);
      
      setTimeout(() => {
        if (result) {
          setCurrentImage(result);
          addHistory(filterName, result);
          toast({
            title: '成功',
            description: `已应用 ${filterName} 滤镜`,
          });
        }
        setIsLoading(false);
        setLoadingProgress(0);
      }, 100);
    }, 50);
  }, [currentImage, addHistory, toast]);

  // 处理文件（通用）
  const processFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      toast({
        title: '错误',
        description: '请选择图片文件',
        variant: 'destructive',
      });
      return;
    }
    
    setIsLoading(true);
    setLoadingProgress(30);
    
    const img = new Image();
    img.onload = () => {
      setLoadingProgress(70);
      
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      
      if (ctx) {
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        const newImage: ImageDataType = {
          width: img.width,
          height: img.height,
          data: new Uint8ClampedArray(imageData.data),
        };
        
        setLoadingProgress(100);
        
        setTimeout(() => {
          setOriginalImage(cloneImage(newImage));
          setCurrentImage(newImage);
          setHistory([]);
          setHistoryIndex(-1);
          addHistory('打开图片', newImage);
          setIsLoading(false);
          setLoadingProgress(0);
          toast({
            title: '成功',
            description: `已打开图片: ${file.name} (${img.width}x${img.height})`,
          });
        }, 100);
      }
    };
    
    img.onerror = () => {
      setIsLoading(false);
      setLoadingProgress(0);
      toast({
        title: '错误',
        description: '无法加载图片',
        variant: 'destructive',
      });
    };
    
    img.src = URL.createObjectURL(file);
  }, [addHistory, toast]);

  // 文件上传处理
  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    processFile(file);
    // 重置input以允许重新选择同一文件
    e.target.value = '';
  }, [processFile]);

  // 从URL加载图片（示例图片）
  const loadFromUrl = useCallback(async (url: string, name: string) => {
    setIsLoading(true);
    setLoadingProgress(20);
    
    try {
      const response = await fetch(url);
      setLoadingProgress(40);
      
      const blob = await response.blob();
      setLoadingProgress(60);
      
      const file = new File([blob], name, { type: blob.type });
      processFile(file);
    } catch (error) {
      setIsLoading(false);
      setLoadingProgress(0);
      toast({
        title: '错误',
        description: '无法加载示例图片',
        variant: 'destructive',
      });
    }
  }, [processFile, toast]);

  // 拖放事件处理
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      processFile(file);
    }
  }, [processFile]);

  // 下载图片
  const handleDownload = useCallback(() => {
    if (!currentImage || !canvasRef.current) return;
    
    const link = document.createElement('a');
    link.download = `processed-image-${Date.now()}.png`;
    link.href = canvasRef.current.toDataURL('image/png');
    link.click();
    
    toast({
      title: '成功',
      description: '图片已保存',
    });
  }, [currentImage, toast]);

  // 鼠标事件处理
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!currentImage || !overlayCanvasRef.current) return;
    
    const canvas = overlayCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / zoom);
    const y = Math.floor((e.clientY - rect.top) / zoom);
    
    if (tool === 'crop') {
      setIsCropping(true);
      setCropStart({ x, y });
      setCropRect({ x, y, width: 0, height: 0 });
    } else if (tool === 'brush' || tool === 'eraser') {
      setIsDrawing(true);
      setLastDrawPoint({ x, y });
      
      // 绘制起始点
      const ctx = canvasRef.current?.getContext('2d');
      if (ctx) {
        ctx.beginPath();
        ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
        if (tool === 'eraser') {
          const pixelIdx = (y * currentImage.width + x) * 4;
          const r = originalImage?.data[pixelIdx] || 255;
          const g = originalImage?.data[pixelIdx + 1] || 255;
          const b = originalImage?.data[pixelIdx + 2] || 255;
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        } else {
          ctx.fillStyle = brushColor;
        }
        ctx.fill();
        
        // 更新当前图像数据
        const imageData = ctx.getImageData(0, 0, currentImage.width, currentImage.height);
        setCurrentImage({
          width: currentImage.width,
          height: currentImage.height,
          data: new Uint8ClampedArray(imageData.data),
        });
      }
    } else if (tool === 'picker') {
      if (x >= 0 && x < currentImage.width && y >= 0 && y < currentImage.height) {
        const idx = (y * currentImage.width + x) * 4;
        setPickedColor({
          r: currentImage.data[idx],
          g: currentImage.data[idx + 1],
          b: currentImage.data[idx + 2],
        });
        toast({
          title: '颜色拾取',
          description: `RGB(${currentImage.data[idx]}, ${currentImage.data[idx + 1]}, ${currentImage.data[idx + 2]})`,
        });
      }
    }
  }, [currentImage, tool, zoom, brushSize, brushColor, originalImage, toast]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!currentImage || !overlayCanvasRef.current) return;
    
    const canvas = overlayCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / zoom);
    const y = Math.floor((e.clientY - rect.top) / zoom);
    
    if (tool === 'crop' && isCropping && cropStart) {
      const width = x - cropStart.x;
      const height = y - cropStart.y;
      setCropRect({
        x: width >= 0 ? cropStart.x : x,
        y: height >= 0 ? cropStart.y : y,
        width: Math.abs(width),
        height: Math.abs(height),
      });
    } else if ((tool === 'brush' || tool === 'eraser') && isDrawing && lastDrawPoint) {
      const ctx = canvasRef.current?.getContext('2d');
      if (ctx) {
        ctx.beginPath();
        ctx.moveTo(lastDrawPoint.x, lastDrawPoint.y);
        ctx.lineTo(x, y);
        ctx.lineWidth = brushSize;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        if (tool === 'eraser') {
          const pixelIdx = (y * currentImage.width + x) * 4;
          const r = originalImage?.data[pixelIdx] || 255;
          const g = originalImage?.data[pixelIdx + 1] || 255;
          const b = originalImage?.data[pixelIdx + 2] || 255;
          ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
        } else {
          ctx.strokeStyle = brushColor;
        }
        ctx.stroke();
        
        setLastDrawPoint({ x, y });
        
        // 更新当前图像数据
        const imageData = ctx.getImageData(0, 0, currentImage.width, currentImage.height);
        setCurrentImage({
          width: currentImage.width,
          height: currentImage.height,
          data: new Uint8ClampedArray(imageData.data),
        });
      }
    }
  }, [currentImage, tool, zoom, isCropping, cropStart, isDrawing, lastDrawPoint, brushSize, brushColor, originalImage]);

  const handleMouseUp = useCallback(() => {
    if ((tool === 'brush' || tool === 'eraser') && isDrawing && currentImage) {
      addHistory(tool === 'brush' ? '画笔绘制' : '橡皮擦擦除', currentImage);
    }
    setIsCropping(false);
    setIsDrawing(false);
    setLastDrawPoint(null);
  }, [tool, isDrawing, currentImage, addHistory]);

  // 确认裁剪
  const confirmCrop = useCallback(() => {
    if (!currentImage || !cropRect || cropRect.width === 0 || cropRect.height === 0) return;
    
    const result = crop(currentImage, cropRect);
    setCurrentImage(result);
    addHistory('裁剪', result);
    setCropRect(null);
    setTool('pointer');
    
    toast({
      title: '成功',
      description: '裁剪完成',
    });
  }, [currentImage, cropRect, addHistory, toast]);

  // 键盘快捷键
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 工具切换
      if (!e.ctrlKey && !e.metaKey) {
        switch (e.key.toLowerCase()) {
          case 'v': setTool('pointer'); break;
          case 'm': setTool('move'); break;
          case 'c': if (!e.shiftKey) setTool('crop'); break;
          case 'b': setTool('brush'); break;
          case 'e': setTool('eraser'); break;
          case 'i': setTool('picker'); break;
          case '+':
          case '=': setZoom(prev => Math.min(prev + 0.25, 5)); break;
          case '-': setZoom(prev => Math.max(prev - 0.25, 0.1)); break;
          case '0': setZoom(1); break;
          case 'delete':
            if (originalImage) {
              setCurrentImage(cloneImage(originalImage));
              addHistory('重置', originalImage);
            }
            break;
        }
      }
      
      // Ctrl组合键
      if (e.ctrlKey || e.metaKey) {
        switch (e.key.toLowerCase()) {
          case 'o':
            e.preventDefault();
            fileInputRef.current?.click();
            break;
          case 's':
            e.preventDefault();
            handleDownload();
            break;
          case 'z':
            e.preventDefault();
            undo();
            break;
          case 'y':
            e.preventDefault();
            redo();
            break;
          case 'r':
            e.preventDefault();
            if (e.shiftKey) {
              applyFilter('逆时针旋转90°', () => rotate(currentImage!, -90));
            } else {
              applyFilter('顺时针旋转90°', () => rotate(currentImage!, 90));
            }
            break;
          case 'h':
            e.preventDefault();
            if (e.shiftKey) {
              applyFilter('垂直翻转', () => flipVertical(currentImage!));
            } else {
              applyFilter('水平翻转', () => flipHorizontal(currentImage!));
            }
            break;
          case 'g':
            e.preventDefault();
            applyFilter('灰度化', () => toGrayscale(currentImage!));
            break;
          case 'i':
            e.preventDefault();
            applyFilter('反色', () => invert(currentImage!));
            break;
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentImage, originalImage, undo, redo, handleDownload, applyFilter, addHistory, toast]);

  // 重置参数
  const resetParameters = () => {
    setBrightness(0);
    setContrast(1);
    setGamma(1);
    setThreshold(128);
    setBlurSigma(1.5);
    setSharpenAmount(1);
    setRotationAngle(90);
    setScaleRatio(100);
    setNoiseDensity(0.05);
    setNoiseStdDev(25);
    setHueDegrees(0);
    setSaturationFactor(1);
    setOilRadius(3);
    setOilIntensity(20);
    setMosaicSize(10);
    setMorphKernelSize(3);
  };

  return (
    <TooltipProvider>
      <div className="min-h-screen flex flex-col bg-background">
        {/* 顶部工具栏 */}
        <header className="border-b bg-card px-4 py-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ImageIcon className="h-6 w-6 text-primary" />
            <h1 className="text-lg font-bold">数字图像处理工具</h1>
            <Badge variant="secondary" className="ml-2">Web版</Badge>
          </div>
          
          <div className="flex items-center gap-2">
            {/* 文件操作 */}
            <div className="flex items-center gap-1">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon" onClick={() => fileInputRef.current?.click()}>
                    <FolderOpen className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>打开图片 (Ctrl+O)</TooltipContent>
              </Tooltip>
              
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon" onClick={handleDownload} disabled={!currentImage}>
                    <Download className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>保存图片 (Ctrl+S)</TooltipContent>
              </Tooltip>
            </div>
            
            <Separator orientation="vertical" className="h-6" />
            
            {/* 撤销/重做 */}
            <div className="flex items-center gap-1">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon" onClick={undo} disabled={historyIndex <= 0}>
                    <Undo className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>撤销 (Ctrl+Z)</TooltipContent>
              </Tooltip>
              
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon" onClick={redo} disabled={historyIndex >= history.length - 1}>
                    <Redo className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>重做 (Ctrl+Y)</TooltipContent>
              </Tooltip>
            </div>
            
            <Separator orientation="vertical" className="h-6" />
            
            {/* 缩放 */}
            <div className="flex items-center gap-1">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon" onClick={() => setZoom(prev => Math.max(prev - 0.25, 0.1))}>
                    <ZoomOut className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>缩小 (-)</TooltipContent>
              </Tooltip>
              
              <span className="text-sm w-16 text-center">{Math.round(zoom * 100)}%</span>
              
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon" onClick={() => setZoom(prev => Math.min(prev + 0.25, 5))}>
                    <ZoomIn className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>放大 (+)</TooltipContent>
              </Tooltip>
              
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon" onClick={() => setZoom(1)}>
                    1:1
                  </Button>
                </TooltipTrigger>
                <TooltipContent>实际大小 (0)</TooltipContent>
              </Tooltip>
            </div>
            
            <Separator orientation="vertical" className="h-6" />
            
            {/* 显示选项 */}
            <div className="flex items-center gap-1">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={showGrid ? 'default' : 'outline'}
                    size="icon"
                    onClick={() => setShowGrid(!showGrid)}
                  >
                    <Grid3X3 className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>显示/隐藏网格</TooltipContent>
              </Tooltip>
              
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={showHistogram ? 'default' : 'outline'}
                    size="icon"
                    onClick={() => setShowHistogram(!showHistogram)}
                  >
                    <Layers className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>显示/隐藏直方图</TooltipContent>
              </Tooltip>
            </div>
            
            <Separator orientation="vertical" className="h-6" />
            
            {/* 快捷键帮助 */}
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" size="icon">
                  <Keyboard className="h-4 w-4" />
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-md">
                <DialogHeader>
                  <DialogTitle>键盘快捷键</DialogTitle>
                  <DialogDescription>
                    使用以下快捷键提高操作效率
                  </DialogDescription>
                </DialogHeader>
                <div className="grid grid-cols-2 gap-2 mt-4">
                  {SHORTCUTS.map((shortcut, index) => (
                    <div key={index} className="flex justify-between items-center p-2 bg-muted rounded">
                      <kbd className="bg-background px-2 py-1 rounded text-xs font-mono">{shortcut.key}</kbd>
                      <span className="text-sm">{shortcut.action}</span>
                    </div>
                  ))}
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </header>
        
        <div className="flex flex-1 overflow-hidden">
          {/* 左侧工具栏 */}
          <aside className="w-14 border-r bg-card flex flex-col items-center py-2 gap-1">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={tool === 'pointer' ? 'default' : 'ghost'}
                  size="icon"
                  onClick={() => setTool('pointer')}
                >
                  <MousePointer className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">选择工具 (V)</TooltipContent>
            </Tooltip>
            
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={tool === 'move' ? 'default' : 'ghost'}
                  size="icon"
                  onClick={() => setTool('move')}
                >
                  <Move className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">移动工具 (M)</TooltipContent>
            </Tooltip>
            
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={tool === 'crop' ? 'default' : 'ghost'}
                  size="icon"
                  onClick={() => setTool('crop')}
                >
                  <Crop className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">裁剪工具 (C)</TooltipContent>
            </Tooltip>
            
            <Separator className="my-2 w-8" />
            
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={tool === 'brush' ? 'default' : 'ghost'}
                  size="icon"
                  onClick={() => setTool('brush')}
                >
                  <Paintbrush className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">画笔工具 (B)</TooltipContent>
            </Tooltip>
            
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={tool === 'eraser' ? 'default' : 'ghost'}
                  size="icon"
                  onClick={() => setTool('eraser')}
                >
                  <Eraser className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">橡皮擦 (E)</TooltipContent>
            </Tooltip>
            
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={tool === 'picker' ? 'default' : 'ghost'}
                  size="icon"
                  onClick={() => setTool('picker')}
                >
                  <Circle className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">吸管工具 (I)</TooltipContent>
            </Tooltip>
            
            <Separator className="my-2 w-8" />
            
            {pickedColor && (
              <div
                className="w-8 h-8 rounded border-2 border-foreground"
                style={{ backgroundColor: `rgb(${pickedColor.r}, ${pickedColor.g}, ${pickedColor.b})` }}
                title={`RGB(${pickedColor.r}, ${pickedColor.g}, ${pickedColor.b})`}
              />
            )}
          </aside>
          
          {/* 主内容区 */}
          <main className="flex-1 flex overflow-hidden relative">
            {/* 加载遮罩 */}
            {isLoading && (
              <div className="absolute inset-0 bg-background/80 z-50 flex flex-col items-center justify-center gap-4">
                <Loader2 className="h-12 w-12 animate-spin text-primary" />
                <p className="text-lg font-medium">处理中...</p>
                <Progress value={loadingProgress} className="w-48" />
              </div>
            )}
            
            {/* 拖放提示遮罩 */}
            {isDragging && !currentImage && (
              <div className="absolute inset-0 bg-primary/10 z-40 flex items-center justify-center border-4 border-dashed border-primary rounded-lg m-4">
                <div className="bg-card p-8 rounded-lg shadow-lg">
                  <Upload className="h-16 w-16 mx-auto mb-4 text-primary" />
                  <p className="text-lg font-medium text-center">释放鼠标以打开图片</p>
                </div>
              </div>
            )}
            
            {/* 画布区域 */}
            <div
              ref={containerRef}
              className="flex-1 overflow-auto bg-muted/30 flex items-center justify-center p-4"
              style={{
                backgroundImage: 'linear-gradient(45deg, #ccc 25%, transparent 25%), linear-gradient(-45deg, #ccc 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #ccc 75%), linear-gradient(-45deg, transparent 75%, #ccc 75%)',
                backgroundSize: '20px 20px',
                backgroundPosition: '0 0, 0 10px, 10px -10px, -10px 0px',
              }}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {currentImage ? (
                <div className="relative" style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}>
                  <canvas
                    ref={canvasRef}
                    className="border shadow-lg"
                    style={{ cursor: tool === 'crop' ? 'crosshair' : tool === 'brush' || tool === 'eraser' ? 'crosshair' : 'default' }}
                  />
                  <canvas
                    ref={overlayCanvasRef}
                    className="absolute top-0 left-0 pointer-events-auto"
                    style={{ cursor: tool === 'crop' ? 'crosshair' : tool === 'brush' || tool === 'eraser' ? 'crosshair' : tool === 'picker' ? 'crosshair' : 'default' }}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                  />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center gap-6 text-muted-foreground">
                  <ImageIcon className="h-24 w-24 opacity-20" />
                  <p className="text-lg">拖放图片到此处或点击打开</p>
                  
                  <div className="flex gap-2">
                    <Button size="lg" onClick={() => fileInputRef.current?.click()}>
                      <FolderOpen className="mr-2 h-5 w-5" />
                      打开图片
                    </Button>
                  </div>
                  
                  <Separator className="my-4 w-64" />
                  
                  <p className="text-sm text-muted-foreground">或使用示例图片快速体验</p>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <Button
                      variant="outline"
                      className="h-20 w-28 flex flex-col gap-1"
                      onClick={() => loadFromUrl('https://picsum.photos/seed/nature/400/300', 'nature.jpg')}
                    >
                      <ImageIcon className="h-6 w-6" />
                      <span className="text-xs">风景图</span>
                    </Button>
                    <Button
                      variant="outline"
                      className="h-20 w-28 flex flex-col gap-1"
                      onClick={() => loadFromUrl('https://picsum.photos/seed/portrait/300/400', 'portrait.jpg')}
                    >
                      <ImageIcon className="h-6 w-6" />
                      <span className="text-xs">人像图</span>
                    </Button>
                    <Button
                      variant="outline"
                      className="h-20 w-28 flex flex-col gap-1"
                      onClick={() => loadFromUrl('https://picsum.photos/seed/abstract/400/400', 'abstract.jpg')}
                    >
                      <ImageIcon className="h-6 w-6" />
                      <span className="text-xs">抽象图</span>
                    </Button>
                    <Button
                      variant="outline"
                      className="h-20 w-28 flex flex-col gap-1"
                      onClick={() => loadFromUrl('https://picsum.photos/seed/tech/400/300', 'tech.jpg')}
                    >
                      <ImageIcon className="h-6 w-6" />
                      <span className="text-xs">科技图</span>
                    </Button>
                  </div>
                </div>
              )}
            </div>
            
            {/* 右侧面板 */}
            <aside className="w-80 border-l bg-card overflow-hidden flex flex-col">
              <Tabs defaultValue="filters" className="flex-1 flex flex-col">
                <TabsList className="grid w-full grid-cols-3 m-2">
                  <TabsTrigger value="filters">滤镜</TabsTrigger>
                  <TabsTrigger value="adjust">调整</TabsTrigger>
                  <TabsTrigger value="history">历史</TabsTrigger>
                </TabsList>
                
                <ScrollArea className="flex-1">
                  {/* 滤镜面板 */}
                  <TabsContent value="filters" className="p-4 space-y-4 m-0">
                    {/* 工具特定参数 */}
                    {(tool === 'brush' || tool === 'eraser') && (
                      <Card>
                        <CardHeader className="py-3">
                          <CardTitle className="text-sm">画笔设置</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          <div className="space-y-2">
                            <Label>大小: {brushSize}px</Label>
                            <Slider
                              value={[brushSize]}
                              onValueChange={([v]) => setBrushSize(v)}
                              min={1}
                              max={50}
                            />
                          </div>
                          {tool === 'brush' && (
                            <div className="space-y-2">
                              <Label>颜色</Label>
                              <input
                                type="color"
                                value={brushColor}
                                onChange={(e) => setBrushColor(e.target.value)}
                                className="w-full h-10 rounded cursor-pointer"
                              />
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    )}
                    
                    {tool === 'crop' && cropRect && cropRect.width > 0 && cropRect.height > 0 && (
                      <Card>
                        <CardHeader className="py-3">
                          <CardTitle className="text-sm">裁剪</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="text-sm text-muted-foreground mb-3">
                            选区: {cropRect.width} x {cropRect.height}
                          </p>
                          <Button className="w-full" onClick={confirmCrop}>
                            确认裁剪
                          </Button>
                        </CardContent>
                      </Card>
                    )}
                    
                    {/* 基本变换 */}
                    <Card>
                      <CardHeader className="py-3">
                        <CardTitle className="text-sm flex items-center gap-2">
                          <RefreshCw className="h-4 w-4" />
                          几何变换
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="grid grid-cols-2 gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('顺时针旋转90°', () => rotate(currentImage!, 90))}
                            disabled={!currentImage}
                          >
                            <RotateCw className="h-4 w-4 mr-1" />
                            顺时针
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('逆时针旋转90°', () => rotate(currentImage!, -90))}
                            disabled={!currentImage}
                          >
                            <RotateCcw className="h-4 w-4 mr-1" />
                            逆时针
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('水平翻转', () => flipHorizontal(currentImage!))}
                            disabled={!currentImage}
                          >
                            <FlipHorizontal className="h-4 w-4 mr-1" />
                            水平翻转
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('垂直翻转', () => flipVertical(currentImage!))}
                            disabled={!currentImage}
                          >
                            <FlipVertical className="h-4 w-4 mr-1" />
                            垂直翻转
                          </Button>
                        </div>
                        
                        <div className="space-y-2">
                          <Label>自定义旋转: {rotationAngle}°</Label>
                          <Slider
                            value={[rotationAngle]}
                            onValueChange={([v]) => setRotationAngle(v)}
                            min={0}
                            max={360}
                          />
                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full"
                            onClick={() => applyFilter(`旋转${rotationAngle}°`, () => rotate(currentImage!, rotationAngle))}
                            disabled={!currentImage}
                          >
                            应用旋转
                          </Button>
                        </div>
                        
                        <div className="space-y-2">
                          <Label>缩放: {scaleRatio}%</Label>
                          <Slider
                            value={[scaleRatio]}
                            onValueChange={([v]) => setScaleRatio(v)}
                            min={10}
                            max={200}
                          />
                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full"
                            onClick={() => applyFilter(`缩放${scaleRatio}%`, () => scale(currentImage!, scaleRatio / 100, scaleRatio / 100))}
                            disabled={!currentImage}
                          >
                            应用缩放
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                    
                    {/* 颜色空间 */}
                    <Card>
                      <CardHeader className="py-3">
                        <CardTitle className="text-sm flex items-center gap-2">
                          <Palette className="h-4 w-4" />
                          颜色空间
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="grid grid-cols-2 gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('灰度化', () => toGrayscale(currentImage!))}
                            disabled={!currentImage}
                          >
                            灰度化
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('反色', () => invert(currentImage!))}
                            disabled={!currentImage}
                          >
                            反色
                          </Button>
                        </div>
                        
                        <div className="space-y-2">
                          <Label>二值化阈值: {threshold}</Label>
                          <Slider
                            value={[threshold]}
                            onValueChange={([v]) => setThreshold(v)}
                            min={0}
                            max={255}
                          />
                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full"
                            onClick={() => applyFilter('二值化', () => toBinary(currentImage!, threshold))}
                            disabled={!currentImage}
                          >
                            二值化
                          </Button>
                        </div>
                        
                        <div className="space-y-2">
                          <Label>色调偏移: {hueDegrees}°</Label>
                          <Slider
                            value={[hueDegrees]}
                            onValueChange={([v]) => setHueDegrees(v)}
                            min={0}
                            max={360}
                          />
                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full"
                            onClick={() => applyFilter(`色调偏移${hueDegrees}°`, () => adjustHue(currentImage!, hueDegrees))}
                            disabled={!currentImage}
                          >
                            应用色调
                          </Button>
                        </div>
                        
                        <Button
                          variant="outline"
                          size="sm"
                          className="w-full"
                          onClick={() => applyFilter('直方图均衡化', () => histogramEqualization(currentImage!))}
                          disabled={!currentImage}
                        >
                          直方图均衡化
                        </Button>
                      </CardContent>
                    </Card>
                    
                    {/* 滤镜效果 */}
                    <Card>
                      <CardHeader className="py-3">
                        <CardTitle className="text-sm flex items-center gap-2">
                          <Filter className="h-4 w-4" />
                          滤镜效果
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="space-y-2">
                          <Label>高斯模糊 σ: {blurSigma}</Label>
                          <Slider
                            value={[blurSigma * 10]}
                            onValueChange={([v]) => setBlurSigma(v / 10)}
                            min={1}
                            max={50}
                          />
                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full"
                            onClick={() => applyFilter('高斯模糊', () => gaussianBlur(currentImage!, blurSigma))}
                            disabled={!currentImage}
                          >
                            高斯模糊
                          </Button>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('均值滤波', () => meanFilter(currentImage!))}
                            disabled={!currentImage}
                          >
                            均值滤波
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('中值滤波', () => medianFilter(currentImage!))}
                            disabled={!currentImage}
                          >
                            中值滤波
                          </Button>
                        </div>
                        
                        <div className="space-y-2">
                          <Label>锐化强度: {sharpenAmount}</Label>
                          <Slider
                            value={[sharpenAmount * 10]}
                            onValueChange={([v]) => setSharpenAmount(v / 10)}
                            min={1}
                            max={50}
                          />
                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full"
                            onClick={() => applyFilter('锐化', () => sharpen(currentImage!, sharpenAmount))}
                            disabled={!currentImage}
                          >
                            锐化
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                    
                    {/* 边缘检测 */}
                    <Card>
                      <CardHeader className="py-3">
                        <CardTitle className="text-sm flex items-center gap-2">
                          <Square className="h-4 w-4" />
                          边缘检测
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="grid grid-cols-2 gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('Sobel边缘检测', () => sobelEdgeDetection(currentImage!))}
                            disabled={!currentImage}
                          >
                            Sobel
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('Prewitt边缘检测', () => prewittEdgeDetection(currentImage!))}
                            disabled={!currentImage}
                          >
                            Prewitt
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('Laplacian边缘检测', () => laplacianEdgeDetection(currentImage!))}
                            disabled={!currentImage}
                          >
                            Laplacian
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('Canny边缘检测', () => cannyEdgeDetection(currentImage!))}
                            disabled={!currentImage}
                          >
                            Canny
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                    
                    {/* 形态学运算 */}
                    <Card>
                      <CardHeader className="py-3">
                        <CardTitle className="text-sm flex items-center gap-2">
                          <Square className="h-4 w-4" />
                          形态学运算
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="space-y-2">
                          <Label>核大小: {morphKernelSize}x{morphKernelSize}</Label>
                          <Slider
                            value={[morphKernelSize]}
                            onValueChange={([v]) => setMorphKernelSize(v)}
                            min={3}
                            max={15}
                            step={2}
                          />
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('腐蚀', () => erode(currentImage!, morphKernelSize))}
                            disabled={!currentImage}
                          >
                            腐蚀
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('膨胀', () => dilate(currentImage!, morphKernelSize))}
                            disabled={!currentImage}
                          >
                            膨胀
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('开运算', () => opening(currentImage!, morphKernelSize))}
                            disabled={!currentImage}
                          >
                            开运算
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('闭运算', () => closing(currentImage!, morphKernelSize))}
                            disabled={!currentImage}
                          >
                            闭运算
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                    
                    {/* 噪声与特效 */}
                    <Card>
                      <CardHeader className="py-3">
                        <CardTitle className="text-sm flex items-center gap-2">
                          <Sparkles className="h-4 w-4" />
                          噪声与特效
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="space-y-2">
                          <Label>椒盐噪声密度: {(noiseDensity * 100).toFixed(0)}%</Label>
                          <Slider
                            value={[noiseDensity * 100]}
                            onValueChange={([v]) => setNoiseDensity(v / 100)}
                            min={1}
                            max={50}
                          />
                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full"
                            onClick={() => applyFilter('添加椒盐噪声', () => addSaltPepperNoise(currentImage!, noiseDensity))}
                            disabled={!currentImage}
                          >
                            添加椒盐噪声
                          </Button>
                        </div>
                        
                        <div className="space-y-2">
                          <Label>高斯噪声标准差: {noiseStdDev}</Label>
                          <Slider
                            value={[noiseStdDev]}
                            onValueChange={([v]) => setNoiseStdDev(v)}
                            min={5}
                            max={100}
                          />
                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full"
                            onClick={() => applyFilter('添加高斯噪声', () => addGaussianNoise(currentImage!, 0, noiseStdDev))}
                            disabled={!currentImage}
                          >
                            添加高斯噪声
                          </Button>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('浮雕效果', () => emboss(currentImage!))}
                            disabled={!currentImage}
                          >
                            浮雕
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => applyFilter('边缘增强', () => edgeEnhance(currentImage!))}
                            disabled={!currentImage}
                          >
                            边缘增强
                          </Button>
                        </div>
                        
                        <div className="space-y-2">
                          <Label>油画半径: {oilRadius}</Label>
                          <Slider
                            value={[oilRadius]}
                            onValueChange={([v]) => setOilRadius(v)}
                            min={1}
                            max={10}
                          />
                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full"
                            onClick={() => applyFilter('油画效果', () => oilPainting(currentImage!, oilRadius, oilIntensity))}
                            disabled={!currentImage}
                          >
                            油画效果
                          </Button>
                        </div>
                        
                        <div className="space-y-2">
                          <Label>马赛克大小: {mosaicSize}</Label>
                          <Slider
                            value={[mosaicSize]}
                            onValueChange={([v]) => setMosaicSize(v)}
                            min={2}
                            max={30}
                          />
                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full"
                            onClick={() => applyFilter('马赛克效果', () => mosaic(currentImage!, mosaicSize))}
                            disabled={!currentImage}
                          >
                            马赛克
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>
                  
                  {/* 调整面板 */}
                  <TabsContent value="adjust" className="p-4 space-y-4 m-0">
                    <Card>
                      <CardHeader className="py-3">
                        <CardTitle className="text-sm flex items-center gap-2">
                          <SlidersHorizontal className="h-4 w-4" />
                          基本调整
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label>亮度</Label>
                            <span className="text-sm text-muted-foreground">{brightness}</span>
                          </div>
                          <Slider
                            value={[brightness + 100]}
                            onValueChange={([v]) => setBrightness(v - 100)}
                            min={0}
                            max={200}
                          />
                        </div>
                        
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label>对比度</Label>
                            <span className="text-sm text-muted-foreground">{contrast.toFixed(1)}</span>
                          </div>
                          <Slider
                            value={[contrast * 50]}
                            onValueChange={([v]) => setContrast(v / 50)}
                            min={0}
                            max={200}
                          />
                        </div>
                        
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label>伽马</Label>
                            <span className="text-sm text-muted-foreground">{gamma.toFixed(2)}</span>
                          </div>
                          <Slider
                            value={[gamma * 50]}
                            onValueChange={([v]) => setGamma(v / 50)}
                            min={10}
                            max={300}
                          />
                        </div>
                        
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label>饱和度</Label>
                            <span className="text-sm text-muted-foreground">{saturationFactor.toFixed(1)}</span>
                          </div>
                          <Slider
                            value={[saturationFactor * 50]}
                            onValueChange={([v]) => setSaturationFactor(v / 50)}
                            min={0}
                            max={200}
                          />
                        </div>
                        
                        <Button
                          className="w-full"
                          onClick={() => {
                            let result = cloneImage(currentImage!);
                            if (brightness !== 0) result = adjustBrightness(result, brightness);
                            if (contrast !== 1) result = adjustContrast(result, contrast);
                            if (gamma !== 1) result = gammaCorrection(result, gamma);
                            if (saturationFactor !== 1) result = adjustSaturation(result, saturationFactor);
                            setCurrentImage(result);
                            addHistory('图像调整', result);
                            toast({
                              title: '成功',
                              description: '已应用图像调整',
                            });
                          }}
                          disabled={!currentImage}
                        >
                          应用调整
                        </Button>
                        
                        <Button
                          variant="outline"
                          className="w-full"
                          onClick={resetParameters}
                        >
                          重置参数
                        </Button>
                      </CardContent>
                    </Card>
                    
                    {/* 直方图 */}
                    {showHistogram && histogramData && (
                      <Card>
                        <CardHeader className="py-3">
                          <CardTitle className="text-sm flex items-center gap-2">
                            <Layers className="h-4 w-4" />
                            直方图
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <HistogramChart data={histogramData} />
                        </CardContent>
                      </Card>
                    )}
                    
                    {/* 图像信息 */}
                    {currentImage && (
                      <Card>
                        <CardHeader className="py-3">
                          <CardTitle className="text-sm flex items-center gap-2">
                            <Info className="h-4 w-4" />
                            图像信息
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="text-sm space-y-1">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">尺寸:</span>
                            <span>{currentImage.width} x {currentImage.height}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">像素数:</span>
                            <span>{(currentImage.width * currentImage.height).toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">通道:</span>
                            <span>RGBA (4)</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">数据大小:</span>
                            <span>{(currentImage.data.length / 1024).toFixed(1)} KB</span>
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </TabsContent>
                  
                  {/* 历史面板 */}
                  <TabsContent value="history" className="p-4 m-0">
                    <Card>
                      <CardHeader className="py-3">
                        <CardTitle className="text-sm">操作历史</CardTitle>
                        <CardDescription>
                          点击恢复到指定步骤
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        {history.length === 0 ? (
                          <p className="text-sm text-muted-foreground text-center py-4">
                            暂无历史记录
                          </p>
                        ) : (
                          <div className="space-y-1 max-h-96 overflow-y-auto">
                            {history.map((item, index) => (
                              <div
                                key={item.id}
                                className={`flex items-center justify-between p-2 rounded cursor-pointer transition-colors ${
                                  index === historyIndex
                                    ? 'bg-primary text-primary-foreground'
                                    : 'hover:bg-muted'
                                }`}
                                onClick={() => {
                                  setHistoryIndex(index);
                                  setCurrentImage(item.imageData ? cloneImage(item.imageData) : null);
                                }}
                              >
                                <span className="text-sm truncate">{item.name}</span>
                                {index === historyIndex && (
                                  <Badge variant="secondary" className="ml-2">当前</Badge>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </TabsContent>
                </ScrollArea>
              </Tabs>
            </aside>
          </main>
        </div>
        
        {/* 底部状态栏 */}
        <footer className="border-t bg-card px-4 py-1 flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            {currentImage && (
              <>
                <span>{currentImage.width} x {currentImage.height}</span>
                <span>缩放: {Math.round(zoom * 100)}%</span>
              </>
            )}
            {!currentImage && <span>未加载图片</span>}
          </div>
          <div className="flex items-center gap-4">
            <span>工具: {tool === 'pointer' ? '选择' : tool === 'move' ? '移动' : tool === 'crop' ? '裁剪' : tool === 'brush' ? '画笔' : tool === 'eraser' ? '橡皮擦' : '吸管'}</span>
            <span>历史: {historyIndex + 1}/{history.length}</span>
          </div>
        </footer>
        
        {/* 隐藏的文件输入 */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileUpload}
          className="hidden"
        />
      </div>
    </TooltipProvider>
  );
}

// 直方图组件
function HistogramChart({ data }: { data: HistogramData }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // 清除画布
    ctx.clearRect(0, 0, width, height);
    
    // 找到最大值
    const maxVal = Math.max(
      ...data.red,
      ...data.green,
      ...data.blue,
      ...data.gray
    );
    
    // 绘制直方图
    const drawChannel = (values: number[], color: string) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.5;
      
      for (let i = 0; i < 256; i++) {
        const x = (i / 255) * width;
        const y = height - (values[i] / maxVal) * height;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      
      ctx.stroke();
      ctx.globalAlpha = 1;
    };
    
    drawChannel(data.gray, '#888888');
    drawChannel(data.red, '#ff0000');
    drawChannel(data.green, '#00ff00');
    drawChannel(data.blue, '#0000ff');
  }, [data]);
  
  return (
    <canvas
      ref={canvasRef}
      width={280}
      height={100}
      className="w-full border rounded"
    />
  );
}
