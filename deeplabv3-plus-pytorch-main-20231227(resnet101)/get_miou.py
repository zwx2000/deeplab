#这段代码是一个Python脚本，用于评估一个基于DeepLabV3模型的图像语义分割任务的性能。该脚本的主要功能包括：

#设置运行模式：根据miou_mode参数，脚本可以选择仅获取预测结果、仅计算miou（Mean Intersection over Union）指标，或者执行整个miou计算流程。
#初始化参数：设置分类个数、类别名称、VOC数据集路径等参数。
#加载模型：如果miou_mode为0或1，加载DeepLabV3模型。
#获取预测结果：遍历验证集的图像，使用DeepLabV3模型对每张图像进行语义分割，并将分割结果保存为PNG文件。
#计算miou：如果miou_mode为0或2，计算验证集的miou指标。
#展示结果：如果需要，展示miou指标的计算结果。
#需要注意的是，miou计算是基于预测结果和真实标签进行比较的，因此需要确保预测结果和真实标签的格式和路径正确无误。此外，由于miou计算可能会生成大量的中间文件，建议在执行计算前创建一个临时目录来存储这些文件。

#代码中使用了tqdm库来显示进度条，使迭代过程更加直观。compute_mIoU函数用于计算miou指标，show_results函数用于展示miou指标的计算结果。

#这段代码是一个完整的miou评估脚本，可以用于评估基于DeepLabV3模型的图像语义分割任务的性能。
import os

from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 21       #类别数
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["_background_","cat","dog"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'    #数据集的路径

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines()    #验证集的图像ID列表，从val.txt文件中读取。
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")          #真实标签的文件夹路径。
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        deeplab = DeeplabV3()                                               #从这里
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))            #到这里是：这部分代码首先加载DeeplabV3模型，然后遍历所有图像ID。对于每个图像ID，代码读取相应的图像，使用DeeplabV3模型生成预测结果，并将结果保存为PNG文件。
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        #这行代码调用compute_mIoU函数，计算IoU、精确度、召回率和总体准确率。这些指标是评估图像分割模型性能的重要指标。
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)