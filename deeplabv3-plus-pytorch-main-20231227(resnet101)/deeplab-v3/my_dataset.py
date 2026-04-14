import os
import torch.utils.data as data
from PIL import Image


class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # 修改点1: 强制将掩码转换为单通道（L模式）
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index]).convert('L')  # <--- 关键修改！

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # 修改点2: 确保target是二维张量 (H, W)
        # 如果转换后target是 (1, H, W)，需要去掉通道维度
        if target.dim() == 3:  # 检查是否是三维
            target = target.squeeze(0)  # 变为 (H, W) <--- 关键修改！

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 修改点3: 自动处理二维张量（例如target的(H, W)）
    tensors = []
    for img in images:
        if img.dim() == 2:  # 如果是二维 (H, W)
            img = img.unsqueeze(0)  # 变为 (1, H, W) <--- 关键修改！
        tensors.append(img)

    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
    batch_shape = (len(tensors),) + max_size
    batched_imgs = tensors[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs