import os
import copy
import numpy as np
import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt


def scatter_np(condition_img, classSeg=5):
# def scatter(condition_img, classSeg=19, label_size=(512, 512)):
    batch, c, height, width = condition_img.shape
    # if height != label_size[0] or width != label_size[1]:
        # condition_img= F.interpolate(condition_img, size=label_size, mode='nearest')
    input_label = np.zeros([batch, classSeg, condition_img.shape[2], condition_img.shape[3]]).astype(np.int_)
    # input_label = torch.zeros(batch, classSeg, *label_size, device=condition_img.device)
    np.put_along_axis(input_label, condition_img, 1, 1)
    return input_label

class MediapipeSegmenter:
    def __init__(self):
        model_path = 'data_gen/utils/mp_feature_extractors/selfie_multiclass_256x256.tflite'
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print("downloading segmenter model from mediapipe...")
            os.system(f"wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")
            os.system(f"mv selfie_multiclass_256x256.tflite {model_path}")
            print("download success")
        base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.ImageSegmenterOptions(base_options=base_options,running_mode=vision.RunningMode.IMAGE, output_category_mask=True)
        self.video_options = vision.ImageSegmenterOptions(base_options=base_options,running_mode=vision.RunningMode.VIDEO, output_category_mask=True)
        
    def _cal_seg_map_for_video(self, imgs, segmenter=None, return_onehot_mask=True, return_segmap_image=True):
        segmenter = vision.ImageSegmenter.create_from_options(self.video_options) if segmenter is None else segmenter
        assert return_onehot_mask or return_segmap_image # you should at least return one
        segmap_masks = []
        segmap_images = []
        for i in tqdm.trange(len(imgs), desc="extracting segmaps from a video..."):
            img = imgs[i]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            out = segmenter.segment_for_video(mp_image, 40 * i)
            segmap = out.category_mask.numpy_view().copy() # [H, W]

            if return_onehot_mask:
                segmap_mask = scatter_np(segmap[None, None, ...], classSeg=6)[0] # [6, H, W]
                segmap_masks.append(segmap_mask)
            if return_segmap_image:
                segmap_image = segmap[:, :, None].repeat(3, 2).astype(float)
                segmap_image = (segmap_image * 40).astype(np.uint8)
                segmap_images.append(segmap_image)
        
        if return_onehot_mask and return_segmap_image:
            return segmap_masks, segmap_images
        elif return_onehot_mask:
            return segmap_masks
        elif return_segmap_image:
            return segmap_images
    
    def _cal_seg_map(self, img, segmenter=None, return_onehot_mask=True):
        """
        segmenter: vision.ImageSegmenter.create_from_options(options)
        img: numpy, [H, W, 3], 0~255
        segmap: [C, H, W]
        0 - background
        1 - hair
        2 - body-skin
        3 - face-skin
        4 - clothes
        5 - others (accessories)
        """
        assert img.ndim == 3
        segmenter = vision.ImageSegmenter.create_from_options(self.options) if segmenter is None else segmenter 
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        out = segmenter.segment(image) 
        segmap = out.category_mask.numpy_view().copy() # [H, W]
        if return_onehot_mask:
            segmap = scatter_np(segmap[None, None, ...], classSeg=6)[0] # [6, H, W]
        return segmap

    def _seg_out_img_with_segmap(self, img, segmap, mode='head'):
        """
        img: [h,w,c], img is in 0~255, np
        """
        # 
        img = copy.deepcopy(img)
        if mode == 'head':
            selected_mask = segmap[[1,3,5] , :, :].sum(axis=0)[None,:] > 0.5 # glasses 也属于others
            img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 255 # (-1,-1,-1) denotes black in our [-1,1] convention
        elif mode == 'torso':
            selected_mask = segmap[[2,4], :, :].sum(axis=0)[None,:] > 0.5
            img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 255 # (-1,-1,-1) denotes black in our [-1,1] convention
        elif mode == 'torso_with_bg':
            selected_mask = segmap[[0, 2,4], :, :].sum(axis=0)[None,:] > 0.5
            img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 255 # (-1,-1,-1) denotes black in our [-1,1] convention
        elif mode == 'bg':
            selected_mask = segmap[[0], :, :].sum(axis=0)[None,:] > 0.5  # only seg out 0, which means background
            img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 255 # (-1,-1,-1) denotes black in our [-1,1] convention
        elif mode == 'body-skin':
            selected_mask = segmap[[2], :, :].sum(axis=0)[None,:] > 0.5
            img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 255
        elif mode == 'full':
            pass
        else:
            raise NotImplementedError()
        return img, selected_mask
    
    def _seg_out_img(self, img, segmenter=None, mode='head'):
        """
        imgs [H, W, 3] 0-255
        return : person_img [B, 3, H, W]
        """
        segmenter = vision.ImageSegmenter.create_from_options(self.options) if segmenter is None else segmenter 
        segmap = self._cal_seg_map(img, segmenter=segmenter, return_onehot_mask=True) # [B, 19, H, W]
        return self._seg_out_img_with_segmap(img, segmap, mode=mode)


if __name__ == '__main__':
    import cv2, time
    seg_model = MediapipeSegmenter()
    fig, axes = plt.subplots(3, 2)
    root = "/mnt/d/4DFace/0001_1_fg/1-00-01/001/images/"
    for i, img_name in enumerate(["A.png", "F.png", "K.png"]):
        img_name = os.path.join(root, img_name)
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("type:{}, shape:{}, dtype:{}".format(type(img), img.shape, img.dtype))
        print("max_value:{}, min_value:{}".format(img.max(), img.min()))
        segmap = seg_model._cal_seg_map(img)
        head_img = seg_model._seg_out_img_with_segmap(img, segmap, mode='head')[0]
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(head_img)

    plt.tight_layout()
    plt.show()
