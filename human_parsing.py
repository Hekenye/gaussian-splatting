import os
import tqdm

import cv2
from utils.segment_utils import MediapipeSegmenter

if __name__ == '__main__':
    set_write_bg = True
    # video_path = "data/xqp.mp4"
    out_path = "data/xqp/frames"
    seg_out_path = "data/xqp/images"
    os.makedirs(seg_out_path, exist_ok=True)
    # extract_images(video_path, out_path, fps=10)
    seg_model = MediapipeSegmenter()
    if set_write_bg:
        pad_color = 255
    else:
        pad_color = 0

    for image in tqdm(os.listdir(out_path)):
        img = cv2.imread(os.path.join(out_path, image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segmap = seg_model._cal_seg_map(img)   # [6, H, W]
        selected_mask = segmap[1:, :, :].sum(axis=0)[None,:] > 0.5  # [H, W] only remove 0, which means background
        img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = pad_color
        cv2.imwrite(os.path.join(seg_out_path, f"{image}"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
