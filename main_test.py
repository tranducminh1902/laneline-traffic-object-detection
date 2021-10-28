import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio
import warnings

from tensorflow.python.keras.backend import sign
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
from sign_updater import *

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(cfg,opt):

    # dictionary to store traffic signs
    current_signs = {}

    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load lane-drivable detection model
    ld_model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    ld_model.load_state_dict(checkpoint['state_dict'])
    ld_model = ld_model.to(device)
    if half:
        ld_model.half()  # to FP16

    # Load object detection model
    od_weights = 'best _test.pt'
    od_model = torch.hub.load('ultralytics/yolov5', 'custom', path=od_weights)  # load FP32 object_detection_model
    od_model.conf = opt.conf_thres  # confidence threshold (0-1)
    od_model.iou = opt.iou_thres  # NMS IoU threshold (0-1)

    if half:
        od_model.half()  # to FP16

    # Load traffic sign classification model
    from traffic_sign_classify import ts_model

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size

    # Get names and colors from object detection model
    names = od_model.module.names if hasattr(od_model, 'module') else od_model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = ld_model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    ld_model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    n_frame = 0 # count frame
    guide_sign_timer = 0 

    for path, img, img_det, vid_cap,shapes in dataset:
        n_frame += 1
        ts_img = img.copy() # Copy for traffic sign classification

        od_img = img.copy() # Copy for object detection inference
        
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
       
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out= ld_model(img) #lane and drivable area inference

        pred = od_model(od_img, augment=opt.augment) #object and traffic sign inference
       
        t2 = time_synchronized()

        inf_time.update(t2-t1,img.size(0))

        # # Apply NMS
        t3 = time_synchronized()
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))


        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        ll_seg_mask = connect_lane(ll_seg_mask)

        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
        
        # Process detections
        for i, det in enumerate(pred.pred):  # detections per image
            s = '%g: ' % i

            gn = torch.tensor(img_det.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to img_det size
                # det[:, :4] = scale_coords(od_img.shape[2:], det[:, :4], img_det.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

        #         # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, img_det , label=label, color=colors[int(cls)], line_thickness=2)
                # print (int(cls))
                if int(cls) == 10 :
                    crop_ts = ts_img[int(xyxy[1]): int(xyxy[3]),int(xyxy[0]): int(xyxy[2])]
                    crop_ts = crop_ts[..., ::-1]
                    crop_ts = cv2.resize(crop_ts, (50, 50))
                    crop_ts = crop_ts/255.0
                    # print (crop_ts.shape)
                    crop_ts_array  = np.expand_dims(crop_ts, axis=0)
                    ts_pred = ts_model.predict(crop_ts_array)
                    sign_label = ts_pred[0].argmax()

                    current_signs = update_traffic_sign(sign_label, current_signs)
        
        img_det = plot_traffic_sign(img_det, current_signs)

            # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, t4 - t1))

        # if dataset.mode == 'images':
        #     cv2.imwrite(save_path,img_det)

        # elif dataset.mode == 'video':
        #     if vid_path != save_path:  # new video
        #         vid_path = save_path
        #         if isinstance(vid_writer, cv2.VideoWriter):
        #             vid_writer.release()  # release previous video writer

        #         fourcc = 'mp4v'  # output video codec
        #         fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #         h,w,_=img_det.shape
        #         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        #     vid_writer.write(img_det)
        
        if opt.source.isnumeric():
            img_det = cv2.resize(img_det, (1280, 960))
        else:
            img_det = cv2.resize(img_det, (1440, 810))
        cv2.imshow('image', img_det)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration  # 1 millisecond

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
