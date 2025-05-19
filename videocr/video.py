from __future__ import annotations
from typing import List
import cv2
import numpy as np
import json
import tqdm
import logging
import os

from . import utils
from .models import PredictedFrames, PredictedSubtitle
from .opencv_adapter import Capture
from paddleocr import PaddleOCR


class Video:
    path: str
    lang: str
    use_fullframe: bool
    det_model_dir: str
    rec_model_dir: str
    num_frames: int
    fps: float
    height: int
    ocr: PaddleOCR
    pred_frames: List[PredictedFrames]
    pred_subs: List[PredictedSubtitle]

    def __init__(self, path: str, det_model_dir: str, rec_model_dir: str):
        self.path = path
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        with Capture(path) as v:
            self.num_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            print('self.fps', v.get(cv2.CAP_PROP_FPS))
            # self.fps = 23.783608253002267
            # self.fps = 23.976

            self.fps = 24
            self.height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def run_ocr(self, use_gpu: bool, lang: str, time_start: str, time_end: str,
                conf_threshold: int, use_fullframe: bool, brightness_threshold: int, similar_image_threshold: int, similar_pixel_threshold: int, frames_to_skip: int,
                crop_x: int, crop_y: int, crop_width: int, crop_height: int) -> None:
        conf_threshold_percent = float(conf_threshold/100)
        self.lang = lang
        self.use_fullframe = use_fullframe
        self.pred_frames = []
        # Configure PaddleOCR to show only errors, no debug or warning messages
        ocr = PaddleOCR(lang=self.lang, rec_model_dir=self.rec_model_dir, det_model_dir=self.det_model_dir, 
                       use_gpu=use_gpu)

        # 创建一个字典来存储每帧的识别结果
        frames_data = {}

        ocr_start = utils.get_frame_index(time_start, self.fps) if time_start else 0
        ocr_end = utils.get_frame_index(time_end, self.fps) if time_end else self.num_frames

        if ocr_end < ocr_start:
            raise ValueError('time_start is later than time_end')
        num_ocr_frames = ocr_end - ocr_start

        crop_x_end = None
        crop_y_end = None
        if crop_x and crop_y and crop_width and crop_height:
            crop_x_end = crop_x + crop_width
            crop_y_end = crop_y + crop_height

        # get frames from ocr_start to ocr_end
        with Capture(self.path) as v:
            v.set(cv2.CAP_PROP_POS_FRAMES, ocr_start)
            prev_grey = None
            predicted_frames = None
            modulo = frames_to_skip + 1
            
            # Calculate how many frames will actually be processed
            actual_frames = (num_ocr_frames + modulo - 1) // modulo
            print(f"Starting OCR processing on {actual_frames} frames...")
            
            # Create progress bar for OCR processing
            pbar = tqdm.tqdm(total=actual_frames, desc="Recognizing text in frames", unit="frame")
            
            frames_processed = 0
            for i in range(num_ocr_frames):
                if i % modulo == 0:
                    frame = v.read()[1]
                    # 保存原始帧用于后续处理
                    original_frame = frame.copy()
                    
                    if not self.use_fullframe:
                        if crop_x_end and crop_y_end:
                            frame = frame[crop_y:crop_y_end, crop_x:crop_x_end]
                            
                            # # 创建保存裁剪帧的文件夹
                            # cropped_frames_dir = 'cropped_frames'
                            # os.makedirs(cropped_frames_dir, exist_ok=True)
                            
                            # # 保存裁剪后的帧，以frame_index命名
                            # frame_index = i + ocr_start
                            # frame_filename = os.path.join(cropped_frames_dir, f'frame_{frame_index}.jpg')
                            # cv2.imwrite(frame_filename, frame)
                        else:
                            # only use bottom third of the frame by default
                            
                            # 创建保存裁剪帧的文件夹
                            # cropped_frames_dir = 'cropped_frames'
                            # os.makedirs(cropped_frames_dir, exist_ok=True)
                            
                            # # 保存裁剪后的帧，以frame_index命名
                            # frame_index = i + ocr_start
                            # frame_filename = os.path.join(cropped_frames_dir, f'frame_{frame_index}.jpg')
                            # cv2.imwrite(frame_filename, frame)
                            frame = frame[self.height // 3:, :]

                    if brightness_threshold:
                        frame = cv2.bitwise_and(frame, frame, mask=cv2.inRange(frame, (brightness_threshold, brightness_threshold, brightness_threshold), (255, 255, 255)))

                    if similar_image_threshold:
                        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if prev_grey is not None:
                            _, absdiff = cv2.threshold(cv2.absdiff(prev_grey, grey), similar_pixel_threshold, 255, cv2.THRESH_BINARY)
                            if np.count_nonzero(absdiff) < similar_image_threshold:
                                predicted_frames.end_index = i + ocr_start
                                prev_grey = grey
                                continue

                        prev_grey = grey

                    # Update progress bar before OCR processing of each frame
                    frames_processed += 1
                    pbar.update(1)
                    
                    # This is where OCR actually happens
                    ocr_result = ocr.ocr(frame)
                    predicted_frames = PredictedFrames(i + ocr_start, ocr_result, conf_threshold_percent)
                    self.pred_frames.append(predicted_frames)
                    
                    # 保存每帧的识别结果，包含更多详细信息
                    frame_index = i + ocr_start
                    
                    # 使用实际时间而不是帧索引计算时间戳，避免累积误差
                    # 通过直接计算帧在视频中的实际时间点来获得更准确的时间戳
                    actual_seconds = frame_index / self.fps
                    h = int(actual_seconds // 3600)
                    m = int((actual_seconds % 3600) // 60)
                    s = int(actual_seconds % 60)
                    ms = int((actual_seconds - int(actual_seconds)) * 1000)
                    frame_time = '{:02d}:{:02d}:{:02d},{:03d}'.format(h, m, s, ms)
                    
                    frames_data[frame_index] = {
                        "frame_index": frame_index,
                        "time": frame_time,
                        "text": predicted_frames.text,
                        "confidence": predicted_frames.confidence
                    }
                else:
                    v.read()
            
            pbar.close()  # Close progress bar when done
            print(f"OCR completed on {frames_processed} frames")
        
        # 将所有帧的识别结果保存到文件
        with open('frame_ocr_results.json', 'w', encoding='utf-8') as f:
            json.dump(frames_data, f, ensure_ascii=False, indent=2)

    def get_subtitles(self, sim_threshold: int) -> str:
        self._generate_subtitles(sim_threshold)
        
        return ''.join(
            '{}\n{} --> {}\n{}\n\n'.format(
                i,
                utils.get_srt_timestamp(sub.index_start, self.fps),
                utils.get_srt_timestamp(sub.index_end, self.fps),
                sub.text)
            for i, sub in enumerate(self.pred_subs)
            if len(sub.text) >= 2)

    def _generate_subtitles(self, sim_threshold: int) -> None:
        self.pred_subs = []

        if self.pred_frames is None:
            raise AttributeError(
                'Please call self.run_ocr() first to perform ocr on frames')

        # Add progress bar for subtitle generation
        print("Generating subtitles...")
        max_frame_merge_diff = int(0.09 * self.fps)
        for frame in tqdm.tqdm(self.pred_frames, desc="Generating subtitles", unit="frame"):
            self._append_sub(PredictedSubtitle([frame], sim_threshold), max_frame_merge_diff)
        self.pred_subs = [sub for sub in self.pred_subs if len(sub.frames[0].lines) > 0]
        print(f"Generated {len(self.pred_subs)} subtitle entries")

    def _append_sub(self, sub: PredictedSubtitle, max_frame_merge_diff: int) -> None:
        if len(sub.frames) == 0:
            return

        # merge new sub to the last subs if they are not empty, similar and within 0.09 seconds apart
        if self.pred_subs:
            last_sub = self.pred_subs[-1]
            if len(last_sub.frames[0].lines) > 0 and sub.index_start - last_sub.index_end <= max_frame_merge_diff and last_sub.is_similar_to(sub):
                del self.pred_subs[-1]
                sub = PredictedSubtitle(last_sub.frames + sub.frames, sub.sim_threshold)

        self.pred_subs.append(sub)
