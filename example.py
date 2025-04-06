from videocr import save_subtitles_to_file

if __name__ == '__main__':
    save_subtitles_to_file('/media/mike/8T_01/VideoLingo_joy/output/Strangest_Animal_Fact__Why_Do_Animals_Eat_Their_Babies__Filial_Cannibalism__Dr._Binocs_Show.mp4', 'example.srt', lang='en',
     sim_threshold=80, conf_threshold=50, use_fullframe=False, use_gpu=True,
    # Models different from the default mobile models can be downloaded here: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_en/models_list_en.md
    # det_model_dir='<PADDLEOCR DETECTION MODEL DIR>', rec_model_dir='<PADDLEOCR RECOGNITION MODEL DIR>',
     crop_x=0, crop_y=900, crop_width=2000, crop_height=300,
     brightness_threshold=210, similar_image_threshold=0, frames_to_skip=0)

#  time_end='0:20',