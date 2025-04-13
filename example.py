from videocr import save_subtitles_to_file
import sys
import os
import argparse

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Extract subtitles from video using OCR')
    parser.add_argument('--video_path', 
                        default='/media/mike/8T_01/VideoLingo_joy/output/Strangest_Animal_Fact__Why_Do_Animals_Eat_Their_Babies__Filial_Cannibalism__Dr._Binocs_Show.mp4',
                        help='Path to the video file')
    parser.add_argument('--output', '-o', help='Output SRT file path (default: based on input filename)')
    parser.add_argument('--lang', default='en', help='Language for OCR (default: en)')
    parser.add_argument('--sim-threshold', type=int, default=80, help='Similarity threshold (default: 80)')
    parser.add_argument('--conf-threshold', type=int, default=50, help='Confidence threshold (default: 50)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--crop-x', type=int, default=0, help='X coordinate for cropping (default: 0)')
    parser.add_argument('--crop-y', type=int, default=900, help='Y coordinate for cropping (default: 900)')
    parser.add_argument('--crop-width', type=int, default=2000, help='Width for cropping (default: 2000)')
    parser.add_argument('--crop-height', type=int, default=300, help='Height for cropping (default: 300)')
    parser.add_argument('--brightness-threshold', type=int, default=210, help='Brightness threshold (default: 210)')
    parser.add_argument('--similar-image-threshold', type=int, default=0, help='Similar image threshold (default: 0)')
    parser.add_argument('--frames-to-skip', type=int, default=0, help='Frames to skip (default: 0)')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Set the video path
    video_path = args.video_path
    
    # Generate output file name based on input file if not specified
    if args.output:
        output_file = args.output
    else:
        video_basename = os.path.basename(video_path)
        video_name = os.path.splitext(video_basename)[0]
        output_file = f"{video_name}.srt"
    
    print(f"Processing video: {video_path}")
    print(f"Output file: {output_file}")
    
    # Use command-line argument values
    save_subtitles_to_file(
        video_path, 
        output_file, 
        lang=args.lang,
        sim_threshold=args.sim_threshold, 
        conf_threshold=args.conf_threshold, 
        use_fullframe=False, 
        use_gpu=not args.no_gpu,
        # Models different from the default mobile models can be downloaded here: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_en/models_list_en.md
        # det_model_dir='<PADDLEOCR DETECTION MODEL DIR>', rec_model_dir='<PADDLEOCR RECOGNITION MODEL DIR>',
        crop_x=args.crop_x, 
        crop_y=args.crop_y, 
        crop_width=args.crop_width, 
        crop_height=args.crop_height,
        brightness_threshold=args.brightness_threshold, 
        similar_image_threshold=args.similar_image_threshold, 
        frames_to_skip=args.frames_to_skip
    )

#  time_end='0:20',