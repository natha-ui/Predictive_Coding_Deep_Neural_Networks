#!/usr/bin/env python3
"""
Video to Frames Converter - Modern Python implementation (2026)

Extracts frames from video files and generates train/test split lists.
Rewritten with modern best practices, error handling, and type hints.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """Extract frames from video files with optional resizing and train/test split"""
    
    def __init__(
        self,
        input_video: Path,
        output_dir: Path,
        prefix: str = 'frame',
        width: int = -1,
        height: int = -1,
        test_ratio: float = 0.1,
        image_format: str = 'jpg',
        quality: int = 95
    ):
        """
        Args:
            input_video: Path to input video file
            output_dir: Directory to save extracted frames
            prefix: Prefix for output image filenames
            width: Target width (-1 to keep original)
            height: Target height (-1 to keep aspect ratio)
            test_ratio: Ratio of frames to use for test set (0.0 to 1.0)
            image_format: Output image format ('jpg' or 'png')
            quality: JPEG quality (1-100, only for jpg format)
        """
        self.input_video = Path(input_video)
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.width = width
        self.height = height
        self.test_ratio = max(0.0, min(1.0, test_ratio))
        self.image_format = image_format.lower()
        self.quality = quality
        
        # Validate inputs
        if not self.input_video.exists():
            raise FileNotFoundError(f"Input video not found: {self.input_video}")
        
        if self.image_format not in ['jpg', 'jpeg', 'png']:
            raise ValueError(f"Unsupported image format: {self.image_format}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def _resize_frame(self, frame: cv2.Mat) -> cv2.Mat:
        """Resize frame if dimensions are specified"""
        if self.width <= 0 and self.height <= 0:
            return frame
        
        orig_height, orig_width = frame.shape[:2]
        
        # Calculate target dimensions
        if self.width > 0:
            target_width = self.width
            if self.height <= 0:
                # Maintain aspect ratio
                target_height = int(orig_height * (self.width / orig_width))
            else:
                target_height = self.height
        else:
            # Only height specified
            target_height = self.height
            target_width = int(orig_width * (self.height / orig_height))
        
        # Use high-quality interpolation
        resized = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_LANCZOS4
        )
        
        return resized
    
    def _get_video_info(self, cap: cv2.VideoCapture) -> dict:
        """Extract video metadata"""
        return {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
    
    def extract_frames(self) -> List[Path]:
        """
        Extract all frames from video
        
        Returns:
            List of paths to saved frame images
        """
        logger.info(f"Opening video: {self.input_video}")
        
        cap = cv2.VideoCapture(str(self.input_video))
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.input_video}")
        
        # Get video info
        video_info = self._get_video_info(cap)
        logger.info(f"Video info: {video_info['width']}x{video_info['height']} "
                   f"@ {video_info['fps']:.2f} FPS, "
                   f"{video_info['frame_count']} frames")
        
        # Setup encoding parameters
        if self.image_format in ['jpg', 'jpeg']:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        else:
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        
        saved_files = []
        frame_idx = 0
        
        # Progress bar
        pbar = tqdm(
            total=video_info['frame_count'],
            desc="Extracting frames",
            unit="frame"
        )
        
        try:
            while True:
                success, frame = cap.read()
                
                if not success:
                    break
                
                # Resize if needed
                if self.width > 0 or self.height > 0:
                    frame = self._resize_frame(frame)
                
                # Generate filename
                filename = f"{self.prefix}_{frame_idx:07d}.{self.image_format}"
                filepath = self.output_dir / filename
                
                # Save frame
                success = cv2.imwrite(str(filepath), frame, encode_params)
                
                if not success:
                    logger.warning(f"Failed to save frame {frame_idx}")
                    continue
                
                saved_files.append(filepath)
                frame_idx += 1
                pbar.update(1)
        
        except Exception as e:
            logger.error(f"Error during frame extraction: {e}")
            raise
        
        finally:
            cap.release()
            pbar.close()
        
        logger.info(f"Extracted {len(saved_files)} frames")
        return saved_files
    
    def create_split_lists(
        self,
        file_paths: List[Path]
    ) -> Tuple[Path, Path]:
        """
        Create train/test split list files
        
        Args:
            file_paths: List of frame file paths
            
        Returns:
            Tuple of (train_list_path, test_list_path)
        """
        if not file_paths:
            raise ValueError("No files to split")
        
        # Calculate split index
        split_idx = int(len(file_paths) * (1.0 - self.test_ratio))
        
        train_files = file_paths[:split_idx]
        test_files = file_paths[split_idx:]
        
        # Write train list
        train_list_path = self.output_dir / "train_list.txt"
        with open(train_list_path, 'w') as f:
            for filepath in train_files:
                f.write(f"{filepath}\n")
        
        logger.info(f"Saved train list ({len(train_files)} files): {train_list_path}")
        
        # Write test list
        test_list_path = self.output_dir / "test_list.txt"
        with open(test_list_path, 'w') as f:
            for filepath in test_files:
                f.write(f"{filepath}\n")
        
        logger.info(f"Saved test list ({len(test_files)} files): {test_list_path}")
        
        return train_list_path, test_list_path
    
    def run(self) -> Tuple[List[Path], Path, Path]:
        """
        Run complete extraction and splitting pipeline
        
        Returns:
            Tuple of (frame_paths, train_list_path, test_list_path)
        """
        # Extract frames
        frame_paths = self.extract_frames()
        
        # Create split lists
        train_list, test_list = self.create_split_lists(frame_paths)
        
        logger.info("Processing complete!")
        
        return frame_paths, train_list, test_list


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Extract frames from video and create train/test splits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input_video',
        type=str,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '-p', '--prefix',
        type=str,
        default='frame',
        help='Prefix for output image filenames'
    )
    
    parser.add_argument(
        '-d', '--dir',
        type=str,
        default='data',
        help='Output directory for extracted frames'
    )
    
    parser.add_argument(
        '-r', '--ratio',
        type=float,
        default=0.1,
        help='Ratio of frames to use for test set (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '-w', '--width',
        type=int,
        default=-1,
        help='Target width for frames (-1 to keep original)'
    )
    
    parser.add_argument(
        '-g', '--height',
        type=int,
        default=-1,
        help='Target height for frames (-1 to maintain aspect ratio)'
    )
    
    parser.add_argument(
        '-f', '--format',
        type=str,
        default='jpg',
        choices=['jpg', 'jpeg', 'png'],
        help='Output image format'
    )
    
    parser.add_argument(
        '-q', '--quality',
        type=int,
        default=95,
        help='JPEG quality (1-100, only used for jpg format)'
    )
    
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=0,
        help='Extract every Nth frame (0 = extract all frames)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        # Create extractor
        extractor = VideoFrameExtractor(
            input_video=args.input_video,
            output_dir=args.dir,
            prefix=args.prefix,
            width=args.width,
            height=args.height,
            test_ratio=args.ratio,
            image_format=args.format,
            quality=args.quality
        )
        
        # Run extraction
        extractor.run()
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return 130
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
