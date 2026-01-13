"""
Modern PredNet implementation using PyTorch with best practices (2026)
Rewritten from Chainer-based code with improvements in:
- Error handling and logging
- Configuration management
- Memory efficiency
- Type hints and documentation
- Modern PyTorch idioms
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PredNetConfig:
    """Configuration management using dataclass pattern"""
    def __init__(self, **kwargs):
        self.image_size = kwargs.get('image_size', (160, 120))  # (width, height)
        self.channels = kwargs.get('channels', [3, 48, 96, 192])
        self.offset = kwargs.get('offset', (0, 0))
        self.input_len = kwargs.get('input_len', 50)
        self.ext_frames = kwargs.get('ext_frames', 10)
        self.bprop_len = kwargs.get('bprop_len', 20)
        self.save_period = kwargs.get('save_period', 10000)
        self.total_frames = kwargs.get('total_frames', 1000000)
        self.batch_size = kwargs.get('batch_size', 1)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = kwargs.get('num_workers', 4)
        
    def save(self, path: Path):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)
    
    @classmethod
    def load(cls, path: Path):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatial-temporal processing"""
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )
        
    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if state is None:
            h = torch.zeros(x.size(0), self.hidden_channels, x.size(2), x.size(3), 
                          device=x.device, dtype=x.dtype)
            c = torch.zeros_like(h)
        else:
            h, c = state
            
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, (h_next, c_next)


class PredNetLayer(nn.Module):
    """Single PredNet layer with prediction and error computation"""
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Representation LSTM (R)
        self.conv_lstm = ConvLSTMCell(input_channels * 2, hidden_channels)
        
        # Prediction
        self.pred_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, error: torch.Tensor, state: Optional[Tuple] = None):
        # Process error through ConvLSTM
        r, state = self.conv_lstm(error, state)
        
        # Generate prediction
        pred = self.pred_conv(r)
        
        return pred, r, state


class PredNet(nn.Module):
    """
    PredNet: Predictive coding network for video prediction
    Based on: Lotter et al. "Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning"
    """
    def __init__(self, width: int, height: int, channels: List[int]):
        super().__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.num_layers = len(channels) - 1
        
        # Create layers
        self.layers = nn.ModuleList([
            PredNetLayer(channels[i], channels[i+1]) 
            for i in range(self.num_layers)
        ])
        
        # Upsampling for predictions
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.reset_state()
        
    def reset_state(self):
        """Reset internal states for new sequence"""
        self.states = [None] * self.num_layers
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Args:
            x: Input tensor [batch, channels, height, width]
        Returns:
            prediction: Predicted next frame
            errors: Prediction errors for loss computation
        """
        errors = []
        representations = []
        
        # Bottom-up pass
        current_input = x
        for i, layer in enumerate(self.layers):
            # Compute error if we have a prediction from previous timestep
            if self.states[i] is not None:
                pred_prev = self.pred_conv if i == 0 else representations[i-1]
                error = torch.cat([
                    torch.relu(current_input - pred_prev),
                    torch.relu(pred_prev - current_input)
                ], dim=1)
            else:
                error = torch.cat([current_input, current_input], dim=1)
            
            # Forward through layer
            pred, r, state = layer(error, self.states[i])
            self.states[i] = state
            
            errors.append(error)
            representations.append(r)
            
            # Prepare input for next layer (downsample)
            if i < self.num_layers - 1:
                current_input = nn.functional.avg_pool2d(r, kernel_size=2)
        
        # Get final prediction from bottom layer
        final_prediction = pred
        
        # Compute total error for loss
        total_error = sum([e.abs().mean() for e in errors])
        
        return final_prediction, total_error


class VideoSequenceDataset(Dataset):
    """Dataset for loading video sequences"""
    def __init__(self, sequence_path: Path, root_dir: Path, config: PredNetConfig):
        self.root_dir = Path(root_dir)
        self.config = config
        self.sequences = self._load_sequences(sequence_path)
        logger.info(f"Loaded {len(self.sequences)} sequences")
        
    def _load_sequences(self, path: Path) -> List[List[Path]]:
        """Load sequence file lists"""
        sequences = []
        
        if not path.exists():
            raise FileNotFoundError(f"Sequence file not found: {path}")
        
        # Check if this is a single image list file or a sequence list file
        try:
            with open(path, 'r', encoding='utf-8', errors='strict') as f:
                first_line = f.readline().strip()
        except UnicodeDecodeError:
            raise ValueError(
                f"File appears to be binary, not a text file: {path}\n"
                f"Expected a text file containing either:\n"
                f"  1. List of image paths (one per line), OR\n"
                f"  2. List of sequence files (one per line)"
            )
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if not lines:
                raise ValueError(f"Sequence file is empty: {path}")
            
            # Check first file to determine format
            first_file = self.root_dir / lines[0].split()[0]
            
            # If first file is an image, treat entire file as single sequence
            if first_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                logger.info(f"Detected single image list file")
                images = []
                for line in lines:
                    img_path = self.root_dir / line.split()[0]
                    if not img_path.exists():
                        logger.warning(f"Image not found: {img_path}")
                        continue
                    images.append(img_path)
                
                if images:
                    sequences.append(images)
                    logger.info(f"Loaded 1 sequence with {len(images)} images")
            
            # Otherwise, treat as list of sequence files
            else:
                logger.info(f"Detected sequence list file")
                for line in lines:
                    seq_file = line.split()[0]
                    seq_path = self.root_dir / seq_file
                    
                    if not seq_path.exists():
                        logger.warning(f"Sequence file not found: {seq_path}")
                        continue
                    
                    # Load images in this sequence
                    images = []
                    try:
                        with open(seq_path, 'r', encoding='utf-8') as seq_f:
                            for img_line in seq_f:
                                img_line = img_line.strip()
                                if not img_line:
                                    continue
                                img_path = self.root_dir / img_line.split()[0]
                                if not img_path.exists():
                                    logger.warning(f"Image not found: {img_path}")
                                    continue
                                images.append(img_path)
                    except Exception as e:
                        logger.error(f"Error reading sequence file {seq_path}: {e}")
                        continue
                    
                    if images:
                        sequences.append(images)
                
                logger.info(f"Loaded {len(sequences)} sequences")
        
        except Exception as e:
            logger.error(f"Error loading sequences: {e}")
            raise
        
        if not sequences:
            raise ValueError(f"No valid sequences found in {path}")
        
        return sequences
    
    def _load_and_preprocess_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess a single image"""
        try:
            img = Image.open(path).convert('RGB')
            img_array = np.array(img).transpose(2, 0, 1)  # CHW format
            
            # Center crop
            _, h, w = img_array.shape
            top = self.config.offset[1] + (h - self.config.image_size[1]) // 2
            left = self.config.offset[0] + (w - self.config.image_size[0]) // 2
            bottom = top + self.config.image_size[1]
            right = left + self.config.image_size[0]
            
            img_array = img_array[:, top:bottom, left:right]
            
            # Normalize to [0, 1]
            img_tensor = torch.from_numpy(img_array).float() / 255.0
            
            return img_tensor
            
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            raise
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        """Get a sequence of frames"""
        sequence = self.sequences[idx]
        frames = [self._load_and_preprocess_image(img_path) for img_path in sequence]
        return frames


class Trainer:
    """Training manager with checkpointing and logging"""
    def __init__(self, model: PredNet, config: PredNetConfig, output_dir: Path):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Setup optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Create output directories
        self.model_dir = self.output_dir / 'models'
        self.result_dir = self.output_dir / 'results'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial model
        self.save_checkpoint(0, 'initial')
        
    def save_checkpoint(self, frame_count: int, suffix: str = ''):
        """Save model and optimizer state"""
        checkpoint_name = f'{suffix}.pt' if suffix else f'checkpoint_{frame_count}.pt'
        checkpoint_path = self.model_dir / checkpoint_name
        
        try:
            torch.save({
                'frame_count': frame_count,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config.__dict__
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model and optimizer state"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            frame_count = checkpoint['frame_count']
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return frame_count
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def save_image(self, tensor: torch.Tensor, path: Path):
        """Save tensor as image"""
        try:
            img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
            img_array = img_array.transpose(1, 2, 0)
            img = Image.fromarray(img_array)
            img.save(path)
        except Exception as e:
            logger.error(f"Error saving image to {path}: {e}")
    
    def train(self, dataset: VideoSequenceDataset):
        """Training loop"""
        self.model.train()
        frame_count = 0
        seq_idx = 0
        
        with open(self.output_dir / 'training_log.csv', 'w') as log_file:
            log_file.write('frame,sequence,loss\n')
            
            pbar = tqdm(total=self.config.total_frames, desc="Training")
            
            while frame_count < self.config.total_frames:
                # Get sequence
                frames = dataset[seq_idx]
                self.model.reset_state()
                
                accumulated_loss = 0
                frame_losses = []
                
                # Process sequence
                for i in range(len(frames) - 1):
                    x = frames[i].unsqueeze(0).to(self.device)
                    y = frames[i + 1].unsqueeze(0).to(self.device)
                    
                    # Forward pass
                    pred, error = self.model(x)
                    loss = self.criterion(pred, y) + 0.1 * error  # Combined loss
                    
                    accumulated_loss += loss
                    frame_losses.append(loss.item())
                    
                    # Backward pass every bprop_len frames
                    if (i + 1) % self.config.bprop_len == 0:
                        self.optimizer.zero_grad()
                        accumulated_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        
                        avg_loss = accumulated_loss.item() / self.config.bprop_len
                        log_file.write(f'{frame_count},{seq_idx},{avg_loss}\n')
                        log_file.flush()
                        
                        accumulated_loss = 0
                        pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
                    
                    # Save checkpoint
                    if frame_count % self.config.save_period == 0 and frame_count > 0:
                        self.save_checkpoint(frame_count)
                    
                    frame_count += 1
                    pbar.update(1)
                    
                    if frame_count >= self.config.total_frames:
                        break
                
                seq_idx = (seq_idx + 1) % len(dataset)
            
            pbar.close()
        
        logger.info("Training completed")
    
    @torch.no_grad()
    def test(self, dataset: VideoSequenceDataset):
        """Testing with extended prediction"""
        self.model.eval()
        
        with open(self.output_dir / 'test_log.csv', 'w') as log_file:
            log_file.write('sequence,frame,loss\n')
            
            for seq_idx in tqdm(range(len(dataset)), desc="Testing"):
                frames = dataset[seq_idx]
                self.model.reset_state()
                
                for i in range(len(frames) - 1):
                    x = frames[i].unsqueeze(0).to(self.device)
                    y = frames[i + 1].unsqueeze(0).to(self.device)
                    
                    pred, _ = self.model(x)
                    loss = self.criterion(pred, y)
                    
                    # Save results
                    self.save_image(x[0], self.result_dir / f'seq{seq_idx}_frame{i}_input.jpg')
                    self.save_image(pred[0], self.result_dir / f'seq{seq_idx}_frame{i}_pred.jpg')
                    
                    log_file.write(f'{seq_idx},{i},{loss.item()}\n')
                    
                    # Extended prediction
                    if i > 0 and self.config.input_len > 0 and i % self.config.input_len == 0:
                        current = pred
                        for ext_i in range(self.config.ext_frames):
                            current, _ = self.model(current)
                            self.save_image(
                                current[0],
                                self.result_dir / f'seq{seq_idx}_frame{i}_ext{ext_i+1}.jpg'
                            )
                        self.model.reset_state()
        
        logger.info("Testing completed")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Modern PredNet implementation in PyTorch',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
File Format Examples:
  
  1. Single image list (--images):
     path/to/image001.jpg
     path/to/image002.jpg
     path/to/image003.jpg
  
  2. Sequence list (--sequences):
     sequences/seq1.txt
     sequences/seq2.txt
     
     Where each sequence file contains:
     path/to/seq1/image001.jpg
     path/to/seq1/image002.jpg
     ...
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--images', '-i', type=str, help='Path to image list file (single sequence)')
    parser.add_argument('--sequences', '-seq', type=str, help='Path to sequence list file (multiple sequences)')
    parser.add_argument('--root', '-r', type=str, default='.', help='Root directory for resolving relative paths')
    parser.add_argument('--output', '-o', type=str, default='output', help='Output directory')
    parser.add_argument('--checkpoint', type=str, help='Load from checkpoint')
    parser.add_argument('--test', action='store_true', help='Test mode (no training)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], 
                       default='auto', help='Device to use')
    parser.add_argument('--validate-data', action='store_true', 
                       help='Validate data files exist before starting')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Load or create config
    if args.config:
        config = PredNetConfig.load(Path(args.config))
    else:
        device = args.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = PredNetConfig(device=device)
    
    # Save config
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / 'config.yaml')
    
    # Validate inputs
    if not args.images and not args.sequences:
        logger.error("Please specify --images or --sequences")
        sys.exit(1)
    
    # Create dataset
    sequence_path = Path(args.images) if args.images else Path(args.sequences)
    dataset = VideoSequenceDataset(sequence_path, Path(args.root), config)
    
    # Create model
    model = PredNet(config.image_size[0], config.image_size[1], config.channels)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = Trainer(model, config, output_dir)
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(Path(args.checkpoint))
    
    # Run training or testing
    if args.test:
        trainer.test(dataset)
    else:
        trainer.train(dataset)


if __name__ == '__main__':
    main()
