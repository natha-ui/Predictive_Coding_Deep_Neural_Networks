"""
Modern PyTorch implementation of PredNet network architecture
Rewritten from Chainer with improvements in clarity and efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class ElementwiseFilter(nn.Module):
    """
    Learnable element-wise filter that applies per-element multiplication and addition.
    Replaces Chainer's custom EltFilter with PyTorch standard approach.
    """
    def __init__(
        self, 
        channels: int, 
        height: int, 
        width: int,
        use_bias: bool = True,
        weight_scale: float = 1.0
    ):
        super().__init__()
        
        # Initialize weights
        std = weight_scale * (1.0 / (width * height * channels)) ** 0.5
        self.weight = nn.Parameter(torch.randn(1, channels, height, width) * std)
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1, channels, height, width))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply element-wise multiplication and optional bias"""
        y = x * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell with peephole connections.
    Implements the full ConvLSTM architecture from the original PredNet paper.
    """
    def __init__(
        self,
        input_channels_list: List[int],
        hidden_channels: int,
        height: int,
        width: int,
        kernel_size: int = 3
    ):
        super().__init__()
        
        self.input_channels_list = input_channels_list
        self.hidden_channels = hidden_channels
        self.height = height
        self.width = width
        
        padding = kernel_size // 2
        
        # Input gate - for each input source
        self.x_i_convs = nn.ModuleList([
            nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=padding, bias=False)
            for in_ch in input_channels_list
        ])
        self.h_i = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.c_i = ElementwiseFilter(hidden_channels, height, width, use_bias=False)
        
        # Forget gate - for each input source
        self.x_f_convs = nn.ModuleList([
            nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=padding, bias=False)
            for in_ch in input_channels_list
        ])
        self.h_f = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.c_f = ElementwiseFilter(hidden_channels, height, width, use_bias=False)
        
        # Cell gate - for each input source
        self.x_c_convs = nn.ModuleList([
            nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=padding, bias=False)
            for in_ch in input_channels_list
        ])
        self.h_c = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        
        # Output gate - for each input source
        self.x_o_convs = nn.ModuleList([
            nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=padding, bias=False)
            for in_ch in input_channels_list
        ])
        self.h_o = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.c_o = ElementwiseFilter(hidden_channels, height, width, use_bias=False)
        
    def forward(
        self, 
        inputs: List[torch.Tensor],
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through ConvLSTM cell
        
        Args:
            inputs: List of input tensors (one per source)
            state: Optional tuple of (h, c) hidden state tensors
            
        Returns:
            output: Hidden state
            state: Tuple of (h, c) for next timestep
        """
        batch_size = inputs[0].size(0)
        
        # Initialize state if None
        if state is None:
            device = inputs[0].device
            dtype = inputs[0].dtype
            h = torch.zeros(batch_size, self.hidden_channels, self.height, self.width,
                          device=device, dtype=dtype)
            c = torch.zeros_like(h)
        else:
            h, c = state
        
        # Input gate
        i = sum(conv(x) for conv, x in zip(self.x_i_convs, inputs))
        i = i + self.h_i(h) + self.c_i(c)
        i = torch.sigmoid(i)
        
        # Forget gate
        f = sum(conv(x) for conv, x in zip(self.x_f_convs, inputs))
        f = f + self.h_f(h) + self.c_f(c)
        f = torch.sigmoid(f)
        
        # Cell gate
        c_tilde = sum(conv(x) for conv, x in zip(self.x_c_convs, inputs))
        c_tilde = c_tilde + self.h_c(h)
        c_tilde = torch.tanh(c_tilde)
        
        # Update cell state
        c_next = f * c + i * c_tilde
        
        # Output gate
        o = sum(conv(x) for conv, x in zip(self.x_o_convs, inputs))
        o = o + self.h_o(h) + self.c_o(c_next)
        o = torch.sigmoid(o)
        
        # Compute hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, (h_next, c_next)


class PredNetLayer(nn.Module):
    """
    Single layer in the PredNet hierarchy.
    Processes errors and generates predictions at one spatial scale.
    """
    def __init__(
        self,
        channels: int,
        r_channels: int,
        height: int,
        width: int,
        has_input_from_above: bool = True
    ):
        super().__init__()
        
        self.channels = channels
        self.r_channels = r_channels
        self.has_input_from_above = has_input_from_above
        
        # Determine ConvLSTM input sources
        # Always receives error (channels * 2 for pos/neg errors)
        input_channels_list = [channels * 2]
        if has_input_from_above:
            # Also receives representation from layer above
            input_channels_list.append(r_channels)
        
        # ConvLSTM for representation
        self.conv_lstm = ConvLSTMCell(
            input_channels_list=input_channels_list,
            hidden_channels=r_channels,
            height=height,
            width=width
        )
        
        # Prediction generation from representation
        self.conv_p = nn.Conv2d(r_channels, channels, kernel_size=3, padding=1)
        
        # State storage
        self.lstm_state = None
        self.prediction = None
        
    def reset_state(self):
        """Reset internal LSTM state and prediction"""
        self.lstm_state = None
        self.prediction = None
        
    def forward(
        self,
        error: torch.Tensor,
        r_above: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through layer
        
        Args:
            error: Error signal (2 * channels from pos/neg errors)
            r_above: Optional representation from layer above (after upsampling)
            
        Returns:
            prediction: Predicted input for this layer
            representation: Hidden representation to pass to layer below
        """
        # Prepare inputs for ConvLSTM
        lstm_inputs = [error]
        if r_above is not None:
            lstm_inputs.append(r_above)
        
        # Update representation via ConvLSTM
        representation, self.lstm_state = self.conv_lstm(lstm_inputs, self.lstm_state)
        
        # Generate prediction from representation
        prediction = self.conv_p(representation)
        
        return prediction, representation


class PredNet(nn.Module):
    """
    Predictive Coding Network for video prediction.
    
    Implements the architecture from:
    Lotter, W., Kreiman, G., & Cox, D. (2016). Deep predictive coding networks 
    for video prediction and unsupervised learning. ICLR 2017.
    
    The network uses a hierarchical architecture where each layer:
    1. Computes prediction errors (difference between prediction and input)
    2. Updates internal representation via ConvLSTM
    3. Generates prediction for the layer below
    """
    def __init__(
        self,
        width: int,
        height: int,
        channels: List[int],
        r_channels: Optional[List[int]] = None
    ):
        """
        Args:
            width: Input image width
            height: Input image height
            channels: Number of channels at each layer (bottom to top)
            r_channels: Number of representation channels at each layer
        """
        super().__init__()
        
        if r_channels is None:
            r_channels = channels.copy()
        
        self.num_layers = len(channels)
        self.channels = channels
        self.r_channels = r_channels
        
        # Calculate spatial dimensions at each layer
        self.layer_sizes = []
        w, h = width, height
        for i in range(self.num_layers):
            self.layer_sizes.append((channels[i], h, w))
            w = w // 2
            h = h // 2
        
        # Create layers
        self.layers = nn.ModuleList()
        self.conv_a_list = nn.ModuleList()  # For processing errors before pooling
        
        for i in range(self.num_layers):
            ch, h, w = self.layer_sizes[i]
            
            # Create representation layer
            is_top = (i == self.num_layers - 1)
            layer = PredNetLayer(
                channels=ch,
                r_channels=r_channels[i],
                height=h,
                width=w,
                has_input_from_above=not is_top
            )
            self.layers.append(layer)
            
            # Conv for processing errors (except at layer 0)
            if i > 0:
                prev_ch = channels[i - 1]
                conv_a = nn.Conv2d(prev_ch * 2, ch, kernel_size=3, padding=1)
                self.conv_a_list.append(conv_a)
            else:
                self.conv_a_list.append(None)
    
    def reset_state(self):
        """Reset state of all layers"""
        for layer in self.layers:
            layer.reset_state()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PredNet
        
        Args:
            x: Input image tensor [batch, channels, height, width]
            
        Returns:
            prediction: Predicted next frame [batch, channels, height, width]
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype
        
        # Initialize predictions if None
        for i, layer in enumerate(self.layers):
            if layer.prediction is None:
                ch, h, w = self.layer_sizes[i]
                layer.prediction = torch.zeros(batch_size, ch, h, w, 
                                              device=device, dtype=dtype)
        
        # Bottom-up pass: Compute errors at each layer
        errors = []
        for i in range(self.num_layers):
            if i == 0:
                # Bottom layer: compare input with prediction
                error_pos = F.relu(x - self.layers[i].prediction)
                error_neg = F.relu(self.layers[i].prediction - x)
                error = torch.cat([error_pos, error_neg], dim=1)
            else:
                # Higher layers: process error from below, pool, then compare
                prev_error = errors[i - 1]
                a = F.relu(self.conv_a_list[i](prev_error))
                a_pooled = F.max_pool2d(a, kernel_size=2, stride=2)
                
                error_pos = F.relu(a_pooled - self.layers[i].prediction)
                error_neg = F.relu(self.layers[i].prediction - a_pooled)
                error = torch.cat([error_pos, error_neg], dim=1)
            
            errors.append(error)
        
        # Top-down pass: Update representations and predictions
        representations = [None] * self.num_layers
        
        for i in reversed(range(self.num_layers)):
            # Get representation from above (if not top layer)
            if i < self.num_layers - 1:
                r_above = representations[i + 1]
                # Upsample to match current layer size
                r_above = F.interpolate(
                    r_above, 
                    scale_factor=2, 
                    mode='nearest'
                )
            else:
                r_above = None
            
            # Update layer
            pred, rep = self.layers[i].forward(errors[i], r_above)
            
            # Store for next iteration
            representations[i] = rep
            
            # Apply activation to prediction
            if i == 0:
                # Bottom layer: clamp to [0, 1]
                self.layers[i].prediction = torch.clamp(pred, 0.0, 1.0)
            else:
                # Higher layers: ReLU activation
                self.layers[i].prediction = F.relu(pred)
        
        # Return bottom layer prediction
        return self.layers[0].prediction


def test_prednet():
    """Test function to verify network architecture"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PredNet architecture')
    parser.add_argument('--size', '-s', default='160,120',
                       help='Size of target images. width,height (pixels)')
    parser.add_argument('--channels', '-c', default='3,48,96,192',
                       help='Number of channels on each layer')
    parser.add_argument('--batch', '-b', default=2, type=int,
                       help='Batch size for testing')
    args = parser.parse_args()
    
    # Parse arguments
    size = [int(x) for x in args.size.split(',')]
    channels = [int(x) for x in args.channels.split(',')]
    
    width, height = size[0], size[1]
    
    # Create model
    print(f"Creating PredNet model:")
    print(f"  Image size: {width}x{height}")
    print(f"  Channels: {channels}")
    print(f"  Batch size: {args.batch}")
    
    model = PredNet(width, height, channels)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    x = torch.randn(args.batch, channels[0], height, width)
    
    try:
        # First forward pass
        output1 = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output1.shape}")
        print(f"  Output range: [{output1.min().item():.4f}, {output1.max().item():.4f}]")
        
        # Second forward pass (testing state persistence)
        output2 = model(x)
        print(f"\n  Second pass output shape: {output2.shape}")
        print(f"  Second pass output range: [{output2.min().item():.4f}, {output2.max().item():.4f}]")
        
        # Test state reset
        model.reset_state()
        output3 = model(x)
        print(f"\n  After reset output shape: {output3.shape}")
        print(f"  After reset output range: [{output3.min().item():.4f}, {output3.max().item():.4f}]")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error during forward pass: {e}")
        raise


if __name__ == "__main__":
    test_prednet()
