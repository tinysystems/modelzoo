"""
Tests for model architectures: FF, CNN, and FFF.

These tests verify that the models can be instantiated, perform forward passes,
and work correctly with typical input shapes (mimicking training scenarios).
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add src to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from models.basics import FF, CNN
from models.fff import FFF


# MNIST-like input parameters
BATCH_SIZE = 32
IN_CHANNELS = 1
IN_DIM = (28, 28)  # MNIST image dimensions
OUT_FEATURES = 10  # 10 classes for MNIST


class TestFFModel:
    """Tests for the Feedforward (FF) model."""

    def test_ff_instantiation(self):
        """Test that FF model can be instantiated."""
        model = FF(in_channels=IN_CHANNELS, in_dim=IN_DIM, width=64, out_features=OUT_FEATURES)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_ff_forward_pass(self):
        """Test that FF model performs forward pass correctly."""
        model = FF(in_channels=IN_CHANNELS, in_dim=IN_DIM, width=64, out_features=OUT_FEATURES)
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, *IN_DIM)
        output = model(x)
        assert output.shape == (BATCH_SIZE, OUT_FEATURES)

    def test_ff_training_step(self):
        """Test a complete training step with FF model (simulates python src/main.py model=ff epochs=1)."""
        model = FF(in_channels=IN_CHANNELS, in_dim=IN_DIM, width=64, out_features=OUT_FEATURES)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Simulate one training batch
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, *IN_DIM)
        targets = torch.randint(0, OUT_FEATURES, (BATCH_SIZE,))
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0


class TestCNNModel:
    """Tests for the CNN model."""

    def test_cnn_instantiation(self):
        """Test that CNN model can be instantiated."""
        model = CNN(in_channels=IN_CHANNELS, in_dim=IN_DIM, out_features=OUT_FEATURES)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_cnn_forward_pass(self):
        """Test that CNN model performs forward pass correctly."""
        model = CNN(in_channels=IN_CHANNELS, in_dim=IN_DIM, out_features=OUT_FEATURES)
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, *IN_DIM)
        output = model(x)
        assert output.shape == (BATCH_SIZE, OUT_FEATURES)

    def test_cnn_training_step(self):
        """Test a complete training step with CNN model (simulates python src/main.py model=cnn epochs=1)."""
        model = CNN(in_channels=IN_CHANNELS, in_dim=IN_DIM, out_features=OUT_FEATURES)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Simulate one training batch
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, *IN_DIM)
        targets = torch.randint(0, OUT_FEATURES, (BATCH_SIZE,))
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0


class TestFFFModel:
    """Tests for the Fast Feedforward (FFF) model."""

    def test_fff_instantiation(self):
        """Test that FFF model can be instantiated."""
        model = FFF(
            in_channels=IN_CHANNELS,
            in_dim=IN_DIM,
            leaf_width=16,
            out_features=OUT_FEATURES,
            depth=4
        )
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_fff_forward_pass_train(self):
        """Test that FFF model performs forward pass in training mode correctly."""
        model = FFF(
            in_channels=IN_CHANNELS,
            in_dim=IN_DIM,
            leaf_width=16,
            out_features=OUT_FEATURES,
            depth=4
        )
        model.train()
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, *IN_DIM)
        output = model(x)
        assert output.shape == (BATCH_SIZE, OUT_FEATURES)

    def test_fff_forward_pass_eval(self):
        """Test that FFF model performs forward pass in eval mode correctly."""
        model = FFF(
            in_channels=IN_CHANNELS,
            in_dim=IN_DIM,
            leaf_width=16,
            out_features=OUT_FEATURES,
            depth=4
        )
        model.eval()
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, *IN_DIM)
        output = model(x)
        assert output.shape == (BATCH_SIZE, OUT_FEATURES)

    def test_fff_training_step(self):
        """Test a complete training step with FFF model (simulates python src/main.py model=fff epochs=1)."""
        model = FFF(
            in_channels=IN_CHANNELS,
            in_dim=IN_DIM,
            leaf_width=16,
            out_features=OUT_FEATURES,
            depth=4
        )
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Simulate one training batch
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, *IN_DIM)
        targets = torch.randint(0, OUT_FEATURES, (BATCH_SIZE,))
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0
