"""
Enhanced Grad-CAM Viewer V05 - Bug Fixes + Progress Dialog

Features:
- Works with or without CSV file
- Load single images for classification and Grad-CAM analysis
- Browse through quality analysis results table (when CSV loaded)
- All visualization controls (opacity, colormap, threshold, split view)
- Display image metadata under the original image (extracted from CSV)
- Automatic EVA model loading (no need for model.py file)
- Error dialogs for missing columns with detailed messages
- Only 'image_path' column is required - all others are optional
- Fetch ISIC Archive metadata via REST API button
"""

import sys
import json
import traceback
import urllib.request
import urllib.error
import torch
import torch.nn.functional as F
from torchvision import transforms
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QSplitter, QFileDialog, QComboBox,
                             QMessageBox, QDialog, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

from PIL import Image

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Worker threads for long-running operations
# ---------------------------------------------------------------------------

class ModelLoadWorker(QThread):
    """Background thread for loading model + checkpoint"""
    finished = pyqtSignal(object)   # result dict or exception
    progress = pyqtSignal(str)      # status text updates

    def __init__(self, checkpoint_path, device, get_model_class_fn):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.get_model_class_fn = get_model_class_fn
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            self.progress.emit("Creating EVA-02 model architecture...")
            if self._stopped:
                self.finished.emit(None)
                return

            ModelClass = self.get_model_class_fn()
            model = ModelClass().to(self.device)
            model.eval()

            if self._stopped:
                self.finished.emit(None)
                return

            self.progress.emit("Loading checkpoint weights...")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Handle key prefix mismatch
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())

            ckpt_has_prefix = all(k.startswith('model.') for k in checkpoint_keys)
            model_has_prefix = any(k.startswith('model.') for k in model_keys)

            if ckpt_has_prefix and not model_has_prefix:
                state_dict = {k[6:] if k.startswith('model.') else k: v
                              for k, v in state_dict.items()}
            elif not ckpt_has_prefix and model_has_prefix:
                state_dict = {f'model.{k}': v for k, v in state_dict.items()}

            if self._stopped:
                self.finished.emit(None)
                return

            self.progress.emit("Applying weights to model...")
            warning_msg = None
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                err = str(e)
                if "Missing key(s)" in err or "Unexpected key(s)" in err:
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    warning_msg = "Model loaded with some mismatches:\n\n"
                    if missing_keys:
                        warning_msg += f"Missing keys ({len(missing_keys)}): First 5:\n"
                        warning_msg += "\n".join([f"  - {k}" for k in missing_keys[:5]])
                        if len(missing_keys) > 5:
                            warning_msg += f"\n  ... and {len(missing_keys) - 5} more"
                        warning_msg += "\n\n"
                    if unexpected_keys:
                        warning_msg += f"Unexpected keys ({len(unexpected_keys)}): First 5:\n"
                        warning_msg += "\n".join([f"  - {k}" for k in unexpected_keys[:5]])
                        if len(unexpected_keys) > 5:
                            warning_msg += f"\n  ... and {len(unexpected_keys) - 5} more"
                    warning_msg += "\n\n[OK] Model loaded but may not work correctly."
                else:
                    raise

            model.eval()
            self.finished.emit({'model': model, 'warning': warning_msg})

        except Exception as e:
            self.finished.emit(e)


class GradCAMWorker(QThread):
    """Background thread for Grad-CAM computation"""
    finished = pyqtSignal(object)   # (heatmap, prediction) or exception
    progress = pyqtSignal(str)

    def __init__(self, model, image_path, transform, device, model_input_size,
                 target_layer, parent_viewer):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.transform = transform
        self.device = device
        self.model_input_size = model_input_size
        self.target_layer = target_layer
        self.viewer = parent_viewer
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            self.progress.emit("Preprocessing image...")
            img = Image.open(self.image_path).convert('RGB')
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)

            if self._stopped:
                self.finished.emit(None)
                return

            self.progress.emit("Running forward pass...")
            # FIX 2: explicit enable_grad for clarity
            with torch.enable_grad():
                self.model.zero_grad()
                output = self.model(input_tensor)
                prediction = torch.sigmoid(output).item()

                if self._stopped:
                    self.finished.emit(None)
                    return

                self.progress.emit("Computing gradients (backward pass)...")
                output.backward()

            activations = self.viewer.activations.clone()
            gradients = self.viewer.gradients.clone()

            if self._stopped:
                self.finished.emit(None)
                return

            self.progress.emit("Generating heatmap...")

            if len(activations.shape) == 3:
                # Vision Transformer
                batch_size, seq_len, channels = activations.shape
                activations = activations[:, 1:, :]
                gradients = gradients[:, 1:, :]

                seq_len = activations.shape[1]
                spatial_dim = int(seq_len ** 0.5)
                if spatial_dim * spatial_dim != seq_len:
                    raise ValueError(f"Cannot reshape seq_len {seq_len} into square")

                activations = activations.transpose(1, 2).reshape(
                    batch_size, channels, spatial_dim, spatial_dim)
                gradients = gradients.transpose(1, 2).reshape(
                    batch_size, channels, spatial_dim, spatial_dim)

                pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
                for i in range(channels):
                    activations[:, i, :, :] *= pooled_gradients[i]
                heatmap = torch.mean(activations, dim=1).squeeze()

            elif len(activations.shape) == 4:
                # CNN
                pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
                for i in range(activations.shape[1]):
                    activations[:, i, :, :] *= pooled_gradients[i]
                heatmap = torch.mean(activations, dim=1).squeeze()
            else:
                raise ValueError(f"Unexpected activation shape: {activations.shape}")

            heatmap = F.relu(heatmap).cpu().numpy()
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            heatmap = cv2.resize(heatmap, (self.model_input_size, self.model_input_size))

            self.finished.emit((heatmap, prediction))

        except Exception as e:
            self.finished.emit(e)


class ISICMetadataWorker(QThread):
    """Background thread for fetching ISIC Archive metadata via REST API"""
    finished = pyqtSignal(object)  # dict result, or Exception
    progress = pyqtSignal(str)

    ISIC_API_BASE = "https://api.isic-archive.com/api/v2/images"

    def __init__(self, isic_id):
        super().__init__()
        self.isic_id = isic_id
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            url = f"{self.ISIC_API_BASE}/{self.isic_id}/"
            self.progress.emit(f"Fetching metadata for {self.isic_id}...")

            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            if self._stopped:
                self.finished.emit(None)
                return

            self.finished.emit(data)

        except urllib.error.HTTPError as e:
            if e.code == 404:
                self.finished.emit(
                    ValueError(f"Image '{self.isic_id}' not found in ISIC Archive (404)."))
            else:
                self.finished.emit(
                    ConnectionError(f"ISIC API returned HTTP {e.code}: {e.reason}"))
        except urllib.error.URLError as e:
            self.finished.emit(
                ConnectionError(f"Could not reach ISIC Archive API:\n{e.reason}"))
        except Exception as e:
            self.finished.emit(e)


# ---------------------------------------------------------------------------
# Progress dialog with stop button
# ---------------------------------------------------------------------------

class ProgressDialog(QDialog):
    """Non-modal-looking progress dialog with a working stop button"""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(420, 140)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self._worker = None

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Starting...")
        self.status_label.setStyleSheet("font-size: 11pt;")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        layout.addWidget(self.progress_bar)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(
            "background-color: #cc3333; color: white; font-weight: bold; padding: 6px;")
        self.stop_btn.clicked.connect(self._on_stop)
        layout.addWidget(self.stop_btn)

    def set_worker(self, worker):
        self._worker = worker

    def update_status(self, text):
        self.status_label.setText(text)

    def _on_stop(self):
        if self._worker is not None:
            self._worker.stop()
        self.status_label.setText("Stopping...")
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        # Treat close button same as stop
        if self._worker is not None:
            self._worker.stop()
        event.accept()


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

class GradCAMViewer(QMainWindow):
    """Enhanced Grad-CAM viewer with standalone and table modes"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grad-CAM Viewer V04")
        self.setGeometry(100, 100, 1800, 900)

        # Data
        self.df = None
        self.filtered_df = None
        self.current_index = 0
        self.csv_type = None
        self.model = None
        self.target_layer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Track loaded files
        self.loaded_csv_path = None
        self.loaded_model_name = None
        self.loaded_checkpoint_path = None
        self.standalone_mode = True

        # FIX 1: store hook handles so we can remove them on reload
        self.hook_handles = []

        # Grad-CAM hooks
        self.gradients = None
        self.activations = None

        # Model input size
        self.model_input_size = 336

        # Display settings
        self.overlay_opacity = 0.5
        self.current_colormap = cv2.COLORMAP_JET
        self.activation_threshold = 0.0
        self.split_view_active = False

        # Current display data (FIX 5: cached for lightweight refresh)
        self.current_heatmap = None
        self.current_image = None
        self.current_image_display = None  # resized to model_input_size
        self.current_prediction = 0.0
        self.current_image_path = None
        self.current_row_data = None  # cached row for table mode refresh

        # Active workers
        self._active_worker = None
        self._progress_dialog = None

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        self.init_ui()

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def show_error_dialog(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def show_info_dialog(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def _show_progress(self, title):
        """Create and show a progress dialog, return it"""
        dlg = ProgressDialog(title, self)
        dlg.show()
        QApplication.processEvents()
        return dlg

    def _close_progress(self):
        """Close progress dialog if open"""
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None

    # ------------------------------------------------------------------
    # EVA model factory
    # ------------------------------------------------------------------

    def get_eva_model_class(self):
        import timm

        class EVAModel(torch.nn.Module):
            def __init__(self, model_name='eva02_small_patch14_336.mim_in22k_ft_in1k',
                         num_classes=1, pretrained=True):
                super(EVAModel, self).__init__()
                self.model = timm.create_model(
                    model_name, pretrained=pretrained, num_classes=num_classes)
                self.img_size = self.model.default_cfg.get('input_size', [3, 336, 336])[1]

            def forward(self, x):
                return self.model(x)

            def get_image_size(self):
                return self.img_size

        return EVAModel

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Status bar
        status_layout = QHBoxLayout()
        self.mode_label = QLabel("Mode: Standalone")
        self.mode_label.setStyleSheet("color: blue; font-weight: bold;")
        status_layout.addWidget(self.mode_label)
        status_layout.addWidget(QLabel(" | "))

        self.csv_status_label = QLabel("CSV: Not loaded")
        self.csv_status_label.setStyleSheet("color: gray; font-weight: bold;")
        status_layout.addWidget(self.csv_status_label)
        status_layout.addWidget(QLabel(" | "))

        self.model_status_label = QLabel("Model: Not loaded")
        self.model_status_label.setStyleSheet("color: gray; font-weight: bold;")
        status_layout.addWidget(self.model_status_label)
        status_layout.addWidget(QLabel(" | "))

        self.checkpoint_status_label = QLabel("Checkpoint: None")
        self.checkpoint_status_label.setStyleSheet("color: gray; font-style: italic;")
        status_layout.addWidget(self.checkpoint_status_label)
        status_layout.addStretch()
        main_layout.addLayout(status_layout)

        # Control row 1
        control_layout1 = QHBoxLayout()
        self.load_image_btn = QPushButton("Load Single Image")
        self.load_image_btn.clicked.connect(self.load_single_image)
        control_layout1.addWidget(self.load_image_btn)

        load_table_btn = QPushButton("Load Quality Results Table")
        load_table_btn.clicked.connect(self.load_results_table)
        control_layout1.addWidget(load_table_btn)

        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model)
        control_layout1.addWidget(load_model_btn)

        self.fetch_isic_btn = QPushButton("Fetch ISIC Metadata")
        self.fetch_isic_btn.clicked.connect(self.fetch_isic_metadata)
        self.fetch_isic_btn.setToolTip("Fetch metadata from ISIC Archive API for the current image")
        control_layout1.addWidget(self.fetch_isic_btn)

        control_layout1.addStretch()
        main_layout.addLayout(control_layout1)

        # Control row 2
        control_layout2 = QHBoxLayout()

        control_layout2.addWidget(QLabel("Overlay Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setMaximumWidth(150)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        control_layout2.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("50%")
        self.opacity_label.setMinimumWidth(40)
        control_layout2.addWidget(self.opacity_label)

        control_layout2.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['JET', 'HOT', 'VIRIDIS', 'TURBO'])
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        control_layout2.addWidget(self.colormap_combo)

        control_layout2.addWidget(QLabel("Activation Threshold:"))
        self.activation_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.activation_threshold_slider.setRange(0, 100)
        self.activation_threshold_slider.setValue(0)
        self.activation_threshold_slider.setMaximumWidth(150)
        self.activation_threshold_slider.valueChanged.connect(self.on_threshold_changed)
        control_layout2.addWidget(self.activation_threshold_slider)
        self.threshold_value_label = QLabel("0.00")
        self.threshold_value_label.setMinimumWidth(40)
        control_layout2.addWidget(self.threshold_value_label)

        self.split_view_btn = QPushButton("Toggle Split View")
        self.split_view_btn.clicked.connect(self.toggle_split_view)
        control_layout2.addWidget(self.split_view_btn)

        export_btn = QPushButton("Export High-Res Overlay")
        export_btn.clicked.connect(self.export_overlay)
        control_layout2.addWidget(export_btn)

        control_layout2.addStretch()
        main_layout.addLayout(control_layout2)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels([
            "Image", "Prediction", "Peak at Lesion", "Distance"])
        self.table_widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.table_widget.cellClicked.connect(self.on_table_cell_clicked)
        splitter.addWidget(self.table_widget)
        self.table_widget.hide()

        # Image panel
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)

        self.info_label = QLabel("Load an image and model to start")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setMinimumHeight(100)
        self.info_label.setStyleSheet("font-size: 14pt; border: 1px solid #ccc; padding: 10px;")
        image_layout.addWidget(self.info_label)

        self.metadata_label = QLabel("Image Metadata: Not available")
        self.metadata_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.metadata_label.setMinimumHeight(80)
        self.metadata_label.setMaximumHeight(150)
        self.metadata_label.setWordWrap(True)
        self.metadata_label.setStyleSheet(
            "font-size: 10pt; border: 1px solid #aaa; padding: 8px; "
            "background-color: #f9f9f9; font-family: monospace;")
        image_layout.addWidget(self.metadata_label)

        self.view_container = QWidget()
        self.view_layout = QVBoxLayout(self.view_container)

        # Split view
        self.split_view_widget = QWidget()
        split_layout = QHBoxLayout(self.split_view_widget)

        self.original_label = QLabel("Original")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(300, 300)
        self.original_label.setStyleSheet("border: 1px solid #ccc;")
        split_layout.addWidget(self.original_label)

        self.heatmap_label = QLabel("Heatmap")
        self.heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heatmap_label.setMinimumSize(300, 300)
        self.heatmap_label.setStyleSheet("border: 1px solid #ccc;")
        split_layout.addWidget(self.heatmap_label)

        self.overlay_label = QLabel("Overlay")
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setMinimumSize(300, 300)
        self.overlay_label.setStyleSheet("border: 1px solid #ccc;")
        split_layout.addWidget(self.overlay_label)

        self.view_layout.addWidget(self.split_view_widget)
        self.split_view_widget.hide()

        self.single_image_label = QLabel("Overlay View")
        self.single_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.single_image_label.setMinimumSize(600, 600)
        self.single_image_label.setStyleSheet("border: 1px solid #ccc;")
        self.view_layout.addWidget(self.single_image_label)

        image_layout.addWidget(self.view_container)

        # Navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_btn)

        self.index_label = QLabel("0 / 0")
        self.index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.index_label)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_btn)

        image_layout.addLayout(nav_layout)
        self.prev_btn.hide()
        self.next_btn.hide()
        self.index_label.hide()

        splitter.addWidget(image_panel)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # Model loading (FIX 1, FIX 4, + progress dialog)
    # ------------------------------------------------------------------

    def load_model(self):
        # FIX 4: QFileDialog returns (path, filter_string), not (path, bool)
        checkpoint_path, _filter = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint", "",
            "Checkpoint Files (*.pth *.pt);;All Files (*)")

        if not checkpoint_path:
            return

        # Show progress dialog
        self._progress_dialog = self._show_progress("Loading Model")

        worker = ModelLoadWorker(checkpoint_path, self.device,
                                 self.get_eva_model_class)
        worker.progress.connect(self._progress_dialog.update_status)
        worker.finished.connect(
            lambda result: self._on_model_loaded(result, checkpoint_path))
        self._progress_dialog.set_worker(worker)
        self._active_worker = worker
        worker.start()

    def _on_model_loaded(self, result, checkpoint_path):
        self._close_progress()
        self._active_worker = None

        if result is None:
            # User stopped
            return

        if isinstance(result, Exception):
            self.show_error_dialog(
                "Model Loading Error",
                f"Error loading model:\n\n{result}\n\nPlease check the checkpoint file.")
            print(f"Error: {result}")
            traceback.print_exc()
            return

        model = result['model']
        warning_msg = result['warning']

        # FIX 1: remove old hooks before registering new ones
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

        self.model = model
        self.model.eval()

        # Find target layer
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'blocks'):
            self.target_layer = self.model.model.blocks[-1]
        elif hasattr(self.model, 'blocks'):
            self.target_layer = self.model.blocks[-1]
        elif hasattr(self.model, 'features'):
            self.target_layer = self.model.features[-1]
        else:
            self.show_error_dialog("Model Error",
                                   "Could not find suitable target layer for Grad-CAM.")
            self.model = None
            return

        # FIX 1: register hooks and store handles
        def forward_hook(module, input, output):
            self.activations = output.clone().detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].clone().detach()

        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(backward_hook))

        # Detect input size
        if hasattr(self.model, 'get_image_size'):
            self.model_input_size = self.model.get_image_size()
        elif hasattr(self.model, 'img_size'):
            self.model_input_size = self.model.img_size
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'patch_embed'):
            sz = self.model.model.patch_embed.img_size
            self.model_input_size = sz[0] if isinstance(sz, tuple) else sz
        elif hasattr(self.model, 'patch_embed'):
            sz = self.model.patch_embed.img_size
            self.model_input_size = sz[0] if isinstance(sz, tuple) else sz
        else:
            self.model_input_size = 336

        self.transform = transforms.Compose([
            transforms.Resize((self.model_input_size, self.model_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        self.loaded_model_name = "EVA-02 Small (336x336)"
        self.loaded_checkpoint_path = Path(checkpoint_path).name

        self.model_status_label.setText(f"Model: {self.loaded_model_name}")
        self.model_status_label.setStyleSheet("color: green; font-weight: bold;")
        self.checkpoint_status_label.setText(f"Checkpoint: {self.loaded_checkpoint_path}")
        self.checkpoint_status_label.setStyleSheet("color: green; font-style: normal;")

        if warning_msg:
            self.show_info_dialog("Model Loaded with Warnings", warning_msg)
        else:
            self.show_info_dialog(
                "Model Loaded",
                f"EVA-02 Small model loaded successfully\n"
                f"Image size: {self.model_input_size}x{self.model_input_size}\n"
                f"Checkpoint: {self.loaded_checkpoint_path}")

    # ------------------------------------------------------------------
    # Single image loading + Grad-CAM (with progress)
    # ------------------------------------------------------------------

    def load_single_image(self):
        if self.model is None:
            self.info_label.setText("Please load a model first")
            return

        file_path, _filter = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)")

        if not file_path:
            return

        self.current_image_path = file_path
        self.standalone_mode = True
        self.mode_label.setText("Mode: Standalone")
        self.mode_label.setStyleSheet("color: blue; font-weight: bold;")

        self.table_widget.hide()
        self.prev_btn.hide()
        self.next_btn.hide()
        self.index_label.hide()

        self._run_gradcam_for_image(file_path)

    def _run_gradcam_for_image(self, image_path, row_data=None):
        """Start Grad-CAM computation in background thread"""
        # Load the original image synchronously (fast)
        img = cv2.imread(image_path)
        if img is None:
            self.info_label.setText(f"Failed to load image: {image_path}")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_image = img.copy()
        self.current_image_display = cv2.resize(
            img, (self.model_input_size, self.model_input_size))
        self.current_row_data = row_data

        if self.model is None:
            # No model - show image only
            if row_data is not None:
                self._show_image_only(row_data, image_path)
            else:
                self.info_label.setText("No model loaded")
            return

        # Show progress dialog
        self._progress_dialog = self._show_progress("Generating Grad-CAM")

        worker = GradCAMWorker(
            self.model, image_path, self.transform, self.device,
            self.model_input_size, self.target_layer, self)
        worker.progress.connect(self._progress_dialog.update_status)
        worker.finished.connect(
            lambda result: self._on_gradcam_done(result, image_path, row_data))
        self._progress_dialog.set_worker(worker)
        self._active_worker = worker
        worker.start()

    def _on_gradcam_done(self, result, image_path, row_data):
        self._close_progress()
        self._active_worker = None

        if result is None:
            # Stopped by user
            return

        if isinstance(result, Exception):
            self.info_label.setText(f"Grad-CAM error: {result}")
            print(f"Error generating Grad-CAM: {result}")
            traceback.print_exc()
            if row_data is not None:
                self._show_image_only(row_data, image_path)
            return

        heatmap, prediction = result
        self.current_heatmap = heatmap
        self.current_prediction = prediction

        # Update info + metadata labels
        self._update_info_label(prediction, image_path, row_data)
        self._update_metadata_label(image_path, row_data)

        # Render from cache
        self._refresh_display()

        # Update navigation in table mode
        if not self.standalone_mode and self.filtered_df is not None:
            self.index_label.setText(
                f"{self.current_index + 1} / {len(self.filtered_df)}")
            self.table_widget.selectRow(self.current_index)

    # ------------------------------------------------------------------
    # FIX 5: Lightweight display refresh (no re-inference)
    # ------------------------------------------------------------------

    def _refresh_display(self):
        """Re-render overlay from cached heatmap + image. No model inference."""
        if self.current_image_display is None:
            return

        img_display = self.current_image_display

        if self.current_heatmap is not None:
            heatmap = self.current_heatmap

            # Apply threshold
            mask = heatmap >= self.activation_threshold
            heatmap_thresholded = heatmap * mask
            heatmap_colored_thresh = cv2.applyColorMap(
                np.uint8(255 * heatmap_thresholded), self.current_colormap)
            heatmap_colored_thresh = cv2.cvtColor(
                heatmap_colored_thresh, cv2.COLOR_BGR2RGB)
            heatmap_pure = heatmap_colored_thresh.copy()

            overlay = cv2.addWeighted(
                img_display, 1 - self.overlay_opacity,
                heatmap_colored_thresh, self.overlay_opacity, 0)

            if self.split_view_active:
                self._display_image(self.original_label, img_display, 300)
                self._display_image(self.heatmap_label, heatmap_pure, 300)
                self._display_image(self.overlay_label, overlay, 300)
            else:
                self._display_image(self.single_image_label, overlay, 600)
        else:
            # No heatmap - show plain image
            if self.split_view_active:
                self._display_image(self.original_label, img_display, 300)
                blank = np.zeros_like(img_display)
                self._display_image(self.heatmap_label, blank, 300)
                self._display_image(self.overlay_label, blank, 300)
            else:
                self._display_image(self.single_image_label, img_display, 600)

    # ------------------------------------------------------------------
    # Info / metadata helpers
    # ------------------------------------------------------------------

    def _update_info_label(self, prediction, image_path, row_data=None):
        risk_level = "HIGH RISK" if prediction >= 0.5 else "LOW RISK"
        risk_color = "#FF0000" if prediction >= 0.5 else "#00AA00"

        if row_data is not None and self.csv_type == 'evaluation':
            true_label = "Unknown"
            if 'true_label' in row_data.index:
                true_label = "Malignant" if row_data['true_label'] == 1 else "Benign"

            correct = row_data.get('correct', None)
            if correct is not None:
                # FIX 7: ASCII symbols instead of unicode
                correct_symbol = "[OK]" if correct else "[X]"
                correct_color = "#00AA00" if correct else "#FF0000"
                correct_text = (
                    f"<span style='color: {correct_color}; font-weight: bold;'>"
                    f"{correct_symbol} {'Correct' if correct else 'Incorrect'}</span>")
            else:
                correct_text = "<span style='color: #888;'>N/A</span>"

            distance_text = "N/A"
            if 'distance_to_center' in row_data.index:
                distance_text = f"{row_data['distance_to_center']:.1f}px"

            info_text = f"""
            <div style='text-align: center;'>
                <span style='font-size: 18pt; font-weight: bold; color: {risk_color};'>
                    {risk_level} ({prediction:.4f})
                </span><br>
                <span style='font-size: 14pt;'>
                    True Label: <b>{true_label}</b> | {correct_text}
                </span><br>
                <span style='font-size: 12pt;'>Distance: {distance_text}</span>
            </div>"""
        elif row_data is not None:
            distance_text = "N/A"
            if 'distance_to_center' in row_data.index:
                distance_text = f"{row_data['distance_to_center']:.1f}px"
            info_text = f"""
            <div style='text-align: center;'>
                <span style='font-size: 18pt; font-weight: bold; color: {risk_color};'>
                    {risk_level} ({prediction:.4f})
                </span><br>
                <span style='font-size: 12pt;'>Distance: {distance_text}</span>
            </div>"""
        else:
            info_text = f"""
            <div style='text-align: center;'>
                <span style='font-size: 18pt; font-weight: bold; color: {risk_color};'>
                    {risk_level} ({prediction:.4f})
                </span><br>
                <span style='font-size: 12pt;'>
                    Malignancy Probability: {prediction:.4f}
                </span><br>
                <span style='font-size: 10pt; color: #555;'>
                    File: {Path(image_path).name}
                </span>
            </div>"""

        self.info_label.setText(info_text)

    def _update_metadata_label(self, image_path, row_data=None):
        metadata_text = self.format_metadata(
            row_data=row_data, image_path=image_path)
        self.metadata_label.setText(f"<b>Metadata:</b> {metadata_text}")

    # ------------------------------------------------------------------
    # CSV / table loading (with progress)
    # ------------------------------------------------------------------

    def load_results_table(self):
        file_path, _filter = QFileDialog.getOpenFileName(
            self, "Select Results CSV", "",
            "CSV Files (*.csv);;All Files (*)")

        if not file_path:
            return

        dlg = self._show_progress("Loading CSV")
        dlg.update_status(f"Reading {Path(file_path).name}...")
        QApplication.processEvents()

        try:
            self.df = pd.read_csv(file_path)
            self.loaded_csv_path = file_path
            csv_filename = Path(file_path).name

            dlg.update_status("Validating columns...")
            QApplication.processEvents()

            # Find image path column
            image_path_column = None
            possible_names = ['image_path', 'filepath', 'path', 'file_path', 'image_file']
            for col_name in possible_names:
                if col_name in self.df.columns:
                    image_path_column = col_name
                    break

            if image_path_column is None:
                dlg.close()
                error_msg = f"Critical: No image path column found in {csv_filename}\n\n"
                error_msg += f"Looking for one of: {', '.join(possible_names)}\n\n"
                error_msg += "Available columns:\n"
                error_msg += "\n".join([f"  - {col}" for col in self.df.columns[:10]])
                if len(self.df.columns) > 10:
                    error_msg += f"\n  ... and {len(self.df.columns) - 10} more"
                error_msg += "\n\nCannot continue without image paths."
                self.show_error_dialog("Missing Critical Column", error_msg)
                return

            if image_path_column != 'image_path':
                self.df.rename(columns={image_path_column: 'image_path'}, inplace=True)
                print(f"Info: Renamed column '{image_path_column}' to 'image_path'")

            # Collect warnings
            warnings = []
            if 'prob_malignant' in self.df.columns:
                self.csv_type = 'evaluation'
                eval_cols = ['prob_malignant', 'true_label']
                missing_eval = [c for c in eval_cols if c not in self.df.columns]
                if missing_eval:
                    warnings.append("Evaluation columns missing:\n  "
                                    + "\n  ".join([f"- {c}" for c in missing_eval]))
            else:
                self.csv_type = 'gradcam'
                gc_cols = ['prediction', 'peak_at_lesion', 'distance_to_center']
                missing_gc = [c for c in gc_cols if c not in self.df.columns]
                if missing_gc:
                    warnings.append("Grad-CAM columns missing:\n  "
                                    + "\n  ".join([f"- {c}" for c in missing_gc]))

            meta_cols = ['isic_id', 'lesion_id', 'age_approx', 'sex',
                         'anatom_site_general', 'iddx_1', 'target']
            missing_meta = [c for c in meta_cols if c not in self.df.columns]
            if missing_meta and len(missing_meta) < len(meta_cols):
                warnings.append("Metadata columns missing:\n  "
                                + "\n  ".join([f"- {c}" for c in missing_meta]))

            dlg.update_status("Populating table...")
            QApplication.processEvents()

            self.csv_status_label.setText(f"CSV: {csv_filename} ({self.csv_type})")
            self.csv_status_label.setStyleSheet("color: green; font-weight: bold;")

            self.standalone_mode = False
            self.mode_label.setText("Mode: Table")
            self.mode_label.setStyleSheet("color: green; font-weight: bold;")

            self.table_widget.show()
            self.prev_btn.show()
            self.next_btn.show()
            self.index_label.show()

            self.apply_filter()

            dlg.close()

            # FIX 7: ASCII instead of checkmark
            if warnings:
                warning_msg = f"Information about {csv_filename}:\n\n"
                warning_msg += "\n\n".join(warnings)
                warning_msg += "\n\n[OK] Will continue with available columns."
                warning_msg += "\n[OK] Missing columns will show as 'N/A' in table and metadata."
                self.show_info_dialog("Missing Optional Columns", warning_msg)

        except FileNotFoundError:
            dlg.close()
            self.show_error_dialog("File Not Found", f"File not found:\n\n{file_path}")
        except pd.errors.EmptyDataError:
            dlg.close()
            self.show_error_dialog("Empty CSV File",
                                   f"CSV file is empty:\n\n{Path(file_path).name}")
        except pd.errors.ParserError as e:
            dlg.close()
            self.show_error_dialog(
                "CSV Parse Error",
                f"Error parsing CSV file {Path(file_path).name}:\n\n{e}\n\n"
                "Please check the file format.")
        except Exception as e:
            dlg.close()
            self.show_error_dialog(
                "CSV Loading Error",
                f"Error loading CSV {Path(file_path).name}:\n\n{e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Table population & navigation
    # ------------------------------------------------------------------

    def apply_filter(self):
        if self.df is None or self.standalone_mode:
            return
        self.filtered_df = self.df.reset_index(drop=True)
        self.populate_table()
        self.current_index = 0
        if len(self.filtered_df) > 0:
            self.show_current_image()

    def populate_table(self):
        if self.filtered_df is None:
            return
        try:
            self.table_widget.setRowCount(len(self.filtered_df))
            for idx, row in self.filtered_df.iterrows():
                try:
                    if 'image_path' in row.index:
                        self.table_widget.setItem(
                            idx, 0, QTableWidgetItem(str(row['image_path'])))
                    else:
                        self.table_widget.setItem(idx, 0, QTableWidgetItem("N/A"))

                    if self.csv_type == 'evaluation':
                        if 'prob_malignant' in row.index:
                            self.table_widget.setItem(
                                idx, 1, QTableWidgetItem(f"{row['prob_malignant']:.3f}"))
                        else:
                            self.table_widget.setItem(idx, 1, QTableWidgetItem("N/A"))
                    else:
                        if 'prediction' in row.index:
                            self.table_widget.setItem(
                                idx, 1, QTableWidgetItem(f"{row['prediction']:.3f}"))
                        else:
                            self.table_widget.setItem(idx, 1, QTableWidgetItem("N/A"))

                    if 'peak_at_lesion' in row.index:
                        self.table_widget.setItem(
                            idx, 2, QTableWidgetItem(f"{row['peak_at_lesion']:.3f}"))
                    else:
                        self.table_widget.setItem(idx, 2, QTableWidgetItem("N/A"))

                    if 'distance_to_center' in row.index:
                        self.table_widget.setItem(
                            idx, 3, QTableWidgetItem(f"{row['distance_to_center']:.1f}"))
                    else:
                        self.table_widget.setItem(idx, 3, QTableWidgetItem("N/A"))

                except Exception as e:
                    self.table_widget.setItem(
                        idx, 0, QTableWidgetItem(f"Error: {e}"))
        except Exception as e:
            self.show_error_dialog("Table Population Error",
                                   f"Error populating table:\n\n{e}")

    def on_table_cell_clicked(self, row, column):
        if self.standalone_mode:
            return
        self.current_index = row
        self.show_current_image()

    def show_current_image(self):
        if (self.filtered_df is None or len(self.filtered_df) == 0
                or self.standalone_mode):
            return
        try:
            row = self.filtered_df.iloc[self.current_index]
            image_path = row['image_path']
            self._run_gradcam_for_image(image_path, row_data=row)
        except Exception as e:
            self.info_label.setText(f"Error displaying image:\n\n{e}")
            print(f"Error in show_current_image: {e}")
            traceback.print_exc()

    def _show_image_only(self, row_data, image_path):
        """Show image without Grad-CAM when model not loaded or failed"""
        if self.current_image_display is None:
            return

        info_text = f"""
        <div style='text-align: center;'>
            <span style='font-size: 18pt; font-weight: bold; color: #0066CC;'>
                Image Preview (No Model Loaded)
            </span><br>
            <span style='font-size: 12pt; color: #666;'>
                Load a model to see predictions and Grad-CAM
            </span><br>
            <span style='font-size: 10pt; color: #999;'>
                File: {Path(image_path).name}
            </span>
        </div>"""
        self.info_label.setText(info_text)
        self._update_metadata_label(image_path, row_data)

        self.current_heatmap = None
        self._refresh_display()

        if not self.standalone_mode and self.filtered_df is not None:
            self.index_label.setText(
                f"{self.current_index + 1} / {len(self.filtered_df)}")
            self.table_widget.selectRow(self.current_index)

    # ------------------------------------------------------------------
    # Metadata formatting (FIX 6: bare except -> except Exception)
    # ------------------------------------------------------------------

    def format_metadata(self, row_data=None, image_path=None):
        if row_data is not None:
            metadata_fields = []

            def safe_get(column_name, label=None):
                if label is None:
                    label = column_name
                try:
                    if column_name in row_data.index:
                        value = row_data[column_name]
                        if pd.notna(value):
                            return f"{label}: {value}"
                        else:
                            return f"{label}: N/A"
                except Exception:  # FIX 6
                    pass
                return None

            field = safe_get('isic_id', 'ISIC ID')
            if field:
                metadata_fields.append(field)
            field = safe_get('lesion_id', 'Lesion ID')
            if field:
                metadata_fields.append(field)
            field = safe_get('age_approx', 'Age')
            if field:
                metadata_fields.append(field)
            field = safe_get('sex', 'Sex')
            if field:
                metadata_fields.append(field)
            field = safe_get('anatom_site_general', 'Location')
            if field:
                metadata_fields.append(field)
            field = safe_get('iddx_1', 'Diagnosis')
            if field:
                metadata_fields.append(field)

            try:
                if 'target' in row_data.index and pd.notna(row_data['target']):
                    target = row_data['target']
                    target_text = "Malignant" if target == 1 else "Benign"
                    metadata_fields.append(f"Label: {target_text}")
            except Exception:  # FIX 6
                pass

            try:
                if 'image_path' in row_data.index and pd.notna(row_data['image_path']):
                    img_name = Path(row_data['image_path']).name
                    metadata_fields.append(f"File: {img_name}")
            except Exception:  # FIX 6
                pass

            if metadata_fields:
                return "  |  ".join(metadata_fields)
            else:
                return "No metadata available in CSV"

        elif image_path is not None:
            return f"File: {Path(image_path).name}"

        return "No metadata available"

    # ------------------------------------------------------------------
    # FIX 3: QImage .copy() to prevent data lifetime crash
    # ------------------------------------------------------------------

    def _display_image(self, label, img, size):
        img_resized = cv2.resize(img, (size, size))
        height, width, channel = img_resized.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_resized.data, width, height, bytes_per_line,
                       QImage.Format.Format_RGB888).copy()  # FIX 3
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap)

    # ------------------------------------------------------------------
    # Grad-CAM generation (kept as sync fallback, used by worker)
    # ------------------------------------------------------------------

    def generate_gradcam(self, image_path: str):
        """Synchronous Grad-CAM - used only as direct fallback"""
        if self.model is None:
            return None, 0.0
        try:
            img = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # FIX 2: explicit enable_grad
            with torch.enable_grad():
                self.model.zero_grad()
                output = self.model(input_tensor)
                prediction = torch.sigmoid(output).item()
                output.backward()

            activations = self.activations.clone()
            gradients = self.gradients.clone()

            if len(activations.shape) == 3:
                batch_size, seq_len, channels = activations.shape
                activations = activations[:, 1:, :]
                gradients = gradients[:, 1:, :]
                seq_len = activations.shape[1]
                spatial_dim = int(seq_len ** 0.5)
                if spatial_dim * spatial_dim != seq_len:
                    raise ValueError(f"Cannot reshape seq_len {seq_len} into square")
                activations = activations.transpose(1, 2).reshape(
                    batch_size, channels, spatial_dim, spatial_dim)
                gradients = gradients.transpose(1, 2).reshape(
                    batch_size, channels, spatial_dim, spatial_dim)
                pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
                for i in range(channels):
                    activations[:, i, :, :] *= pooled_gradients[i]
                heatmap = torch.mean(activations, dim=1).squeeze()
            elif len(activations.shape) == 4:
                pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
                for i in range(activations.shape[1]):
                    activations[:, i, :, :] *= pooled_gradients[i]
                heatmap = torch.mean(activations, dim=1).squeeze()
            else:
                raise ValueError(f"Unexpected activation shape: {activations.shape}")

            heatmap = F.relu(heatmap).cpu().numpy()
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            heatmap = cv2.resize(heatmap, (self.model_input_size, self.model_input_size))
            return heatmap, prediction

        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            traceback.print_exc()
            return None, 0.0

    # ------------------------------------------------------------------
    # FIX 5: Visualization controls call lightweight refresh
    # ------------------------------------------------------------------

    def on_opacity_changed(self, value):
        self.overlay_opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        self._refresh_display()

    def on_colormap_changed(self, colormap_name):
        colormap_dict = {
            'JET': cv2.COLORMAP_JET,
            'HOT': cv2.COLORMAP_HOT,
            'VIRIDIS': cv2.COLORMAP_VIRIDIS,
            'TURBO': cv2.COLORMAP_TURBO,
        }
        self.current_colormap = colormap_dict.get(colormap_name, cv2.COLORMAP_JET)
        self._refresh_display()

    def on_threshold_changed(self, value):
        self.activation_threshold = value / 100.0
        self.threshold_value_label.setText(f"{self.activation_threshold:.2f}")
        self._refresh_display()

    def toggle_split_view(self):
        self.split_view_active = not self.split_view_active
        if self.split_view_active:
            self.single_image_label.hide()
            self.split_view_widget.show()
        else:
            self.split_view_widget.hide()
            self.single_image_label.show()
        self._refresh_display()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_overlay(self):
        if self.current_image is None or self.current_heatmap is None:
            self.info_label.setText("No image to export")
            return

        save_path, _filter = QFileDialog.getSaveFileName(
            self, "Save Overlay", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")

        if not save_path:
            return

        try:
            img = cv2.resize(self.current_image,
                             (self.model_input_size, self.model_input_size))
            mask = self.current_heatmap >= self.activation_threshold
            heatmap_thresholded = self.current_heatmap * mask
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap_thresholded), self.current_colormap)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(
                img, 1 - self.overlay_opacity,
                heatmap_colored, self.overlay_opacity, 0)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, overlay_bgr)
            self.info_label.setText(f"Exported to {Path(save_path).name}")
        except Exception as e:
            self.info_label.setText(f"Export failed: {e}")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def show_previous(self):
        if (self.standalone_mode or self.filtered_df is None
                or len(self.filtered_df) == 0):
            return
        self.current_index = (self.current_index - 1) % len(self.filtered_df)
        self.show_current_image()

    def show_next(self):
        if (self.standalone_mode or self.filtered_df is None
                or len(self.filtered_df) == 0):
            return
        self.current_index = (self.current_index + 1) % len(self.filtered_df)
        self.show_current_image()

    def keyPressEvent(self, event):
        if (not self.standalone_mode and self.filtered_df is not None
                and len(self.filtered_df) > 0):
            if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Left):
                self.show_previous()
                event.accept()
            elif event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Right):
                self.show_next()
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # ISIC Archive metadata fetch
    # ------------------------------------------------------------------

    def _extract_isic_id(self):
        """Try to extract ISIC ID from current context (filename, CSV row, path)"""
        # Try from CSV row data
        if self.current_row_data is not None:
            for col in ['isic_id', 'ISIC_id', 'isic_ID']:
                if col in self.current_row_data.index:
                    val = self.current_row_data[col]
                    if pd.notna(val):
                        return str(val).strip()

        # Try to extract from filename (e.g. ISIC_0024306.jpg)
        path = self.current_image_path
        if path:
            stem = Path(path).stem  # e.g. "ISIC_0024306"
            if stem.upper().startswith("ISIC_"):
                return stem
        return None

    def fetch_isic_metadata(self):
        """Fetch metadata from ISIC Archive REST API for the current image"""
        isic_id = self._extract_isic_id()
        if isic_id is None:
            self.show_error_dialog(
                "No ISIC ID",
                "Could not determine ISIC ID for the current image.\n\n"
                "The filename must start with 'ISIC_' or the CSV must "
                "contain an 'isic_id' column.")
            return

        self._progress_dialog = self._show_progress("Fetching ISIC Metadata")

        worker = ISICMetadataWorker(isic_id)
        worker.progress.connect(self._progress_dialog.update_status)
        worker.finished.connect(self._on_isic_metadata_done)
        self._progress_dialog.set_worker(worker)
        self._active_worker = worker
        worker.start()

    def _on_isic_metadata_done(self, result):
        self._close_progress()
        self._active_worker = None

        if result is None:
            return  # stopped

        if isinstance(result, Exception):
            self.show_error_dialog("ISIC API Error", str(result))
            return

        # Build a readable display from the API response
        isic_id = result.get('isic_id', 'Unknown')
        meta = result.get('metadata', {})
        clinical = meta.get('clinical', {})
        acquisition = meta.get('acquisition', {})
        copyright_license = result.get('copyright_license', 'N/A')
        attribution = result.get('attribution', 'N/A')

        lines = []
        lines.append(f"<h3>ISIC Archive Metadata: {isic_id}</h3>")
        lines.append("<table style='border-collapse:collapse; width:100%;'>")

        def add_section(title):
            lines.append(
                f"<tr><td colspan='2' style='padding:6px 4px; font-weight:bold; "
                f"background:#4CAF50; color:white;'>{title}</td></tr>")

        def add_row(label, value):
            lines.append(
                f"<tr><td style='padding:3px 8px; border-bottom:1px solid #ddd; "
                f"font-weight:bold; width:40%;'>{label}</td>"
                f"<td style='padding:3px 8px; border-bottom:1px solid #ddd;'>"
                f"{value}</td></tr>")

        add_section("Clinical")
        field_map = [
            ('sex', 'Sex'), ('age_approx', 'Age (approx)'),
            ('diagnosis_1', 'Diagnosis 1'), ('diagnosis_2', 'Diagnosis 2'),
            ('diagnosis_3', 'Diagnosis 3'),
            ('diagnosis_confirm_type', 'Confirmation'),
            ('anatom_site_general', 'Anatomical Site'),
            ('benign_malignant', 'Benign/Malignant'),
            ('melanocytic', 'Melanocytic'),
            ('lesion_id', 'Lesion ID'),
            ('concomitant_biopsy', 'Concomitant Biopsy'),
            ('family_hx_mm', 'Family Hx Melanoma'),
            ('personal_hx_mm', 'Personal Hx Melanoma'),
            ('nevus_type', 'Nevus Type'),
            ('mel_class', 'Melanoma Class'),
            ('mel_type', 'Melanoma Type'),
            ('mel_thick_mm', 'Breslow Thickness (mm)'),
            ('mel_mitotic_index', 'Mitotic Index'),
            ('mel_ulcer', 'Ulceration'),
        ]
        has_clinical = False
        for key, label in field_map:
            if key in clinical and clinical[key] is not None:
                add_row(label, clinical[key])
                has_clinical = True
        if not has_clinical:
            add_row("(none)", "No clinical metadata available")

        add_section("Acquisition")
        acq_map = [
            ('pixels_x', 'Width (px)'), ('pixels_y', 'Height (px)'),
            ('image_type', 'Image Type'),
            ('dermoscopic_type', 'Dermoscopic Type'),
        ]
        has_acq = False
        for key, label in acq_map:
            if key in acquisition and acquisition[key] is not None:
                add_row(label, acquisition[key])
                has_acq = True
        if not has_acq:
            add_row("(none)", "No acquisition metadata available")

        add_section("License / Attribution")
        add_row("License", copyright_license)
        add_row("Attribution", attribution)
        add_row("Public", result.get('public', 'N/A'))

        lines.append("</table>")

        # Show in a resizable dialog
        dlg = QDialog(self)
        dlg.setWindowTitle(f"ISIC Metadata - {isic_id}")
        dlg.setMinimumSize(500, 400)
        dlg.resize(550, 550)
        layout = QVBoxLayout(dlg)

        label = QLabel("".join(lines))
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setAlignment(Qt.AlignmentFlag.AlignTop)
        label.setStyleSheet("font-size: 10pt; padding: 8px;")

        from PyQt6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        dlg.exec()


def main():
    app = QApplication(sys.argv)
    viewer = GradCAMViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()