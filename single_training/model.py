# ===== Standard Library Imports =====
from pathlib import Path
# ===== Third-Party Imports =====
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from tqdm import tqdm
from prettytable import PrettyTable
# ===== Own Modules =====
from .custom_cnn import CustomCNN  
from settings import setting
import functions as fn

class CNN_Model():

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the CNN model wrapper with settings from the configuration file.
    # Sets up paths, model parameters, and internal state variables.
    def __init__(self) -> None:

        # Paths
        self.pth_train = setting["pth_train"]
        self.pth_input = setting["pth_input"]
        self.pth_checkpoint = setting["pth_checkpoint"]
        self.chckpt_pth = setting["pth_checkpoint"]

        # Input shape data
        self.input_channels = setting["img_channels"]
        self.batch_size = setting["ds_batch_size"]
        self.img_width = setting["img_width"]
        self.img_height = setting["img_height"]
        self.input_size = (self.batch_size, self.input_channels, self.img_height, self.img_width)

        # Model configuration
        self.cnn_type = setting["cnn_type"]
        self.is_pretrained = setting["cnn_is_pretrained"]
        self.initialization = setting["cnn_initialization"]
        self.chckpt_save = setting["chckpt_save"]

        # Class information
        self.class_list = setting["classes"]
        self.num_classes = len(self.class_list)

        # Model state
        self.model = None
        self.model_loaded = False
        self.checkpoint_loaded = False
        self.loaded_checkpoint_name = None

    #############################################################################################################
    # METHODS

    # Universal CNN loader supporting multiple architectures:
    #   - ResNet (18/34/50/101/152)
    #   - ResNeXt-101 (32x8d/64x4d)
    #   - AlexNet
    #   - VGG (11/13/16/19, with/without BN)
    #   - DenseNet (121/169/201)
    #   - EfficientNet (B0/B3/B4/B7)
    #   - ConvNeXt (Tiny/Small)
    #   - Custom CNN architecture
    # Args:
    #   device (torch.device): The device to load the model on
    # Returns:
    #   nn.Module: The loaded model
    def load_model(self, device: torch.device) -> nn.Module:
        # Supported models and their constructors
        model_dict = {
            # ResNet variants
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
            # ResNeXt variants
            "resnext101_32x8d": models.resnext101_32x8d,
            "resnext101_64x4d": models.resnext101_64x4d,
            # AlexNet
            "alexnet": models.alexnet,
            # VGG variants
            "vgg11": models.vgg11,
            "vgg13": models.vgg13,
            "vgg16": models.vgg16,
            "vgg19": models.vgg19,
            "vgg11_bn": models.vgg11_bn,
            "vgg13_bn": models.vgg13_bn,
            "vgg16_bn": models.vgg16_bn,
            "vgg19_bn": models.vgg19_bn,
            # DenseNet variants
            "densenet121": models.densenet121,
            "densenet169": models.densenet169,
            "densenet201": models.densenet201,
            # EfficientNet variants
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b3": models.efficientnet_b3,
            "efficientnet_b4": models.efficientnet_b4,
            "efficientnet_b7": models.efficientnet_b7,
            # ConvNeXt variants
            "convnext_tiny": models.convnext_tiny,
            "convnext_small": models.convnext_small,
            # Custom model constructor
            "custom": self._build_custom_model
        }

        # Custom Model Case
        if self.cnn_type == "custom":
            self.is_pretrained = False
            self.model = CustomCNN(
                input_channels=self.input_channels,
                num_classes=self.num_classes,
                batch_size=self.batch_size,
                img_size=(self.img_height, self.img_width),
                dropout=setting.get("cnn_dropout", 0.5)
            )

        # Predefined Model Case
        else:
            constructor = model_dict[self.cnn_type]
            self.model = constructor(weights="DEFAULT" if self.is_pretrained else None)

            # Adapt for grayscale input (only for predefined models)
            if self.input_channels == 1:
                original_conv = self._adapt_first_conv(self.model, self.input_channels)
                if self.is_pretrained and original_conv:
                    # Transfer pretrained weights by averaging RGB channels
                    if hasattr(self.model, 'conv1'):
                        self.model.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                    elif hasattr(self.model.features, '0'):
                        self.model.features[0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

            elif self.input_channels not in [1, 3]:
                raise ValueError("Input must be 1 (grayscale) or 3 (RGB) channels")

        # Modify final layer for all models
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
        elif hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Sequential):
                in_features = self.model.classifier[-1].in_features
                self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
            else:
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Linear(in_features, self.num_classes)
        # ConvNeXt classifier
        elif hasattr(self.model, 'classifier') and hasattr(self.model.classifier, '2'):
            if isinstance(self.model.classifier, nn.Sequential):
                for i, layer in enumerate(self.model.classifier):
                    if isinstance(layer, nn.Linear):
                        in_features = layer.in_features
                        self.model.classifier[i] = nn.Linear(in_features, self.num_classes)
                        break

        # Initialize non-pretrained weights
        if not self.is_pretrained:
            self.model.apply(self._init_weights)

        # Final setup
        self.model.to(device)
        self.model_loaded = True
        return self.model

    # Construct the custom CNN architecture.
    # Returns:
    #   CustomCNN: The custom CNN model instance
    def _build_custom_model(self) -> 'CustomCNN':
        return CustomCNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            batch_size=self.batch_size,
            img_size=(self.img_height, self.img_width),
            dropout=setting.get("cnn_dropout", 0.5)
        )

    # Weight initialization for non-pretrained CNNs.
    # Args:
    #   m (nn.Module): The module to initialize
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            if self.initialization == "kaiming":
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif self.initialization == "xavier":
                init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Modify the first convolutional layer to match the input channel count.
    # Args:
    #   model (nn.Module): The model to adapt
    #   input_channels (int): Number of input channels (1 or 3)
    # Returns:
    #   nn.Conv2d or None: The original convolutional layer if replaced, else None
    def _adapt_first_conv(self, model: nn.Module, input_channels: int):
        # ResNet/ResNeXt/AlexNet
        if hasattr(model, 'conv1'):
            original_conv = model.conv1
            model.conv1 = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            return original_conv

        # VGG
        elif hasattr(model, 'features') and isinstance(model.features[0], nn.Conv2d):
            original_conv = model.features[0]
            model.features[0] = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            return original_conv

        # DenseNet
        elif hasattr(model, 'features') and hasattr(model.features, 'conv0'):
            original_conv = model.features.conv0
            model.features.conv0 = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            return original_conv

        # EfficientNet
        elif hasattr(model, 'features') and hasattr(model.features[0], '0'):
            original_conv_block = model.features[0]
            original_conv = original_conv_block[0]
            new_conv = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            model.features[0][0] = new_conv
            return original_conv

        # ConvNeXt
        elif hasattr(model, 'features') and hasattr(model.features, '0'):
            if hasattr(model.features[0], '0') and isinstance(model.features[0][0], nn.Conv2d):
                original_conv = model.features[0][0]
                model.features[0][0] = nn.Conv2d(
                    input_channels,
                    original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=False
                )
                return original_conv

        return None

    # Read classes from training data directory (subfolder names).
    # Returns:
    #   list: Sorted list of class names
    def get_class_list(self) -> list:
        class_list = sorted([j.name for j in self.pth_train.iterdir() if j.is_dir()])
        return class_list

    # Print all classes with their count.
    def print_class_list(self) -> None:
        print("Number of classes:", self.num_classes)
        print("Classes: ", end="")
        print(', '.join(self.class_list))

    # Get a list of all checkpoint files in the checkpoint directory.
    # Args:
    #   pth_checkpoint (Path): Directory containing checkpoints
    #   extensions (list): List of file extensions to look for
    # Returns:
    #   list: List of tuples (id, filename) sorted by filename
    def get_checkpoints_list(self, pth_checkpoint: Path, extensions: list = ['.model', '.pt']) -> list:
        checkpoints_list = []
        for ext in extensions:
            checkpoints_list.extend([file.name for file in pth_checkpoint.glob(f'*{ext}')])

        checkpoints_list = sorted(set(checkpoints_list))
        return [(i + 1, name) for i, name in enumerate(checkpoints_list)]

    # Print a table of available checkpoints.
    # Args:
    #   pth_checkpoint (Path): Directory containing checkpoints
    #   print_table (bool): If True, print the table
    # Returns:
    #   list: List of tuples (id, filename)
    def print_checkpoints_table(self, pth_checkpoint: Path, print_table: bool = True) -> list:
        checkpoints = self.get_checkpoints_list(pth_checkpoint)

        if print_table and checkpoints:
            table = PrettyTable(["ID", "Checkpoint"])
            for display_id, name in checkpoints:
                table.add_row([display_id, name])
            print()
            print(table)

        return checkpoints

    # Prompt the user to select a checkpoint from the list.
    # Args:
    #   checkpoints (list): List of tuples (id, filename)
    #   prompt (str): Prompt to display
    # Returns:
    #   str: Selected filename, or None if selection fails
    def select_checkpoint(self, checkpoints: list, prompt: str) -> str:
        max_id = len(checkpoints)
        while True:
            nr = input(prompt)
            if not fn.check_int(nr):
                print("Input is not an integer number! Try again...")
            else:
                nr = int(nr)
                if not fn.check_int_range(nr, 1, max_id):
                    print("Index out of range! Try again...")
                else:
                    return checkpoints[nr - 1][1]

    # Load weights from a checkpoint file.
    # Args:
    #   chckpt_pth (Path): Directory containing the checkpoint
    #   chckpt_file (str): Name of the checkpoint file
    # Returns:
    #   str: The loaded checkpoint filename
    def load_weights(self, chckpt_pth: Path, chckpt_file: str) -> str:
        checkpoint = torch.load(chckpt_pth / chckpt_file)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f'Weights from checkpoint {chckpt_file} successfully loaded.')
        return chckpt_file

    # Load a checkpoint interactively.
    # If only one checkpoint exists, load it automatically.
    # If multiple exist, show a table and prompt for selection.
    # Returns:
    #   bool: True if loading succeeded, False otherwise
    def load_checkpoint(self) -> bool:
        silent_checkpoints = self.print_checkpoints_table(self.pth_checkpoint, print_table=False)

        if not silent_checkpoints:
            print("The checkpoint folder is empty!")
            return False

        if len(silent_checkpoints) == 1:
            checkpoint_file = silent_checkpoints[0][1]
            print(f"\nFound single checkpoint: {checkpoint_file}")
            print("Loading automatically...")
        else:
            self.print_checkpoints_table(self.pth_checkpoint)
            checkpoint_file = self.select_checkpoint(silent_checkpoints, "Select a checkpoint: ")
            if not checkpoint_file:
                return False

        try:
            full_path = self.pth_checkpoint / checkpoint_file
            self.load_weights(self.pth_checkpoint, checkpoint_file)
            self.checkpoint_loaded = True
            self.loaded_checkpoint_name = full_path.stem
            return True
        except FileNotFoundError as e:
            print(f"\nError loading checkpoint: {str(e)}")
            print(f"Full path attempted: {full_path}")
            return False
        except Exception as e:
            print(f"\nError loading checkpoint: {str(e)}")
            return False

    # Predict classes for a given dataset.
    # Args:
    #   dataset (torch.utils.data.Dataset): The dataset to predict on
    # Returns:
    #   tuple: (accuracy, confusion_matrix_dict) where the dict has 'y' and 'y_hat' keys
    def predict(self, dataset) -> tuple:
        num_correct = 0
        num_samples = 0
        cm = {"y": [], "y_hat": []}

        self.model.eval()

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(dataset)):
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()

                scores = self.model(images)
                _, predictions = scores.max(1)

                num_correct += (predictions == labels).sum()
                num_samples += predictions.size(0)

                cm["y"].append(labels.item())
                cm["y_hat"].append(predictions.item())

        acc = num_correct / num_samples
        self.model.train()
        return acc, cm

    # Print the number of trainable parameters in the model.
    def print_model_size(self) -> None:
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {total_params:,}")

    # Capture output shapes of all layers through forward hooks.
    # Args:
    #   device (torch.device): The device to run the forward pass on
    # Returns:
    #   tuple: (layer_info_dict, hook_list) for cleanup
    def get_layer_info(self, device: torch.device) -> tuple:
        layer_info = {}
        hooks = []

        def hook_fn(module, input, output):
            if id(module) not in layer_info:
                params = 0
                if not list(module.children()):
                    params = sum(p.numel() for p in module.parameters())

                layer_info[id(module)] = {
                    'name': module.__class__.__name__,
                    'output': list(output.shape),
                    'params': params,
                    'trainable': any(p.requires_grad for p in module.parameters())
                }

        for name, module in self.model.named_modules():
            if not list(module.children()):
                hooks.append(module.register_forward_hook(hook_fn))

        dummy_input = torch.rand(*self.input_size).to(device)
        self.model.to(device).eval()
        with torch.no_grad():
            self.model(dummy_input)

        return layer_info, hooks

    # Generate an accurate model summary table without double-counting parameters.
    # Skips layers without trainable parameters by default.
    # Args:
    #   device (torch.device): The device to run the summary on
    #   show_non_trainable (bool): If True, show non-trainable layers as well
    def model_summary(self, device: torch.device, show_non_trainable: bool = False) -> None:
        layer_info, hooks = self.get_layer_info(device)

        table = PrettyTable()
        table.field_names = ["Layer (type)", "Output Shape", "Param #", "Trainable"]
        table.align = "l"

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        for module_id, info in layer_info.items():
            if not show_non_trainable:
                if info['params'] == 0 and not info['trainable']:
                    continue

            table.add_row([
                info['name'],
                str(info['output']),
                f"{info['params']:,}",
                str(info['trainable'])
            ])

        for hook in hooks:
            hook.remove()

        print(table)
        print(f"\nInput: {list(self.input_size)} (Device: {device})")
        print(f"Total params: {total_params:,} (ground truth)")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")