import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import string
import os
import random
from functools import lru_cache
import requests
from io import BytesIO


@lru_cache
def get_font(bold=True, size=32):
    if bold:
        font_url = "https://github.com/openmaptiles/fonts/raw/master/noto-sans/NotoSans-Bold.ttf"
    else:
        font_url = "https://github.com/openmaptiles/fonts/raw/master/noto-sans/NotoSans-Regular.ttf"
    font_response = requests.get(font_url)
    font_data = BytesIO(font_response.content)
    font = ImageFont.truetype(font_data, size)
    return font


class ShapeDataset(Dataset):

    def __init__(self, length=1000, image_size=256, max_shapes=3, seed=42, offset=512, nocolor=False, granularity=8, download_url=None):
        """
        Initialize the dataset.

        Args:
            length (int): Number of samples in the dataset.
            image_size (int): Size of the square images.
            max_shapes (int): Maximum number of shapes per image.
            seed (int): Random seed for reproducibility.
            offset (int): Offset value for shape type encoding.
            granularity (int): Granularity of the parameters. Make everything multiple of this, so its less difficult if granularity is larger.
        """
        self.length = length
        self.image_size = image_size
        self.max_shapes = max_shapes
        self.generator = torch.Generator().manual_seed(seed)
        self.offset = offset
        self.nocolor = nocolor
        self.granularity = granularity
        # Available shapes (using alphabet letters as shapes)
        self.alphabet = list(string.ascii_uppercase)  # A-Z
        self.num_shapes = len(self.alphabet)

        # Try to load font
        self.font_size = image_size // 3 # A reasonable size for the shape
        self.font = get_font(size = self.font_size)

        if download_url is not None:
            self.download_url = download_url
            self._download_parameters()
        else:
            # Generate all parameters for the entire dataset at initialization
            self._generate_parameters()

        # Transform to convert PIL images to tensors
        self.to_tensor = transforms.ToTensor()
        
    def _download_parameters(self):
        # save here
        import requests
        import os
        # Create a directory for the downloaded files
        os.makedirs("./shapedataset_cache", exist_ok=True)
        filename = os.path.basename(self.download_url)
        filepath = os.path.join("./shapedataset_cache", filename)
        if not os.path.exists(filepath):
            response = requests.get(self.download_url)
            with open(filepath, "wb") as f:
                f.write(response.content)
                
        self.parameters = torch.load(filepath, weights_only=False)
        
    def _generate_parameters(self):
        """Generate all parameters for the dataset."""
        # Generate the number of shapes for each image (1 to max_shapes)
        self.num_shapes_per_image = torch.randint(
            1, self.max_shapes + 1, (self.length,), generator=self.generator
        )

        self.bg_colors = torch.randint(
            0, 256, (self.length, 3), generator=self.generator
        )
        
        if self.nocolor:
            self.bg_colors = torch.ones_like(self.bg_colors) * 255

        # Max size for the parameters:
        # For each possible shape, we need:
        # - Shape type (integer): 1
        # - Position (x1, y1, x2, y2): 4
        # - Color (RGB): 3
        # Total per shape: 1 + 4 + 3 = 8
        max_param_size = 3 + (8 * self.max_shapes)  # 3 for background RGB

        # Initialize parameter tensor
        # We'll fill it with -1 for unused parameters (when an image has fewer than max_shapes)
        self.parameters = torch.full(
            (self.length, max_param_size), -1, dtype=torch.float
        )

        # Fill in background colors for all images
        self.parameters[:, 0:3] = self.bg_colors
        # For each image
        for i in range(self.length):
            num_shapes = self.num_shapes_per_image[i].item()

            # For each shape in this image
            for j in range(num_shapes):
                # Calculate starting index for this shape's parameters
                start_idx = 3 + (j * 8)

                # Generate shape type (random letter from alphabet)
                shape_idx = torch.randint(
                    0, self.num_shapes, (1,), generator=self.generator
                ).item()
                
                # Encode shape as integer from OFFSET to OFFSET + num_shapes
                shape_int = self.offset + shape_idx

                # Store the integer shape type
                self.parameters[i, start_idx] = shape_int

                # Generate position and size
                # Ensure reasonable positioning based on font size
                margin = self.font_size // 2
                x1 = torch.randint(
                    margin, self.image_size - margin, (1,), generator=self.generator
                ).item()
                y1 = torch.randint(
                    margin, self.image_size - margin, (1,), generator=self.generator
                ).item()

                # Size is determined by the font, but we need x2, y2 for bbox
                # We'll estimate it based on font_size (characters are roughly square)
                x2 = min(x1 + self.font_size, self.image_size - 1)
                y2 = min(y1 + self.font_size, self.image_size - 1)

                # Put position into parameters
                pos_idx = start_idx + 1
                self.parameters[i, pos_idx : pos_idx + 4] = torch.tensor([x1, y1, x2, y2])

                # Generate color (RGB, 0-255)
                color = torch.randint(0, 256, (3,), generator=self.generator)

                # Put color into parameters
                color_idx = pos_idx + 4
                self.parameters[i, color_idx : color_idx + 3] = color
                
            _param = (self.parameters / self.granularity).long() * self.granularity
            self.parameters = torch.where(self.parameters == -1, self.parameters, _param)
                
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.length

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, label) where image is a 3xHxW tensor and
                  label is a tensor containing shape parameters.
        """
        if idx >= self.length:
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {self.length}"
            )

        # Get parameters for this sample
        params = self.parameters[idx]

        # Get background color
        bg_color = (int(params[0] ), int(params[1] ), int(params[2] ))

        # Create image with background color
        img = Image.new("RGB", (self.image_size, self.image_size), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Determine how many shapes are in this image
        num_shapes = self.num_shapes_per_image[idx].item()

        # Draw each shape
        for j in range(num_shapes):
            # Calculate starting index for this shape's parameters
            start_idx = 3 + (j * 8)

            # Get shape type (integer encoding)
            shape_int = params[start_idx]
            if shape_int < 0:  # Check for unused shape slot
                continue
                
            # Convert from integer to alphabet index
            shape_idx = int(shape_int - self.offset)
            shape_char = self.alphabet[shape_idx]

            # Get position (normalized 0-1)
            pos_idx = start_idx + 1
            x1, y1, x2, y2 = params[pos_idx : pos_idx + 4]

            # Get color (normalized 0-1)
            color_idx = pos_idx + 4
            r, g, b = params[color_idx : color_idx + 3]
            color = (int(r), int(g), int(b))

            # Draw the character as the shape
            draw.text((x1, y1), shape_char, fill=color, font=self.font)

        # Convert to tensor
        x = self.to_tensor(img)  # Shape: [3, image_size, image_size]

        # Return image and parameters
        y = params.clone().long()

        return x, y + 1

    def decode_params(self, params):
        params = params - 1
        """
        Decode parameters into a more readable format.

        Args:
            params (torch.Tensor): Parameters tensor for one sample.

        Returns:
            dict: Dictionary with decoded parameters.
        """
        # Get background color
        bg_color = [int(c) for c in params[0:3].tolist()]

        shapes = []
        num_shapes = 0

        # Check each potential shape slot
        for j in range(self.max_shapes):
            start_idx = 3 + (j * 8)

            # If we've reached the end of the parameters tensor or found an unused slot
            if start_idx >= len(params) or params[start_idx] < 0:
                break

            # Get shape type (integer encoding)
            shape_int = int(params[start_idx])
            shape_idx = shape_int - self.offset
            
            # Validate shape index
            if shape_idx < 0 or shape_idx >= len(self.alphabet):
                continue
                
            shape_char = self.alphabet[shape_idx]

            # Get position
            pos_idx = start_idx + 1
            x1, y1, x2, y2 = (params[pos_idx : pos_idx + 4]).tolist()

            # Get color
            color_idx = pos_idx + 4
            color = [int(c) for c in params[color_idx : color_idx + 3].tolist()]

            shapes.append(
                {
                    "shape": shape_char,
                    "shape_int": shape_int,
                    "position": [round(p) for p in [x1, y1, x2, y2]],
                    "color": color,
                }
            )

            num_shapes += 1

        return {
            "background_color": bg_color,
            "num_shapes": num_shapes,
            "shapes": shapes,
        }

    def visualize_sample(self, idx):
        """
        Visualize a sample from the dataset.

        Args:
            idx (int): Index of the sample to visualize.

        Returns:
            None (displays the image)
        """
        x, y = self.__getitem__(idx)

        # Convert tensor to numpy for visualization
        img = x.permute(1, 2, 0).numpy()

        # Decode parameters
        decoded = self.decode_params(y)

        # Create figure
        plt.figure(figsize=(10, 8))
        plt.imshow(img)

        title = f"Sample {idx}\nBg RGB: {decoded['background_color']}\nShapes: {decoded['num_shapes']}"
        plt.title(title)

        plt.axis("off")

        # Add shape information as text
        shape_info = ""
        for i, shape in enumerate(decoded["shapes"]):
            shape_info += f"Shape {i+1}: {shape['shape']} (Int: {shape['shape_int']}), Color: {shape['color']}, Pos: {shape['position']}\n"

        plt.figtext(
            0.5,
            0.01,
            shape_info,
            ha="center",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.5},
        )
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = ShapeDataset(length=100, max_shapes=3, seed=42, offset=10)

    # Get a sample
    x, y = dataset[0]
    print("Image shape:", x.shape)
    print("Label shape:", y.shape)

    # Decode the parameters
    decoded = dataset.decode_params(y)
    print("Decoded parameters:", decoded)

    # Visualize a few samples
    for i in range(3):
        dataset.visualize_sample(i)

    # Create DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Get a batch
    images, labels = next(iter(dataloader))
    print("Batch of images shape:", images.shape)
    print("Batch of labels shape:", labels.shape)