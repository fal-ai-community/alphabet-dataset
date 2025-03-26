import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import string
from torchvision import transforms
from torch.utils.data import DataLoader
import io

from dataset import ShapeDataset


def display_image_with_details(image_tensor, params, dataset):
    """
    Display image with its parameters.

    Args:
        image_tensor: PyTorch tensor of shape [3, H, W]
        params: PyTorch tensor with shape parameters
        dataset: The dataset instance for decoding parameters
    """
    # Convert tensor to numpy for visualization
    img = image_tensor.permute(1, 2, 0).numpy()

    # Decode parameters
    decoded = dataset.decode_params(params)

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis("off")
    plt.tight_layout()

    # Return the figure and parameters
    return fig, decoded


def create_labeled_sample(
    image_tensor, params, dataset, include_bboxes=False, include_text=True
):
    """
    Create a visualization of a sample with labels and bounding boxes.

    Args:
        image_tensor: PyTorch tensor of shape [3, H, W]
        params: PyTorch tensor with shape parameters
        dataset: The dataset instance for decoding parameters
        include_bboxes: Whether to draw bounding boxes around shapes

    Returns:
        PIL Image with annotations
    """
    img_np = image_tensor.permute(1, 2, 0).numpy() * 255
    img = Image.fromarray(img_np.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Get a smaller font for annotations
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()

    # Decode the parameters
    decoded = dataset.decode_params(params)

    # Draw bounding boxes and labels for each shape
    for i, shape in enumerate(decoded["shapes"]):
        x1, y1, x2, y2 = shape["position"]

        # Draw bounding box if requested
        if include_bboxes:
            draw.rectangle([x1, y1, x2, y2], outline="white", width=2)

        # Draw shape label (shape letter + index)
        label = f"{shape['shape']}{i+1}"
        text_color = (
            "black" if sum(shape["color"]) > 382 else "white"
        )  # Choose contrasting text color

        # Draw text background for better visibility
        if include_text:
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(
                text_bbox, fill="white" if text_color == "black" else "black"
            )
            draw.text((x1, y1), label, fill=text_color, font=font)

    return img


# Set page title
st.set_page_config(page_title="Advanced Synthetic Dataset Visualizer", layout="wide")

# Create a header
st.title("Advanced Synthetic Dataset Visualizer")
st.write(
    "Explore the advanced synthetic dataset of images with multiple letter shapes."
)

# Create sidebar controls
with st.sidebar:
    st.header("Dataset Parameters")

    dataset_length = st.slider(
        "Dataset Length", min_value=10, max_value=1000, value=100, step=10
    )
    image_size = st.slider(
        "Image Size", min_value=128, max_value=512, value=256, step=32
    )
    max_shapes = st.slider(
        "Max Shapes Per Image", min_value=1, max_value=10, value=3, step=1
    )
    seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)

    st.header("Navigation")
    sample_index = st.slider(
        "Sample Index", min_value=0, max_value=dataset_length - 1, value=0, step=1
    )

    st.header("Visualization Options")
    show_bboxes = st.checkbox("Show Bounding Boxes", value=True)
    show_text = st.checkbox("Show Text", value=True)
    st.header("Batch Visualization")
    batch_size = st.slider("Batch Size", min_value=1, max_value=16, value=4, step=1)
    show_batch = st.checkbox("Show Batch", value=False)

    if st.button("Generate New Dataset"):
        # This will trigger a full rerun
        st.session_state.force_new_dataset = True

# Check if we need to initialize or re-initialize the dataset
if (
    "advanced_dataset" not in st.session_state
    or "advanced_dataset_params" not in st.session_state
    or st.session_state.get("advanced_dataset_params")
    != (dataset_length, image_size, max_shapes, seed)
    or st.session_state.get("force_new_dataset", False)
):

    # Create a new dataset
    with st.spinner("Generating dataset..."):
        try:
            st.session_state.advanced_dataset = ShapeDataset(
                length=dataset_length,
                image_size=image_size,
                max_shapes=max_shapes,
                seed=seed,
            )
            st.session_state.advanced_dataset_params = (
                dataset_length,
                image_size,
                max_shapes,
                seed,
            )
            if "force_new_dataset" in st.session_state:
                st.session_state.force_new_dataset = False
        except Exception as e:
            st.error(f"Error creating dataset: {e}")
            st.stop()

# Get the current dataset
dataset = st.session_state.advanced_dataset

# Display single sample
st.header(f"Sample #{sample_index}")

try:
    # Get the sample
    x, y = dataset[sample_index]
    print(x, y)

    # Create columns for the visualization
    col1, col2 = st.columns([3, 2])

    with col1:
        # Create and display the labeled image
        labeled_img = create_labeled_sample(
            x, y, dataset, include_bboxes=show_bboxes, include_text=show_text
        )
        st.image(labeled_img, use_column_width=True)

    with col2:
        st.subheader("Image Parameters")
        
        st.markdown("Raw Parameter : ")
        st.code(y.tolist(), language="json")
        # Decode parameters
        decoded = dataset.decode_params(y)

        # Show background color
        bg_color = decoded["background_color"]
        st.markdown(f"**Background Color:** RGB{bg_color}")

        # Display a color swatch
        bg_color_html = f"rgb({bg_color[0]}, {bg_color[1]}, {bg_color[2]})"
        st.markdown(
            f"""
        <div style="background-color: {bg_color_html}; width: 50px; height: 20px; display: inline-block;"></div>
        """,
            unsafe_allow_html=True,
        )

        # Show number of shapes
        st.markdown(f"**Number of Shapes:** {decoded['num_shapes']}")

        # Show details for each shape
        for i, shape in enumerate(decoded["shapes"]):
            st.markdown(f"### Shape {i+1}: {shape['shape']}")

            col_a, col_b = st.columns([1, 3])

            with col_a:
                # Display color swatch
                shape_color = shape["color"]
                shape_color_html = (
                    f"rgb({shape_color[0]}, {shape_color[1]}, {shape_color[2]})"
                )
                st.markdown(
                    f"""
                <div style="background-color: {shape_color_html}; width: 50px; height: 50px; display: inline-block;"></div>
                """,
                    unsafe_allow_html=True,
                )

            with col_b:
                # Display shape details
                st.markdown(f"**Color:** RGB{shape_color}")
                st.markdown(f"**Position:** {shape['position']}")
                width = shape["position"][2] - shape["position"][0]
                height = shape["position"][3] - shape["position"][1]
                st.markdown(f"**Size:** {width}×{height} pixels")

    # Display a batch of samples if requested
    if show_batch:
        st.header(f"Batch Visualization (Size: {batch_size})")

        # Create a DataLoader for batch sampling
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Get the specific batch that contains our sample
        batch_number = sample_index // batch_size
        batch_x, batch_y = None, None

        for i, (x_batch, y_batch) in enumerate(dataloader):
            if i == batch_number:
                batch_x, batch_y = x_batch, y_batch
                break

        # Display the batch as a grid
        if batch_x is not None:
            # Calculate grid dimensions
            n_cols = min(4, batch_size)
            n_rows = (batch_size + n_cols - 1) // n_cols  # Ceiling division

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

            # Flatten axes if it's a multi-dimensional array
            if n_rows > 1 or n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]  # Make it iterable if it's a single plot

            # Plot each image in the batch
            for i in range(batch_size):
                if i < len(batch_x):
                    # Create labeled image
                    img = create_labeled_sample(
                        batch_x[i],
                        batch_y[i],
                        dataset,
                        include_bboxes=show_bboxes,
                        include_text=show_text,
                    )

                    # Convert PIL Image to numpy array for matplotlib
                    img_np = np.array(img)

                    # Display the image
                    axes[i].imshow(img_np)
                    axes[i].set_title(f"Sample #{batch_number * batch_size + i}")
                    axes[i].axis("off")

            # Hide any unused subplots
            for i in range(batch_size, len(axes)):
                axes[i].axis("off")

            plt.tight_layout()
            st.pyplot(fig)

except Exception as e:
    st.error(f"Error displaying sample: {e}")

# Add some extra information
st.sidebar.markdown("---")
st.sidebar.info(
    """
This visualizer allows you to explore the advanced synthetic dataset with multiple letter shapes.
- Use the sliders to adjust dataset parameters
- Navigate through samples with the sample index slider
- Toggle bounding boxes around shapes
- View a batch of samples by enabling the 'Show Batch' option
"""
)
