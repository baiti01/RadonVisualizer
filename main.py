#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Ti Bai
# Email: tibaiw@gmail.com
# datetime:4/2/2025
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Ti Bai
# Email: tibaiw@gmail.com
# datetime:4/2/2025
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Ti Bai
# Email: tibaiw@gmail.com
# datetime:4/2/2025
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Ti Bai
# Email: tibaiw@gmail.com
# datetime:4/2/2025
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Ti Bai
# Email: tibaiw@gmail.com
# datetime:4/2/2025
import os
from pathlib import Path
import io

import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from PIL import Image

# Set up the Streamlit page
st.set_page_config(page_title="3D Medical Image Viewer", layout="wide")
st.title("3D Medical Image Viewer (1:1 Pixel Display)")

# Sidebar: folder selection and controls
st.sidebar.header("Data Selection & Controls")
folder_path = st.sidebar.text_input("📁 Folder containing NIfTI files", value="", placeholder="Enter path to folder")


def rgba_to_hex(rgba):
    """Convert an RGBA tuple (0-1 range) to a hex color string."""
    r, g, b, a = rgba
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


@st.cache_data
def load_nifti_data(folder):
    """
    Load the mandatory image and optional dose and structure masks.
    Apply the initial flip transformation: volume = volume[:, ::-1, ::-1].
    Returns:
      image_data, image_spacing,
      dose_data, dose_spacing,
      structures: dict mapping structure name to {"data": array, "spacing": tuple}
    """
    folder = Path(folder)
    image_file = folder / "image.nii.gz"
    if not image_file.exists():
        raise FileNotFoundError("Mandatory file 'image.nii.gz' not found in the folder.")
    # Load main image and apply initial flip
    img_obj = nib.load(str(image_file))
    image_data = np.asanyarray(img_obj.dataobj)
    image_data = image_data[:, ::-1, ::-1]
    image_spacing = img_obj.header.get_zooms()[:3]

    # Dose is optional
    dose_file = folder / "dose.nii.gz"
    if dose_file.exists():
        dose_obj = nib.load(str(dose_file))
        dose_data = np.asanyarray(dose_obj.dataobj)
        dose_data = dose_data[:, ::-1, ::-1]
        dose_spacing = dose_obj.header.get_zooms()[:3]
    else:
        dose_data = None
        dose_spacing = None

    # Load structure masks (all other .nii.gz files)
    structures = {}
    for file_path in folder.glob("*.nii.gz"):
        if file_path.name in ["image.nii.gz", "dose.nii.gz"]:
            continue
        try:
            struct_obj = nib.load(str(file_path))
            mask_array = np.asanyarray(struct_obj.dataobj)
            mask_array = mask_array[:, ::-1, ::-1]
            mask_spacing = struct_obj.header.get_zooms()[:3]
            struct_name = file_path.stem  # file name without extension
            structures[struct_name] = {"data": mask_array, "spacing": mask_spacing}
        except Exception as e:
            st.warning(f"Could not load {file_path.name}: {e}")
    return image_data, image_spacing, dose_data, dose_spacing, structures


# --- Pre-transformation functions for each view ---
def transform_volume_for_view(vol, view):
    """
    Apply a combined 3D transformation to pre-orient a volume for a given view.
    Assumes vol has already been flipped (vol[:, ::-1, ::-1]).

    For each view:
      - axial: rotate the x-y plane clockwise by 90° (axes=(0,1))
      - sagittal: rotate the y-z plane by 90° counterclockwise (axes=(1,2))
      - coronal: rotate the x-z plane by 90° counterclockwise and flip left-right (axes=(0,2))
    """
    if view == "axial":
        # Axial: transform entire volume along x-y plane, then flip vertically.
        return np.flip(np.rot90(vol, k=1, axes=(0,1)), axis=0)
    elif view == "sagittal":
        return np.rot90(vol, k=1, axes=(1, 2))
    elif view == "coronal":
        return np.fliplr(np.rot90(vol, k=1, axes=(0, 2)))
    else:
        return vol


def crop_to_union(image_vol, dose_vol, structures):
    """
    Crop volumes to the bounding box covering the union of all structure masks.
    If no structure exists, returns the original volumes.
    """
    if not structures:
        return image_vol, dose_vol, structures, None
    union_mask = np.zeros_like(image_vol, dtype=bool)
    for struct in structures.values():
        union_mask |= (struct["data"] > 0)
    if not union_mask.any():
        return image_vol, dose_vol, structures, None
    coords = np.array(np.nonzero(union_mask))
    min_indices = coords.min(axis=1)
    max_indices = coords.max(axis=1) + 1  # include last index
    slices = tuple(slice(mi, ma) for mi, ma in zip(min_indices, max_indices))
    image_crop = image_vol[slices]
    dose_crop = dose_vol[slices] if dose_vol is not None else None
    structures_crop = {}
    for name, struct in structures.items():
        mask_crop = struct["data"][slices]
        structures_crop[name] = {"data": mask_crop, "spacing": struct["spacing"]}
    return image_crop, dose_crop, structures_crop, slices


def resample_volume(volume, original_spacing, target_spacing, order=1):
    """
    Resample a 3D volume to isotropic resolution.
    """
    zoom_factors = [orig / target_spacing for orig in original_spacing]
    return zoom(volume, zoom=zoom_factors, order=order)


@st.cache_data(show_spinner=False)
def preprocess_data(folder_path):
    """
    Load, crop, resample, and pre-transform the volumes for all views.
    Returns:
      image_views: dict with keys "axial", "sagittal", "coronal" for the main image
      dose_views: same as above for dose (or None)
      structure_views: dict mapping structure name to a dict for each view
      target_spacing: the resolution used for resampling
    """
    image_vol, image_spacing, dose_vol, dose_spacing, structures = load_nifti_data(folder_path)
    image_vol, dose_vol, structures, crop_slices = crop_to_union(image_vol, dose_vol, structures)

    # Choose target resolution: larger of max(image_spacing) and 2 mm.
    target_spacing = max(max(image_spacing), 2.0)

    # Resample volumes
    image_resampled = resample_volume(image_vol, image_spacing, target_spacing, order=1)
    if dose_vol is not None and dose_spacing is not None:
        dose_resampled = resample_volume(dose_vol, dose_spacing, target_spacing, order=1)
    else:
        dose_resampled = None
    structures_resampled = {}
    for name, struct in structures.items():
        structures_resampled[name] = {
            "data": resample_volume(struct["data"], struct["spacing"], target_spacing, order=0),
            "spacing": (target_spacing, target_spacing, target_spacing)
        }

    # Apply pre-transformation for each view so that later we only need to index.
    image_views = {
        "axial": transform_volume_for_view(image_resampled, "axial"),
        "sagittal": transform_volume_for_view(image_resampled, "sagittal"),
        "coronal": transform_volume_for_view(image_resampled, "coronal")
    }
    dose_views = None
    if dose_resampled is not None:
        dose_views = {
            "axial": transform_volume_for_view(dose_resampled, "axial"),
            "sagittal": transform_volume_for_view(dose_resampled, "sagittal"),
            "coronal": transform_volume_for_view(dose_resampled, "coronal")
        }
    structure_views = {}
    for name, struct in structures_resampled.items():
        structure_views[name] = {
            "axial": transform_volume_for_view(struct["data"], "axial"),
            "sagittal": transform_volume_for_view(struct["data"], "sagittal"),
            "coronal": transform_volume_for_view(struct["data"], "coronal")
        }

    return image_views, dose_views, structure_views, target_spacing


# --- Composite slice display via PIL with cross-lines ---
def composite_slice_to_pil(base_slice, window_min, window_max, dose_slice=None, dose_params=None, structure_slices=None,
                           structure_colors=None, cross_lines=False, cross_coords=(None, None)):
    """
    Render a composite slice (base image with optional dose and structure overlays) and return a PIL Image.
    The image is created with a 1:1 mapping (each voxel becomes one pixel).
    Optionally, if cross_lines is True and cross_coords is provided (tuple of (x,y) coordinates),
    dashed yellow lines are drawn at those positions.
    """
    height, width = base_slice.shape
    dpi = 100
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(base_slice, cmap="gray", origin="lower", vmin=window_min, vmax=window_max, interpolation="nearest")
    if dose_slice is not None and dose_params is not None:
        dose_low, dose_high = dose_params
        dose_masked = np.ma.masked_less(dose_slice, dose_low)
        ax.imshow(dose_masked, cmap="jet", origin="lower", alpha=0.5, vmin=dose_low, vmax=dose_high,
                  interpolation="nearest")
    if structure_slices is not None:
        for name, mask in structure_slices.items():
            if np.any(mask):
                color = structure_colors.get(name, "#00FF00")
                ax.contour(mask, levels=[0.5], colors=[color], origin="lower", linewidths=1.5)
    if cross_lines and cross_coords[0] is not None and cross_coords[1] is not None:
        ax.axvline(x=cross_coords[0], color='yellow', linestyle='--')
        ax.axhline(y=cross_coords[1], color='yellow', linestyle='--')
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# --- Sidebar: load and adjust parameters ---
if folder_path:
    try:
        image_views, dose_views, structure_views, target_spacing = preprocess_data(folder_path)
    except Exception as e:
        st.sidebar.error(f"Error processing data: {e}")
        st.stop()

    # For the main image, use the axial view to determine intensity range.
    img_axial = image_views["axial"]
    img_min = float(img_axial.min())
    img_max = float(img_axial.max())
    if img_min < -200:
        modality = "CT"
        default_window = (-250, 250)
    else:
        modality = "MRI"
        default_window = (img_min, img_max)
    default_window = (max(img_min, default_window[0]), min(img_max, default_window[1]))

    st.sidebar.write(f"Detected Modality: {modality}")
    window_min, window_max = st.sidebar.slider(
        "Image Intensity Window",
        min_value=img_min, max_value=img_max,
        value=default_window,
        step=0.1 if modality == "CT" else 0.01 * (img_max - img_min)
    )

    # Dose overlay controls (if dose exists)
    if dose_views is not None:
        dose_ax = dose_views["axial"]
        dose_min_val = float(dose_ax.min())
        dose_max_val = float(dose_ax.max())
        default_dose_window = (0.5 * dose_max_val, dose_max_val)
        dose_low, dose_high = st.sidebar.slider(
            "Dose Overlay Window",
            min_value=dose_min_val, max_value=dose_max_val,
            value=default_dose_window,
            step=0.01 * dose_max_val
        )
        show_dose = st.sidebar.checkbox("Show Dose Overlay", value=True)
        dose_params = (dose_low, dose_high)
    else:
        show_dose = False
        dose_params = None

    # Structure overlays: list available structures with colorized labels.
    structure_keys = sorted(structure_views.keys())
    if structure_keys:
        if st.sidebar.button("Hide All Structure Overlays"):
            for key in structure_keys:
                st.session_state[f"structure_{key}"] = False
    mask_visibility = {}
    structure_colors = {}
    for idx, key in enumerate(structure_keys):
        color_rgba = plt.cm.tab10(idx % 10)
        hex_color = rgba_to_hex(color_rgba)
        structure_colors[key] = hex_color
        default_val = st.session_state.get(f"structure_{key}", True)
        col_cb, col_label = st.sidebar.columns([0.15, 0.85])
        with col_cb:
            mask_visibility[key] = st.checkbox("", value=default_val, key=f"structure_{key}")
        with col_label:
            st.markdown(f"<span style='color:{hex_color}; font-weight:bold'>{key}</span>", unsafe_allow_html=True)

    # --- Display Views ---
    # Instead of applying per-slice orientation functions, we now index into the pre-transformed volumes.
    col_axial, col_sagittal, col_coronal = st.columns(3)
    # Get dimensions from one view (they should match the transformed volume)
    nx_axial, ny_axial, nz_axial = image_views["axial"].shape  # axial view: shape (nx, ny, nz)
    nx_sag, ny_sag, nz_sag = image_views["sagittal"].shape  # sagittal view: shape (nx, ny, nz)
    nx_cor, ny_cor, nz_cor = image_views["coronal"].shape  # coronal view: shape (nx, ny, nz)

    # For crosslines, we assume:
    # - In the axial view, a vertical line indicates the sagittal slice position and a horizontal line indicates the coronal slice.
    # - In the sagittal view, a vertical line indicates the coronal slice position and a horizontal line indicates the axial slice.
    # - In the coronal view, a vertical line indicates the sagittal slice position and a horizontal line indicates the axial slice.

    # Axial: slice along third dimension.
    with col_axial:
        st.subheader("Axial")
        axial_index = st.slider("Axial Slice", 0, nz_axial - 1, nz_axial // 2, key="axial_slider")
        base_slice = image_views["axial"][:, :, axial_index]
        dose_slice = None
        if show_dose and dose_views is not None:
            dose_slice = dose_views["axial"][:, :, axial_index]
        # Cross-lines: vertical at sagittal slider, horizontal at coronal slider.
        sagittal_index = st.session_state.get("sagittal_slider", nx_sag // 2)
        coronal_index = st.session_state.get("coronal_slider", ny_cor // 2)
        cross_coords_axial = (sagittal_index, ny_cor - coronal_index)
        struct_slices = {}
        for key in structure_keys:
            if mask_visibility.get(key, False):
                struct_slices[key] = structure_views[key]["axial"][:, :, axial_index]
        pil_img = composite_slice_to_pil(base_slice, window_min, window_max,
                                         dose_slice=dose_slice, dose_params=dose_params,
                                         structure_slices=struct_slices, structure_colors=structure_colors,
                                         cross_lines=True, cross_coords=cross_coords_axial)
        st.image(pil_img, use_column_width=False)

    # Sagittal: slice along first dimension.
    with col_sagittal:
        st.subheader("Sagittal")
        sagittal_index = st.slider("Sagittal Slice", 0, nx_sag - 1, nx_sag // 2, key="sagittal_slider")
        base_slice = image_views["sagittal"][sagittal_index, :, :]
        dose_slice = None
        if show_dose and dose_views is not None:
            dose_slice = dose_views["sagittal"][sagittal_index, :, :]
        # Cross-lines: vertical at coronal slider, horizontal at axial slider.
        axial_index_for_sag = st.session_state.get("axial_slider", nz_axial // 2)
        coronal_index_for_sag = st.session_state.get("coronal_slider", ny_cor // 2)
        cross_coords_sagittal = (ny_cor - coronal_index_for_sag, nz_axial - axial_index_for_sag)
        struct_slices = {}
        for key in structure_keys:
            if mask_visibility.get(key, False):
                struct_slices[key] = structure_views[key]["sagittal"][sagittal_index, :, :]
        pil_img = composite_slice_to_pil(base_slice, window_min, window_max,
                                         dose_slice=dose_slice, dose_params=dose_params,
                                         structure_slices=struct_slices, structure_colors=structure_colors,
                                         cross_lines=True, cross_coords=cross_coords_sagittal)
        st.image(pil_img, use_column_width=False)

    # Coronal: slice along second dimension.
    with col_coronal:
        st.subheader("Coronal")
        coronal_index = st.slider("Coronal Slice", 0, ny_cor - 1, ny_cor // 2, key="coronal_slider")
        base_slice = image_views["coronal"][:, coronal_index, :]
        dose_slice = None
        if show_dose and dose_views is not None:
            dose_slice = dose_views["coronal"][:, coronal_index, :]
        # Cross-lines: vertical at sagittal slider, horizontal at axial slider.
        sagittal_index_for_cor = st.session_state.get("sagittal_slider", nx_sag // 2)
        axial_index_for_cor = st.session_state.get("axial_slider", nz_axial // 2)
        cross_coords_coronal = (sagittal_index_for_cor, nz_axial - axial_index_for_cor)
        struct_slices = {}
        for key in structure_keys:
            if mask_visibility.get(key, False):
                struct_slices[key] = structure_views[key]["coronal"][:, coronal_index, :]
        pil_img = composite_slice_to_pil(base_slice, window_min, window_max,
                                         dose_slice=dose_slice, dose_params=dose_params,
                                         structure_slices=struct_slices, structure_colors=structure_colors,
                                         cross_lines=True, cross_coords=cross_coords_coronal)
        st.image(pil_img, use_column_width=False)
