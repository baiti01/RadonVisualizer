#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Ti Bai
# Email: tibaiw@gmail.com
# datetime:4/2/2025
import os
from pathlib import Path
import io
import concurrent.futures

import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from PIL import Image

# Set up the Streamlit page
st.set_page_config(page_title="RadonVisualizer", layout="wide")
st.title("RadonVisualizer")

# Sidebar: folder selection and controls
st.sidebar.header("Data Selection & Controls")
folder_path = st.sidebar.text_input("📁 Folder containing NIfTI files", value="", placeholder="Enter path to folder")


def rgba_to_hex(rgba):
    """Convert an RGBA tuple (0-1 range) to a hex color string."""
    r, g, b, a = rgba
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


def load_structure(file_path):
    """
    Helper function to load a structure mask file.
    Converts the mask to boolean (i.e. >0) and applies the initial flip.
    """
    try:
        obj = nib.load(str(file_path))
        # Convert to bool (1-bit) mask
        data = (np.asanyarray(obj.dataobj) > 0)
        data = data[:, ::-1, ::-1]
        spacing = obj.header.get_zooms()[:3]
        name = file_path.stem
        return name[:-4], data, spacing
    except Exception as e:
        return None


@st.cache_data
def load_nifti_data(folder):
    """
    Load the mandatory image and optional dose and structure masks.
    Applies the initial flip transformation: volume = volume[:, ::-1, ::-1].
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

    # Load structure masks in parallel
    structures = {}
    structure_files = [f for f in folder.glob("*.nii.gz") if f.name not in ["image.nii.gz", "dose.nii.gz"]]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(load_structure, structure_files))
    for result in results:
        if result is not None:
            name, data, spacing = result
            structures[name] = {"data": data, "spacing": spacing}
    return image_data, image_spacing, dose_data, dose_spacing, structures


# --- Pre-transformation functions for each view ---
def transform_volume_for_view(vol, view):
    """
    Apply a combined 3D transformation to pre-orient a volume for a given view.
    Assumes vol has already been flipped (vol[:, ::-1, ::-1]).
    """
    if view == "axial":
        return np.flip(np.rot90(vol, k=1, axes=(0,1)), axis=0)
    elif view == "sagittal":
        return np.rot90(vol, k=1, axes=(1,2))
    elif view == "coronal":
        return np.fliplr(np.rot90(vol, k=1, axes=(0,2)))
    else:
        return vol


def crop_to_union(image_vol, dose_vol, structures):
    """
    Crop volumes to the bounding box covering the union of all structure masks.
    """
    if not structures:
        return image_vol, dose_vol, structures, None
    union_mask = np.zeros_like(image_vol, dtype=bool)
    for struct in structures.values():
        union_mask |= struct["data"]
    if not union_mask.any():
        return image_vol, dose_vol, structures, None
    coords = np.array(np.nonzero(union_mask))
    min_indices = coords.min(axis=1)
    max_indices = coords.max(axis=1) + 1
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
    If the x and y resolutions are already at the target, only interpolate along z.
    """
    zoom_factors = [orig / target_spacing for orig in original_spacing]
    if np.allclose(zoom_factors[0:2], [1, 1], atol=1e-3):
        zoom_factors[0] = 1.0
        zoom_factors[1] = 1.0
    return zoom(volume, zoom=zoom_factors, order=order)


@st.cache_data(show_spinner=False)
def preprocess_data(folder_path, target_resolution):
    """
    Load, crop, resample, and pre-transform volumes.
    Returns:
      image_views, dose_views, structure_views, target_spacing
    """
    image_vol, image_spacing, dose_vol, dose_spacing, structures = load_nifti_data(folder_path)
    image_vol, dose_vol, structures, _ = crop_to_union(image_vol, dose_vol, structures)

    # Use the user-selected target resolution.
    target_spacing = target_resolution

    # Resample volumes
    image_resampled = resample_volume(image_vol, image_spacing, target_spacing, order=0)
    if dose_vol is not None and dose_spacing is not None:
        dose_resampled = resample_volume(dose_vol, dose_spacing, target_spacing, order=0)
    else:
        dose_resampled = None
    structures_resampled = {}
    for name, struct in structures.items():
        # Resample and convert to boolean (1-bit)
        resampled = resample_volume(struct["data"], struct["spacing"], target_spacing, order=0)
        structures_resampled[name] = {
            "data": resampled.astype(bool),
            "spacing": (target_spacing, target_spacing, target_spacing)
        }

    # Pre-transform for each view
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


def composite_slice_to_pil(base_slice, window_min, window_max,
                           dose_slice=None, dose_params=None,
                           structure_slices=None, structure_colors=None,
                           cross_lines=False, cross_coords=(None, None)):
    """
    Render a composite slice (base image with optional overlays) as a PIL image.
    Optionally draws dashed yellow cross-lines at cross_coords=(x,y).
    """
    height, width = base_slice.shape
    dpi = 100
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(base_slice, cmap="gray", origin="lower",
              vmin=window_min, vmax=window_max, interpolation="nearest")
    if dose_slice is not None and dose_params is not None:
        dose_low, dose_high = dose_params
        dose_masked = np.ma.masked_less(dose_slice, dose_low)
        ax.imshow(dose_masked, cmap="jet", origin="lower", alpha=0.5,
                  vmin=dose_low, vmax=dose_high, interpolation="nearest")
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


# =========================
# Sidebar: Load and Adjust Parameters
# =========================
if folder_path:
    try:
        _, image_spacing, _, _, _ = load_nifti_data(folder_path)
        min_spacing = float(min(image_spacing))
    except Exception as e:
        st.sidebar.error(f"Error loading metadata: {e}")
        st.stop()

    # Resolution slider: from min_spacing to 3.0 mm, default is 2mm (or min_spacing if min_spacing > 2)
    default_resolution = 2.0 if min_spacing <= 2.0 else min_spacing
    resolution_slider = st.sidebar.slider("Target Resolution (mm)",
                                            min_value=float(min_spacing),
                                            max_value=3.0,
                                            value=float(default_resolution),
                                            step=0.1)

# =========================
# Main: Preprocess Data and Display Views
# =========================
if folder_path:
    try:
        image_views, dose_views, structure_views, target_spacing = preprocess_data(folder_path, resolution_slider)
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
    default_window = (max(img_min, default_window[0]),
                      min(img_max, default_window[1]))

    st.sidebar.write(f"Modality: {modality}")
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
            "Dose Overlay Window [Gy]",
            min_value=dose_min_val, max_value=dose_max_val,
            value=default_dose_window,
            step=0.01 * dose_max_val
        )
        show_dose = st.sidebar.checkbox("Show Dose Overlay", value=True)
        dose_params = (dose_low, dose_high)
    else:
        show_dose = False
        dose_params = None

    # Single "Show/Hide All Structures" button
    toggle_button = st.sidebar.button("Show/Hide All Structures")

    # Prepare structure checkboxes with default True (all structures shown)
    structure_keys = sorted(structure_views.keys())
    mask_visibility = {}
    structure_colors = {}
    # Determine current state: if all structures are enabled
    all_on = all(st.session_state.get(f"structure_{key}", True) for key in structure_keys)
    if toggle_button:
        new_state = not all_on
        for key in structure_keys:
            st.session_state[f"structure_{key}"] = new_state

    for idx, key in enumerate(structure_keys):
        color_rgba = plt.cm.tab10(idx % 10)
        hex_color = rgba_to_hex(color_rgba)
        structure_colors[key] = hex_color
        default_val = st.session_state.get(f"structure_{key}", True)
        col_cb, col_label = st.sidebar.columns([0.15, 0.85])
        with col_cb:
            mask_visibility[key] = st.checkbox("", value=default_val, key=f"structure_{key}")
        with col_label:
            st.markdown(f"<span style='color:{hex_color}; font-weight:bold'>{key}</span>",
                        unsafe_allow_html=True)

    # =========================
    # Display Views
    # =========================
    col_axial, col_sagittal, col_coronal = st.columns(3)
    nx_axial, ny_axial, nz_axial = image_views["axial"].shape
    nx_sag, ny_sag, nz_sag = image_views["sagittal"].shape
    nx_cor, ny_cor, nz_cor = image_views["coronal"].shape

    # Axial: slice along third dimension.
    with col_axial:
        st.subheader("Axial")
        axial_index = st.slider("Axial Slice", 0, nz_axial - 1, nz_axial // 2, key="axial_slider")
        base_slice = image_views["axial"][:, :, axial_index]
        dose_slice = None
        if show_dose and dose_views is not None:
            dose_slice = dose_views["axial"][:, :, axial_index]
        sagittal_index = st.session_state.get("sagittal_slider", nx_sag // 2)
        coronal_index = st.session_state.get("coronal_slider", ny_cor // 2)
        cross_coords_axial = (sagittal_index, ny_cor - coronal_index)
        struct_slices = {}
        for key in structure_keys:
            if st.session_state.get(f"structure_{key}", True):
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
        axial_index_for_sag = st.session_state.get("axial_slider", nz_axial // 2)
        coronal_index_for_sag = st.session_state.get("coronal_slider", ny_cor // 2)
        cross_coords_sagittal = (ny_cor - coronal_index_for_sag, nz_axial - axial_index_for_sag)
        struct_slices = {}
        for key in structure_keys:
            if st.session_state.get(f"structure_{key}", True):
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
        sagittal_index_for_cor = st.session_state.get("sagittal_slider", nx_sag // 2)
        axial_index_for_cor = st.session_state.get("axial_slider", nz_axial // 2)
        cross_coords_coronal = (sagittal_index_for_cor, nz_axial - axial_index_for_cor)
        struct_slices = {}
        for key in structure_keys:
            if st.session_state.get(f"structure_{key}", True):
                struct_slices[key] = structure_views[key]["coronal"][:, coronal_index, :]
        pil_img = composite_slice_to_pil(base_slice, window_min, window_max,
                                         dose_slice=dose_slice, dose_params=dose_params,
                                         structure_slices=struct_slices, structure_colors=structure_colors,
                                         cross_lines=True, cross_coords=cross_coords_coronal)
        st.image(pil_img, use_column_width=False)
