# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os

from scripts.graphfrmpts import equation_plane, midpoint, normal_line_to_plane, frustum_equation, point_in_frustum

# os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_DEVICE_ID"] = "0"

import sys
from argparse import ArgumentParser
import random
import pickle as pkl
import numpy as np
from PIL import Image, ImageOps
import torch
from tqdm import tqdm
import time

from utils import (
    normalize_rgb,
    render_meshes,
    get_focalLength_from_fieldOfView,
    demo_color as color,
    print_distance_on_image,
    render_side_views,
    create_scene,
    MEAN_PARAMS,
    CACHE_DIR_MULTIHMR,
    SMPLX_DIR,
)
from model import Model
from pathlib import Path
import warnings

torch.cuda.empty_cache()

np.random.seed(seed=0)
random.seed(0)


def open_image(img_path, img_size, device=torch.device("cuda")):
    """Open image at path, resize and pad"""

    # Open and reshape
    img_pil = Image.open(img_path).convert("RGB")
    img_pil = ImageOps.contain(
        img_pil, (img_size, img_size)
    )  # keep the same aspect ratio

    # Keep a copy for visualisations.
    img_pil_bis = ImageOps.pad(
        img_pil.copy(), size=(img_size, img_size), color=(255, 255, 255)
    )
    img_pil = ImageOps.pad(
        img_pil, size=(img_size, img_size)
    )  # pad with zero on the smallest side

    # Go to numpy
    resize_img = np.asarray(img_pil)

    # Normalize and go to torch.
    resize_img = normalize_rgb(resize_img)
    x = torch.from_numpy(resize_img).unsqueeze(0).to(device)
    return x, img_pil_bis


def get_camera_parameters(
    img_size, fov=60, p_x=None, p_y=None, device=torch.device("cuda")
):
    """Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = torch.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0, 0], K[1, 1] = focal, focal

    # Set principal point
    if p_x is not None and p_y is not None:
        K[0, -1], K[1, -1] = p_x * img_size, p_y * img_size
    else:
        K[0, -1], K[1, -1] = img_size // 2, img_size // 2

    # Add batch dimension
    K = K.unsqueeze(0).to(device)
    return K


def load_model(model_name, device=torch.device("cuda")):
    """Open a checkpoint, build Multi-HMR using saved arguments, load the model weigths."""
    # Model
    ckpt_path = os.path.join(CACHE_DIR_MULTIHMR, model_name + ".pt")
    if not os.path.isfile(ckpt_path):
        os.makedirs(CACHE_DIR_MULTIHMR, exist_ok=True)
        print(f"{ckpt_path} not found...")
        print("It should be the first time you run the demo code")
        print("Downloading checkpoint from NAVER LABS Europe website...")

        try:
            os.system(
                f"wget -O {ckpt_path} https://download.europe.naverlabs.com/ComputerVision/MultiHMR/{model_name}.pt"
            )
            print(f"Ckpt downloaded to {ckpt_path}")
        except:
            assert "Please contact fabien.baradel@naverlabs.com or open an issue on the github repo"

    # Load weights
    print("Loading model")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Get arguments saved in the checkpoint to rebuild the model
    kwargs = {}
    for k, v in vars(ckpt["args"]).items():
        kwargs[k] = v

    # Build the model.
    kwargs["type"] = ckpt["args"].train_return_type
    kwargs["img_size"] = ckpt["args"].img_size[0]
    model = Model(**kwargs).to(device)

    # Load weights into model.
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("Weights have been loaded")

    return model


def forward_model(
    model,
    input_image,
    camera_parameters,
    det_thresh=0.3,
    nms_kernel_size=1,
):
    """Make a forward pass on an input image and camera parameters."""

    # Forward the model.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            humans = model(
                input_image,
                is_training=False,
                nms_kernel_size=int(nms_kernel_size),
                det_thresh=det_thresh,
                K=camera_parameters,
            )

    return humans


def overlay_human_meshes(humans, K, model, img_pil, unique_color=False):

    # Color of humans seen in the image.
    _color = [color[0] for _ in range(len(humans))] if unique_color else color

    # Get focal and princpt for rendering.
    focal = np.asarray([K[0, 0, 0].cpu().numpy(), K[0, 1, 1].cpu().numpy()])
    princpt = np.asarray([K[0, 0, -1].cpu().numpy(), K[0, 1, -1].cpu().numpy()])

    # Get the vertices produced by the model.
    verts_list = [humans[j]["verts_smplx"].cpu().numpy() for j in range(len(humans))]

    # save verts to file
    l_eye1 = list(verts_list[0][9504 - 1].tolist())
    r_eye1 = list(verts_list[0][10050 - 1].tolist())
    chin1 = list(verts_list[0][8765 - 1].tolist())
    midpoint1 = midpoint(l_eye1, r_eye1, chin1)
    plane1 = equation_plane(*l_eye1, *r_eye1, *chin1)
    vline1 = normal_line_to_plane(*plane1, *midpoint1)
    frustum = frustum_equation(*plane1, l_eye1, r_eye1, chin1, 0.1, 0.1)

    if len(verts_list) > 1:
        l_eye2 = list(verts_list[1][9504 - 1].tolist())
        r_eye2 = list(verts_list[1][10050 - 1].tolist())
        chin2 = list(verts_list[1][8765 - 1].tolist())
        midpoint2 = midpoint(l_eye2, r_eye2, chin2)
        plane2 = equation_plane(*l_eye2, *r_eye2, *chin2)
        vline2 = normal_line_to_plane(*plane2, *midpoint2)
        point_frustum = point_in_frustum(*plane1, l_eye1, r_eye1, chin1, 0.1, 0.05, midpoint2)

    with open("verts.txt", "w") as f:
        f.write(f"l_eye1: {l_eye1}\n")
        f.write(f"r_eye1: {r_eye1}\n")
        f.write(f"chin1: {chin1}\n")
        f.write(
            f"Equation of the plane is: {plane1[0]}x + {plane1[1]}y + {plane1[2]}z + {plane1[3]} = 0\n"
        )
        f.write(f"midpoint1: {midpoint1}\n")
        f.write(f"vline1: {vline1}\n")
        if len(verts_list) > 1:
            f.write(f"l_eye2: {l_eye2}\n")
            f.write(f"r_eye2: {r_eye2}\n")
            f.write(f"chin2: {chin2}\n")
            f.write(
                f"Equation of the plane is: {plane2[0]}x + {plane2[1]}y + {plane2[2]}z + {plane2[3]} = 0\n"
            )
            f.write(f"midpoint2: {midpoint2}\n")
            f.write(f"vline2: {vline2}\n")
        f.write(f"frustum: {frustum}\n")
        try:
            f.write(f"EYE CONTACT: {point_frustum}\n")
        except:
            print("NO PERSON 2 IN IMAGE!")
    with open("verts1.txt", "w") as f:
        for vert in verts_list[0]:
            f.write(f"{list(vert)}\n")
    if len(verts_list) > 1:
        with open("verts2.txt", "w") as f:
            for vert in verts_list[1]:
                f.write(f"{list(vert)}\n")
    faces_list = [model.smpl_layer["neutral"].bm_x.faces for j in range(len(humans))]
    print(faces_list)

    # Render the meshes onto the image.
    pred_rend_array = render_meshes(
        np.asarray(img_pil),
        verts_list,
        faces_list,
        {"focal": focal, "princpt": princpt},
        alpha=1.0,
        color=_color,
    )

    return pred_rend_array, _color


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="multiHMR_896_L_synth")
    parser.add_argument("--img_path", type=str, default="example_data/images.jpg")
    parser.add_argument("--out_folder", type=str, default="demo_out")
    parser.add_argument("--save_mesh", type=int, default=0, choices=[0, 1])
    parser.add_argument("--extra_views", type=int, default=0, choices=[0, 1])
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--nms_kernel_size", type=float, default=3)
    parser.add_argument("--fov", type=float, default=60)
    parser.add_argument(
        "--distance",
        type=int,
        default=0,
        choices=[0, 1],
        help="add distance on the reprojected mesh",
    )
    parser.add_argument(
        "--unique_color",
        type=int,
        default=0,
        choices=[0, 1],
        help="only one color for all humans",
    )

    args = parser.parse_args()

    dict_args = vars(args)

    assert torch.cuda.is_available()

    # SMPL-X models
    smplx_fn = os.path.join(SMPLX_DIR, "smplx", "SMPLX_NEUTRAL.npz")
    if not os.path.isfile(smplx_fn):
        print(f"{smplx_fn} not found, please download SMPLX_NEUTRAL.npz file")
        print("To do so you need to create an account in https://smpl-x.is.tue.mpg.de")
        print(
            "Then download 'SMPL-X-v1.1 (NPZ+PKL, 830MB) - Use thsi for SMPL-X Python codebase'"
        )
        print(f"Extract the zip file and move SMPLX_NEUTRAL.npz to {smplx_fn}")
        print(
            "Sorry for this incovenience but we do not have license for redustributing SMPLX model"
        )
        assert NotImplementedError
    else:
        print("SMPLX found")

    # SMPL mean params download
    if not os.path.isfile(MEAN_PARAMS):
        print("Start to download the SMPL mean params")
        os.system(
            f"wget -O {MEAN_PARAMS}  https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4"
        )
        print("SMPL mean params have been succesfully downloaded")
    else:
        print("SMPL mean params is already here")

    # Input images
    suffixes = (".jpg", ".jpeg", ".png", ".webp")
    l_img_path = [args.img_path]

    # Loading
    model = load_model(args.model_name)

    # Model name for saving results.
    model_name = os.path.basename(args.model_name)

    # All images
    os.makedirs(args.out_folder, exist_ok=True)
    l_duration = []
    for i, img_path in enumerate(tqdm(l_img_path)):

        # Path where the image + overlays of human meshes + optional views will be saved.
        save_fn = os.path.join(
            args.out_folder, f"{Path(img_path).stem}_{model_name}.png"
        )

        # Get input in the right format for the model
        img_size = model.img_size
        x, img_pil_nopad = open_image(img_path, img_size)

        # Get camera parameters
        p_x, p_y = None, None
        K = get_camera_parameters(model.img_size, fov=args.fov, p_x=p_x, p_y=p_y)

        # Make model predictions
        start = time.time()
        humans = forward_model(
            model,
            x,
            K,
            det_thresh=args.det_thresh,
            nms_kernel_size=args.nms_kernel_size,
        )
        duration = time.time() - start
        l_duration.append(duration)

        # Superimpose predicted human meshes to the input image.
        img_array = np.asarray(img_pil_nopad)
        img_pil_visu = Image.fromarray(img_array)
        pred_rend_array, _color = overlay_human_meshes(
            humans, K, model, img_pil_visu, unique_color=args.unique_color
        )

        # Optionally add distance as an annotation to each mesh
        if args.distance:
            pred_rend_array = print_distance_on_image(pred_rend_array, humans, _color)

        # List of images too view side by side.
        l_img = [img_array, pred_rend_array]

        # More views
        if args.extra_views:
            # Render more side views of the meshes.
            pred_rend_array_bis, pred_rend_array_sideview, pred_rend_array_bev = (
                render_side_views(img_array, _color, humans, model, K)
            )

            # Concat
            _img1 = np.concatenate([img_array, pred_rend_array], 1).astype(np.uint8)
            _img2 = np.concatenate(
                [pred_rend_array_bis, pred_rend_array_sideview, pred_rend_array_bev], 1
            ).astype(np.uint8)
            _h = int(_img2.shape[0] * (_img1.shape[1] / _img2.shape[1]))
            _img2 = np.asarray(Image.fromarray(_img2).resize((_img1.shape[1], _h)))
            _img = np.concatenate([_img1, _img2], 0).astype(np.uint8)
        else:
            # Concatenate side by side
            _img = np.concatenate([img_array, pred_rend_array], 1).astype(np.uint8)

        # Save to path.
        Image.fromarray(_img).save(save_fn)
        print(
            f"Avg Multi-HMR inference time={int(1000*np.median(np.asarray(l_duration[-1:])))}ms on a {torch.cuda.get_device_name()}"
        )

        # Saving mesh
        if args.save_mesh:
            # npy file
            l_mesh = [hum["verts_smplx"].cpu().numpy() for hum in humans]

            with open(save_fn + ".pkl", "wb") as f:
                pkl.dump(l_mesh, f)
            mesh_fn = save_fn + ".npy"
            np.save(mesh_fn, np.asarray(l_mesh), allow_pickle=True)
            x = np.load(mesh_fn, allow_pickle=True)

            # glb file
            l_mesh = [
                humans[j]["verts_smplx"].detach().cpu().numpy()
                for j in range(len(humans))
            ]
            l_face = [
                model.smpl_layer["neutral"].bm_x.faces for j in range(len(humans))
            ]
            scene = create_scene(
                img_pil_visu,
                l_mesh,
                l_face,
                color=None,
                metallicFactor=0.0,
                roughnessFactor=0.5,
            )
            scene_fn = save_fn + ".glb"
            scene.export(scene_fn)

    print("end")
