import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

from utils import (
    normalize_rgb,
    render_meshes,
    get_focalLength_from_fieldOfView,
    demo_color as color,
    print_distance_on_image,
    MEAN_PARAMS,
    CACHE_DIR_MULTIHMR,
    SMPLX_DIR,
)
from model import Model

torch.cuda.empty_cache()

np.random.seed(seed=0)


def open_image_from_pil(img_pil, img_size, device=torch.device("cuda")):
    """Open image from PIL, resize and pad"""
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
    """Open a checkpoint, build Multi-HMR using saved arguments, load the model weights"""
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
            raise RuntimeError(
                "Please contact fabien.baradel@naverlabs.com or open an issue on the github repo"
            )

    # Load weights
    print("Loading model")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Get arguments saved in the checkpoint to rebuild the model
    kwargs = {}
    for k, v in vars(ckpt["args"]).items():
        kwargs[k] = v

    # Build the model
    kwargs["type"] = ckpt["args"].train_return_type
    kwargs["img_size"] = ckpt["args"].img_size[0]
    model = Model(**kwargs).to(device)

    # Load weights into model
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("Weights have been loaded")

    return model


def forward_model(
    model, input_image, camera_parameters, det_thresh=0.3, nms_kernel_size=1
):
    """Make a forward pass on an input image and camera parameters"""
    # Forward the model
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
    # Color of humans seen in the image
    _color = [color[0] for _ in range(len(humans))] if unique_color else color

    # Get focal and princpt for rendering
    focal = np.asarray([K[0, 0, 0].cpu().numpy(), K[0, 1, 1].cpu().numpy()])
    princpt = np.asarray([K[0, 0, -1].cpu().numpy(), K[0, 1, -1].cpu().numpy()])

    # Get the vertices produced by the model
    verts_list = [humans[j]["verts_smplx"].cpu().numpy() for j in range(len(humans))]
    faces_list = [model.smpl_layer["neutral"].bm_x.faces for j in range(len(humans))]
    # print(faces_list[0])

    # Render the meshes onto the image
    pred_rend_array = render_meshes(
        np.asarray(img_pil),
        verts_list,
        faces_list,
        {"focal": focal, "princpt": princpt},
        alpha=1.0,
        color=_color,
    )
    return pred_rend_array, _color


def process_video(
    video_path,
    output_folder,
    model,
    img_size,
    fov,
    det_thresh,
    nms_kernel_size,
    extra_views,
    distance,
    unique_color,
):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Capture video
    video_cap = cv2.VideoCapture(video_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    sample_rate = int(fps / 20)  # sampling every nth frame to get ~10 fps

    frames = []
    timestamps = []

    for frame_idx in tqdm(range(frame_count), desc="Extracting frames"):
        success, frame = video_cap.read()
        if not success:
            break
        if frame_idx % sample_rate == 0:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x, img_pil_nopad = open_image_from_pil(img_pil, img_size)
            K = get_camera_parameters(img_size, fov=fov)
            humans = forward_model(
                model, x, K, det_thresh=det_thresh, nms_kernel_size=nms_kernel_size
            )
            pred_rend_array, _color = overlay_human_meshes(
                humans, K, model, img_pil_nopad, unique_color=unique_color
            )
            if distance:
                pred_rend_array = print_distance_on_image(
                    pred_rend_array, humans, _color
                )
            frames.append(pred_rend_array)
            timestamps.append(frame_idx / fps)

    video_cap.release()

    # Save the frames back into a video
    output_video_path = os.path.join(
        output_folder, f"{Path(video_path).stem}_processed.mp4"
    )
    clip = ImageSequenceClip(frames, fps=10)
    clip = clip.set_duration(duration)  # maintain the original video length
    clip.write_videofile(output_video_path, codec="libx264")

    print(f"Processed video saved to {output_video_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="multiHMR_896_L_synth")
    parser.add_argument("--video_path", type=str, required=True)
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

    assert torch.cuda.is_available()

    # SMPL-X models
    smplx_fn = os.path.join(SMPLX_DIR, "smplx", "SMPLX_NEUTRAL.npz")
    if not os.path.isfile(smplx_fn):
        print(f"{smplx_fn} not found, please download SMPLX_NEUTRAL.npz file")
        print("To do so you need to create an account in https://smpl-x.is.tue.mpg.de")
        print(
            "Then download 'SMPL-X-v1.1 (NPZ+PKL, 830MB) - Use this for SMPL-X Python codebase'"
        )
        print(f"Extract the zip file and move SMPLX_NEUTRAL.npz to {smplx_fn}")
        print(
            "Sorry for this inconvenience but we do not have license for redistributing SMPLX model"
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
        print("SMPL mean params have been successfully downloaded")
    else:
        print("SMPL mean params is already here")

    # Loading
    model = load_model(args.model_name)

    # Process video
    process_video(
        args.video_path,
        args.out_folder,
        model,
        model.img_size,
        args.fov,
        args.det_thresh,
        args.nms_kernel_size,
        args.extra_views,
        args.distance,
        args.unique_color,
    )
