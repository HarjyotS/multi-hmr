import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

from scripts.graphfrmpts import (
    equation_plane,
    frustum_equation,
    midpoint,
    normal_line_to_plane,
    point_in_frustum,
)
from utils import (
    normalize_rgb,
    render_meshes,
    get_focalLength_from_fieldOfView,
    demo_color as color,
    print_distance_on_image,
    MEAN_PARAMS,
    CACHE_DIR_MULTIHMR,
    SMPLX_DIR,
    create_scene,
)
from model import Model
from utils.render import print_eye_contact, print_orientation

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
    # set color

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
        alpha=0.45,
        color=_color,
    )

    l_eye1 = list(verts_list[0][9504 - 1].tolist())
    r_eye1 = list(verts_list[0][10050 - 1].tolist())
    chin1 = list(verts_list[0][8765 - 1].tolist())
    midpoint1 = midpoint(l_eye1, r_eye1, chin1)
    plane1 = equation_plane(*l_eye1, *r_eye1, *chin1)
    vline1 = normal_line_to_plane(*plane1, *midpoint1)

    if len(verts_list) > 1:

        # code to which place the first person is looking
        nose1 = list(verts_list[0][2922 - 1].tolist())
        back_of_head1 = list(verts_list[0][8981 - 1].tolist())
        l_wrist1 = list(verts_list[0][4628 - 1].tolist())
        w_wrist1 = list(verts_list[0][7593 - 1].tolist())
        nose2 = list(verts_list[1][2922 - 1].tolist())
        back_of_head2 = list(verts_list[1][8981 - 1].tolist())
        l_wrist2 = list(verts_list[1][4628 - 1].tolist())
        w_wrist2 = list(verts_list[1][7593 - 1].tolist())
        # if z of nose smaller than z of back of head then facing towards camera
        # then check if right hand x is larger than left hand x
        # if z of nose is larger than z of head, then check if left hand x is larger than right hand x to check if arms crossed
        if nose1[2] < back_of_head1[2]:
            if w_wrist1[0] > l_wrist1[0]:
                print("arms crossed")
                pred_rend_array = print_orientation(pred_rend_array, "crossed", 0)
            else:
                print("arms not crossed")
                pred_rend_array = print_orientation(pred_rend_array, "not crossed", 0)
        else:
            if w_wrist1[0] < l_wrist1[0]:
                print("not crossed")
                pred_rend_array = print_orientation(pred_rend_array, "not crossed", 0)
            else:
                print("arms crossed")
                pred_rend_array = print_orientation(pred_rend_array, "crossed", 0)
        if nose2[2] < back_of_head2[2]:
            if w_wrist2[0] > l_wrist2[0]:
                print("arms crossed")
                pred_rend_array = print_orientation(pred_rend_array, "crossed", 1)
            else:
                print("arms not crossed")
                pred_rend_array = print_orientation(pred_rend_array, "not crossed", 1)
        else:
            if w_wrist2[0] < l_wrist2[0]:
                print("not crossed")
                pred_rend_array = print_orientation(pred_rend_array, "not crossed", 1)
            else:
                print("arms crossed")
                pred_rend_array = print_orientation(pred_rend_array, "crossed", 0)
        print("More than one human detected")
        l_eye2 = list(verts_list[1][9504 - 1].tolist())
        r_eye2 = list(verts_list[1][10050 - 1].tolist())
        chin2 = list(verts_list[1][8765 - 1].tolist())
        midpoint2 = midpoint(l_eye2, r_eye2, chin2)
        plane2 = equation_plane(*l_eye2, *r_eye2, *chin2)
        vline2 = normal_line_to_plane(*plane2, *midpoint2)
        frustum = frustum_equation(*plane2, l_eye2, r_eye2, chin2, 0.3, 0.1)

        point_frustum = point_in_frustum(
            *plane2, l_eye2, r_eye2, chin2, 0.3, 0.1, midpoint1
        )
        print("eye contact", point_frustum)

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

        pred_rend_array = print_eye_contact(pred_rend_array, point_frustum)
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
    save_mesh,
):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create mesh output folder
    mesh_output_folder = os.path.join(output_folder, "meshes")
    os.makedirs(mesh_output_folder, exist_ok=True)

    # Capture video
    video_cap = cv2.VideoCapture(video_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    sample_rate = int(fps / 10)  # sampling every nth frame to get ~10 fps

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

            if save_mesh:
                # Save the mesh for the current frame
                l_mesh = [
                    humans[j]["verts_smplx"].detach().cpu().numpy()
                    for j in range(len(humans))
                ]
                l_face = [
                    model.smpl_layer["neutral"].bm_x.faces for j in range(len(humans))
                ]
                scene = create_scene(
                    img_pil_nopad,
                    l_mesh,
                    l_face,
                    color=None,
                    metallicFactor=0.0,
                    roughnessFactor=0.5,
                )
                mesh_filename = os.path.join(
                    mesh_output_folder, f"frame_{frame_idx:04d}.glb"
                )
                scene.export(mesh_filename)

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
        args.save_mesh,
    )
