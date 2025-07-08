import os
from typing import List
import torch
from lightglue.utils import rbd
from lightglue import LightGlue, SuperPoint
import networkx as nx
from tqdm import tqdm
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
from PIL import Image
from torchvision import transforms as t

tf = t.Compose(
    [
        t.ToTensor(),  # converts to (C, H, W), [0,1]
    ]
)
import cv2
import numpy as np

import cv2
from tqdm import tqdm


def pad_to_equirectangular(image):
    h, w = image.shape[:2]
    target_height = w // 2

    if h == target_height:
        print("Image is already 2:1 equirectangular.")
        return image
    elif h > target_height:
        print(
            f"Resizing image from ({w}x{h}) to ({w}x{target_height}) to match 2:1 ratio."
        )
        resized_image = cv2.resize(
            image, (w, target_height), interpolation=cv2.INTER_AREA
        )
        return resized_image
    pad_total = target_height - h
    print(f"Padding: {pad_total} px to reach {target_height}px height.")
    padded_image = cv2.copyMakeBorder(
        image,
        pad_total,
        0,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),  # black padding
    )

    return padded_image


def read_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read {path}")
        images.append(img)
    return images


def compute_matches(images, match_conf=0.6):
    extractor = SuperPoint(max_num_keypoints=512).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    features, feats_np = [], []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = Image.fromarray(gray, mode="L")
        gray = tf(gray).unsqueeze(0).to(device)
        feat = extractor.extract(gray, device=device)
        features.append(feat)
        feats_np.append(gray)
    G = nx.Graph()
    for i in range(len(images)):
        G.add_node(i)  # no need for image_paths
    print("Computing pairwise matches using LightGlue with progress bar...")
    pairs = [(i, j) for i in range(len(images)) for j in range(i + 1, len(images))]
    for i, j in tqdm(pairs, desc="Matching pairs", unit="pair"):
        f0, f1 = features[i], features[j]
        match_dict = matcher({"image0": f0, "image1": f1})
        f0, f1, match_dict = [rbd(x) for x in [f0, f1, match_dict]]
        matches = match_dict["matches"].cpu().numpy()  # (K,2)
        if matches.shape[0] < 5:
            continue
        kpts0 = f0["keypoints"][matches[:, 0]].cpu().numpy()
        kpts1 = f1["keypoints"][matches[:, 1]].cpu().numpy()
        if len(kpts1) < 8:
            continue
        H, inliers = cv2.findHomography(kpts0, kpts1, cv2.RANSAC, 4.0)
        inlier_ratio = np.sum(inliers) / len(inliers) if inliers is not None else 0
        if H is not None and inlier_ratio > match_conf:
            conf = inlier_ratio
            G.add_edge(i, j, weight=-conf)
            tqdm.write(f"Match {i}-{j}: {len(kpts1)} matches, inlier ratio={conf:.2f}")
        del match_dict, f0, f1, matches
    del features, feats_np, matcher, extractor
    return G


def compute_mst_order(G, start):
    mst = nx.minimum_spanning_tree(G, weight="weight")
    order = list(nx.dfs_preorder_nodes(mst, source=start))
    print("MST-based order:", order)
    return order


def generate_panorama_images(images):
    if len(images) < 2:
        print("Need at least two images to stitch a panorama.")
        return None
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    try:
        cv2.ocl.setUseOpenCL(False)
        stitcher.setWaveCorrection(True)
        stitcher.setRegistrationResol(0.6)
        stitcher.setSeamEstimationResol(0.6)
        stitcher.setInterpolationFlags(cv2.INTER_CUBIC)
    except Exception as e:
        print(f"Warning: could not set advanced options: {e}")

    status, pano = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        print(f"Error during stitching: {status}")
        return None
    return pano


def fill_black_with_inpainting(image):
    mask = np.all(image == 0, axis=2).astype(np.uint8) * 255
    impainted = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return impainted


def generate_panorama_image_from_path(image_paths: List[str]):
    images = read_images(image_paths)
    graph = compute_matches(images, match_conf=0.6)
    degrees = dict(graph.degree())
    start = max(degrees, key=degrees.get)
    mst_order = compute_mst_order(graph, start)
    panorama = generate_panorama_images([images[i] for i in mst_order])
    if panorama is None:
        return None
    panorama = fill_black_with_inpainting(panorama)
    panorama = pad_to_equirectangular(panorama)
    return fill_black_with_inpainting(panorama)


def file_has_moov_atom(path, check_size=10 * 1024 * 1024):
    """Quick scan for 'moov' atom in first and last few MB of the file."""
    try:
        with open(path, "rb") as f:
            head = f.read(check_size)
            f.seek(-check_size, os.SEEK_END)
            tail = f.read(check_size)
        return b"moov" in head or b"moov" in tail
    except Exception as e:
        print(f"Error while checking moov atom: {e}")
        return False


def extract_and_select_frames(
    video_path,
    min_movement=5,
    max_movement=50,
    step=5,
    allowed_extensions={".mp4", ".mov", ".avi", ".mkv", ".webm"},
):
    # 1. Check file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 2. Validate extension
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in allowed_extensions:
        raise ValueError(
            f"Unsupported video format '{ext}'. Allowed: {allowed_extensions}"
        )

    # 3. Validate moov atom (only for .mp4/.mov files)
    if ext in {".mp4", ".mov"} and not file_has_moov_atom(video_path):
        raise ValueError("Invalid or incomplete MP4/MOV file: 'moov' atom not found.")

    # 4. Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            "Failed to open video. The file might be corrupted or unsupported."
        )

    # 5. Validate basic properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0 or width == 0 or height == 0:
        cap.release()
        raise ValueError("Invalid video file: zero frame count or resolution.")

    # 6. Validate first frame
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        cap.release()
        raise RuntimeError(
            "Could not read the first frame. The file might be corrupted."
        )
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to first frame

    # --- Proceed with frame extraction ---
    selected_frames = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.3, 3, 15, 3, 8, 1.5, 0
            )
            movement = np.linalg.norm(flow, axis=2).mean()
            print(f"Frame {frame_idx}: avg movement={movement:.2f}")
            if min_movement < movement < max_movement:
                selected_frames.append(frame)
        else:
            selected_frames.append(frame)

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return selected_frames


def generate_panorama_image_from_video(video_path: str):
    images = extract_and_select_frames(video_path)
    gc.collect()
    graph = compute_matches(images, match_conf=0.6)
    degrees = dict(graph.degree())
    start = max(degrees, key=degrees.get)
    mst_order = compute_mst_order(graph, start)
    panorama = generate_panorama_images([images[i] for i in mst_order])
    if panorama is None:
        return None
    panorama = fill_black_with_inpainting(panorama)
    panorama = pad_to_equirectangular(panorama)
    return fill_black_with_inpainting(panorama)
