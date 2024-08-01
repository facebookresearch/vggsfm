import os
import numpy as np
from PIL import Image
import torch
from multiprocessing import Pool
from pytorch3d.renderer import PerspectiveCameras


def process_scene(scene, scene_info_dir, train_dir, min_num_images):
    scene_name = "re10k" + scene
    print(scene_name)
    scene_info_name = os.path.join(scene_info_dir, os.path.basename(scene) + ".txt")
    scene_info = np.loadtxt(scene_info_name, delimiter=" ", dtype=np.float64, skiprows=1)

    filtered_data = []

    for frame_idx in range(len(scene_info)):
        try:
            raw_line = scene_info[frame_idx]
            timestamp = raw_line[0]
            intrinsics = raw_line[1:7]
            extrinsics = raw_line[7:]

            imgpath = os.path.join(train_dir, scene, f"{int(timestamp)}.png")
            image_size = Image.open(imgpath).size
            Posemat = extrinsics.reshape(3, 4).astype("float64")
            focal_length = intrinsics[:2] * image_size
            principal_point = intrinsics[2:4] * image_size

            # if True:
            #     Posemat_full =  np.pad(Posemat, ((0, 1), (0, 0)), 'constant', constant_values=0)
            #     Posemat_full[-1,-1] = 1
            #     Posemat_full_inv =  np.linalg.inv(Posemat_full)

            #     rot = Posemat_full_inv[:3, :3]
            #     trans = Posemat_full_inv[:3, -1]
            # else:
            rot = Posemat[:3, :3]
            trans = Posemat[:3, -1]

            data = {
                "filepath": imgpath,
                # "depth_filepath": None,
                # "R": torch.from_numpy(Posemat[:3, :3]),
                # "T": torch.from_numpy(Posemat[:3, -1]),
                "R": rot,
                "T": trans,
                "focal_length": focal_length,
                "principal_point": principal_point,
            }

            filtered_data.append(data)
        except:
            # print("one image is missing")
            haha = 1

    if len(filtered_data) > min_num_images:
        return scene_name, filtered_data
    else:
        print(f"scene {scene_name} does not have enough image nums")
        return scene_name, None


def main(scenes, scene_info_dir, train_dir, min_num_images):
    # scenes = scenes[:100]
    # num_processes = 10
    process_scene(scenes[0], scene_info_dir, train_dir, min_num_images)
    print(len(scenes))
    with Pool() as pool:
        results = pool.starmap(process_scene, [(scene, scene_info_dir, train_dir, min_num_images) for scene in scenes])

    wholedata = {scene_name: data for scene_name, data in results if data is not None}

    import pdb

    pdb.set_trace()
    filename = "/data/home/jianyuan/Re10K_anno/processed.pkl"
    with open(filename, "wb") as file:
        pickle.dump(wholedata, file)

    print("finished")
    return wholedata


# Example usage
if __name__ == "__main__":
    video_loc = "/fsx-repligen/shared/datasets/RealEstate10K/frames/train/video_loc.txt"
    scenes = np.loadtxt(video_loc, dtype=np.str_)
    scene_info_dir = "/data/home/jianyuan/Re10K_anno/RealEstate10K/train"
    train_dir = "/fsx-repligen/shared/datasets/RealEstate10K/frames/train"

    min_num_images = 24  # Example threshold
    wholedata = main(scenes, scene_info_dir, train_dir, min_num_images)
