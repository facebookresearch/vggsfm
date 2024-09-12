import copy
import pycolmap
import numpy as np

try:
    from scipy.spatial.transform import Rotation as R, Slerp
    from scipy.interpolate import interp1d
except:
    print("Scipy not found; Please disable pose interpolation")

def interpolate_transformations(rotation_matrices, translation_vectors, timesteps, new_timesteps):
    """
    Interpolate rotation matrices and translation vectors.

    Parameters:
    rotation_matrices (list of np.ndarray): List of N 3x3 rotation matrices.
    translation_vectors (list of np.ndarray): List of N 3D translation vectors.
    timesteps (list of float): List of N timesteps corresponding to the rotations and translations.
    new_timesteps (list of float): List of M new timesteps to interpolate.

    Returns:
    tuple: Interpolated rotation matrices and translation vectors at the new timesteps.
    """
    
    # Step 1: Convert rotation matrices to Rotation objects
    rotations = R.from_matrix(rotation_matrices)
    
    # Step 2: Create a Slerp object with the known rotations and timesteps
    slerp = Slerp(timesteps, rotations)
    
    # Step 3: Interpolate the rotations for the new timesteps
    interpolated_rotations = slerp(new_timesteps)
    
    # Step 4: Convert interpolated rotations to rotation matrices
    interpolated_matrices = interpolated_rotations.as_matrix()
    
    # Step 5: Interpolate the translation vectors
    translation_interp = interp1d(timesteps, translation_vectors, axis=0, kind='linear')
    interpolated_translations = translation_interp(new_timesteps)
    
    return interpolated_matrices, interpolated_translations


def interpolate_pycolmap(rec):
    rec = copy.deepcopy(rec)
    fname_to_id = {}
    for image_id in rec.images:
        fname_to_id[rec.images[image_id].name]=image_id
    frame_names= sorted(list(fname_to_id.keys()))
    
    timesteps = []
    intris = []
    extris = []
    cam_ids = []
    for fname in frame_names:
        timestep = int(fname[-8:-4])
        pyimg = rec.images[fname_to_id[fname]]
        pycam = rec.cameras[pyimg.camera_id]
        cam_ids.append(pyimg.camera_id)
        timesteps.append(timestep)
        intris.append(pycam.calibration_matrix())
        extris.append(pyimg.cam_from_world.matrix())
    intris = np.array(intris)
    extris = np.array(extris)
    
    interp_timesteps = np.arange(min(timesteps), max(timesteps) + 1)


    all_interped_rot, all_interped_trans = interpolate_transformations(extris[:,:,:3], extris[:,:,-1], 
                                                            timesteps, interp_timesteps)
    

    interp_timesteps = np.setdiff1d(interp_timesteps, timesteps)


    interped_rot, interped_trans = interpolate_transformations(extris[:,:,:3], extris[:,:,-1], 
                                                            timesteps, interp_timesteps)

    bins = np.digitize(interp_timesteps, timesteps) - 1

    max_image_id = max(fname_to_id.values())
    
    # save interped images back to the reconstruction object
    for idx, new_timestep in enumerate(interp_timesteps): 
        image_filename = f"image_{new_timestep:04d}.png"        
        cam_id = cam_ids[bins[idx]]
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(interped_rot[idx].astype(np.float64)), interped_trans[idx]) 

        pyimage = pycolmap.Image(id=max_image_id+idx+1, name=image_filename, camera_id=cam_id, cam_from_world=cam_from_world)
        pyimage.registered = True
        rec.add_image(pyimage)
        
    return rec, all_interped_rot, all_interped_trans





def compute_sparse_depth_for_interped_rec(reconstruction):
    from collections import defaultdict

    sparse_depth = defaultdict(list)
    sparse_point = defaultdict(list)

    point3D_idxes = [point3D_idx for point3D_idx in reconstruction.points3D]
    point3D_idxes = sorted(point3D_idxes)
    
    points3D_xyz = [reconstruction.points3D[point3D_idx].xyz for point3D_idx in point3D_idxes]
    points3D_xyz = np.array(points3D_xyz)
    point3D_idxes = np.array(point3D_idxes)



    for image_id in reconstruction.images: 
        pyimg = reconstruction.images[image_id]
        pycam = reconstruction.cameras[pyimg.camera_id]
        img_name = pyimg.name
        
        projection = pyimg.cam_from_world * points3D_xyz
        depth = projection[:, -1]
        uv = pycam.img_from_cam(projection)

        valid_depth = depth>=0.01
    
        sparse_depth[img_name] = (np.hstack((uv[valid_depth], depth[valid_depth][:,None])))
        sparse_point[img_name] = (np.hstack((points3D_xyz[valid_depth], point3D_idxes[valid_depth][:,None])))

    
    return sparse_depth, sparse_point