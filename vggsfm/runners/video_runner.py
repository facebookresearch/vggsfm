import torch
import cv2
import os
import copy
import random
import pyceres
import pycolmap
import numpy as np
import datetime
import logging
from .runner import VGGSfMRunner, move_to_device, add_batch_dimension, predict_tracks

from collections import defaultdict

from vggsfm.utils.utils import (
    write_array,
    generate_rank_by_midpoint,
    generate_rank_by_dino,
    generate_rank_by_interval,
    calculate_index_mappings,
    extract_dense_depth_maps,
    align_dense_depth_maps,
    switch_tensor_order,
    average_camera_prediction,
    average_camera_prediction,
    create_video_with_reprojections,
    save_video_with_reprojections,
)


from vggsfm.utils.tensor_to_pycolmap import (
    batch_matrix_to_pycolmap,
    pycolmap_to_batch_matrix,
)
from vggsfm.utils.align import align_camera_extrinsics, apply_transformation


from vggsfm.utils.triangulation import triangulate_tracks
from vggsfm.utils.triangulation_helpers import project_3D_points, filter_all_points3D, cam_from_img

class VideoRunner(VGGSfMRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # TODO: make a new config for video runner
        self.cfg.query_by_midpoint = True
        self.cfg.shared_camera = True
        self.cfg.camera_type = "SIMPLE_RADIAL"
        self.cfg.query_frame_num = 1
        self.cfg.fine_tracking = False

        self.point_dict = {}
        self.frame_dict = defaultdict(dict)
        self.crop_params = None
        self.intrinsics = None
        
        assert self.cfg.shared_camera== True, "Only shared camera is supported for video runner"
        
        # TODO: add a loop detection
        # TODO: support the handle of invalid frames
        # TODO: support camera parameter change in the future
        
    def calculate_bounding_boxes(self, crop_params, images):
        """
        Calculate bounding boxes if crop parameters are provided.
        """
        if crop_params is not None:
            # We know bound_bboxes is the same for a video 
            bound_bboxes = crop_params[:, 0:1, -4:-2].abs().to(self.device)
            # also remove those near the boundary
            bound_bboxes[bound_bboxes != 0] += self.remove_borders
            
            bound_bboxes = torch.cat(
                [bound_bboxes, images.shape[-1] - bound_bboxes], dim=-1
            )
            return bound_bboxes
        return None

    def process_initial_window(self, start_idx, end_idx, images, masks, crop_params, image_paths, query_frame_num, seq_name, output_dir):
        init_images, init_masks, init_crop_params = extract_window(
            start_idx, end_idx, images, masks, crop_params
        )

        init_pred = self.sparse_reconstruct(
            init_images,
            masks=init_masks,
            crop_params=init_crop_params,
            image_paths=image_paths[start_idx:end_idx],
            query_frame_num=query_frame_num,
            seq_name=seq_name,
            output_dir=output_dir,
            back_to_original_resolution=False,
        )
        return init_pred

    def convert_pred_to_point_dict(self, pred, start_idx, end_idx, points3D_idx = None):
        pred_track, pred_vis, valid_2D_mask, valid_tracks, points3D, points3D_rgb = pred["pred_track"], pred["pred_vis"], pred["valid_2D_mask"], pred["valid_tracks"],pred["points3D"], pred["points3D_rgb"]
                
        point_to_track_mapping = valid_tracks.nonzero().squeeze(1).cpu().numpy()
        
        if points3D_idx is None:
            points3D_idx = np.arange(len(points3D))

        extrinsics = pred["extrinsics_opencv"]

        
        # save them in cpu
        pred_track = pred_track.squeeze(0).cpu()
        pred_vis = pred_vis.squeeze(0).cpu()
        points3D = points3D.cpu()
        points3D_rgb = points3D_rgb.cpu()
        extrinsics = extrinsics.cpu()


        
        for frame_idx in range(start_idx, end_idx): 
            self.frame_dict[frame_idx]["extri"] = extrinsics[frame_idx]
            if "visible_points" not in self.frame_dict[frame_idx]:
                self.frame_dict[frame_idx]["visible_points"] = []

        
        for point_idx in points3D_idx:
            track_idx = point_to_track_mapping[point_idx]

            point_valid_2D_mask = valid_2D_mask[:, track_idx]
            point_track_dict = {}
            for frame_idx in range(start_idx, end_idx): 
                if point_valid_2D_mask[frame_idx]:
                    point_track_dict[frame_idx] = {"uv": pred_track[frame_idx, track_idx], 
                                                   "vis": pred_vis[frame_idx, track_idx]}
                    
                    self.frame_dict[frame_idx]["visible_points"].append(point_idx)
            
            point_dict = {"id": point_idx, 
                          "xyz": points3D[point_idx],
                          "rgb": points3D_rgb[point_idx],
                          "track": point_track_dict,
                          }
            
            self.point_dict[point_idx] = point_dict
            


    def run(
        self,
        images,
        masks=None,
        original_images=None,
        image_paths=None,
        crop_params=None,
        query_frame_num=None,
        seq_name=None,
        output_dir=None,
        init_window_size=16,
        window_size=16,
        stride=10,
    ):
        # (Pdb) images.shape
        # torch.Size([1, 411, 3, 1024, 1024])
        # masks None or torch.Size([1, 411, 1, 1024, 1024])
        # image_paths list of str
        # (Pdb) crop_params.shape
        # torch.Size([1, 411, 8])


        # NOTE
        # We assume crop_params, intrinsics, and extra_params are the same for a video
        
        if output_dir is None:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M")
            output_dir = f"{seq_name}_{timestamp}"

        with torch.no_grad():
            images = move_to_device(images, self.device)
            masks = move_to_device(masks, self.device)
            crop_params = move_to_device(crop_params, self.device)



            # Add batch dimension if necessary
            if len(images.shape) == 4:
                images = add_batch_dimension(images)
                masks = add_batch_dimension(masks)
                crop_params = add_batch_dimension(crop_params)


            # Calculate bounding boxes if crop parameters are provided
            # NOTE: We assume crop_params are the same for a video
            self.bound_bboxes = self.calculate_bounding_boxes(crop_params, images)

            B, T, C, H, W = images.shape
            self.B = B
            self.img_dim = C
            self.H = H
            self.W = W
            
            self.image_size = torch.tensor(
                [W, H], dtype=images.dtype, device=self.device
            )
            
            self.images = images
            self.masks = masks
            self.image_paths = image_paths

            if query_frame_num is None:
                query_frame_num = self.cfg.query_frame_num

            start_idx = 0
            end_idx = start_idx + init_window_size
            
            init_pred = self.process_initial_window(
                start_idx, end_idx, images, masks, crop_params, image_paths, query_frame_num, seq_name, output_dir
            )
            
            init_extri, init_intri, init_extra = init_pred["extrinsics_opencv"], init_pred["intrinsics_opencv"], init_pred["extra_params"]
            
            self.intrinsics = init_intri[0:1].clone() # 1x3x3
            self.extra_params = init_extra[0:1].clone() # 1xnum_extra_params
            
            self.convert_pred_to_point_dict(init_pred, start_idx, end_idx)

            
            # last_pred_track, last_pred_vis, last_pred_score, last_valid_2D_mask, last_points3D = init_pred["pred_track"], init_pred["pred_vis"], init_pred["pred_score"], init_pred["valid_2D_mask"], init_pred["points3D"]
            
            
            start_idx, end_idx = self.move_window(start_idx, end_idx, window_size)
            
            
    def move_window(self, start_idx, end_idx, window_size):
        
        # Move forward to the right first        
        last_start_idx = start_idx
        start_idx = end_idx 
        end_idx = start_idx + window_size
        
        # Include the last window in the next window for average_camera_prediction
        # because we need to align the camera prediction of the next window with the last window

        two_window_images = extract_window(last_start_idx, end_idx, self.images)[0]
        two_window_size = end_idx - last_start_idx

        # Predict extri for the combination of (last_window, current_window)
        pred_cameras = average_camera_prediction(self.camera_predictor,
                                    two_window_images.reshape(-1, self.img_dim, self.H, self.W),
                                    self.B, query_indices=[0, two_window_size//2, two_window_size-1])
        
        last_extri = torch.stack([self.frame_dict[frame_idx]["extri"] for frame_idx in range(last_start_idx, start_idx)]).to(self.device)
        
        pred_extri = torch.cat((pred_cameras.R, pred_cameras.T.unsqueeze(-1)), dim=-1)
        
        # Align to the last window
        rel_r, rel_t, rel_s = align_camera_extrinsics(pred_extri[last_start_idx:start_idx], last_extri)
        aligned_pred_extri_next_window = apply_transformation(pred_extri[start_idx:end_idx], rel_r, rel_t, rel_s)

        # First, only use existing point cloud to optimize aligned_pred_extri
        # 1. predict track with shape window_size + 1
    
        last_end_visible_points_idx = self.frame_dict[start_idx-1]["visible_points"]
        
        if len(last_end_visible_points_idx) > self.cfg.max_query_pts:
            last_end_visible_points_idx = sorted(random.sample(last_end_visible_points_idx, self.cfg.max_query_pts))
        
        last_end_visible_points_3D = [self.point_dict[point3D_idx]["xyz"] for point3D_idx in last_end_visible_points_idx]
        last_end_visible_points_3D = torch.stack(last_end_visible_points_3D)
        
        last_end_visible_points_2D = [self.point_dict[point3D_idx]["track"][start_idx-1]["uv"] for point3D_idx in last_end_visible_points_idx]
        last_end_visible_points_2D = torch.stack(last_end_visible_points_2D)
        
        window_images, window_masks = extract_window(start_idx-1, end_idx, self.images, self.masks)
        window_fmaps_for_tracker = self.track_predictor.process_images_to_fmaps(window_images)
        
        # NOTE self.cfg.query_method and self.cfg.max_query_pts are not used inside predict_tracks()
        # when query_points_dict is not None
        window_pred_track, window_pred_vis, window_pred_score = predict_tracks(
            self.cfg.query_method,  
            self.cfg.max_query_pts,
            self.track_predictor,
            window_images,
            window_masks,
            window_fmaps_for_tracker,
            [0],
            self.cfg.fine_tracking,
            self.bound_bboxes.expand(-1, window_images.shape[1], -1),
            query_points_dict = {0: last_end_visible_points_2D.to(self.device)[None]}
        )
        window_pred_track = window_pred_track.squeeze(0)
        window_pred_vis = window_pred_vis.squeeze(0)
        track_vis_thres = 0.1
        track_vis_inlier = window_pred_vis>track_vis_thres
        track_vis_valid = track_vis_inlier.sum(0)>2    # a track is invalid if without two inliers
        
        # Align to the last window by BA (without optimizing point cloud)
        last_end_extri = self.frame_dict[start_idx-1]["extri"][None].to(self.device)
        extri_window_plus_one = torch.cat([last_end_extri, aligned_pred_extri_next_window], dim=0)        
        align_extri_window_plus_one = self.align_next_window(extri_window_plus_one, window_pred_track, track_vis_inlier, last_end_visible_points_3D, )
    
    
        
        # NOTE It is window_size + 1 instead of window_size
        ba_options = pycolmap.BundleAdjustmentOptions()
        ba_options.refine_focal_length = False
        ba_options.refine_extra_params = False
        rec = batch_matrix_to_pycolmap(last_end_visible_points_3D, 
                                       align_extri_window_plus_one, 
                                       self.intrinsics.expand(window_size+1, -1, -1), 
                                       window_pred_track, 
                                       track_vis_inlier, 
                                       self.image_size, 
                                       shared_camera =self.cfg.shared_camera, 
                                       camera_type=self.cfg.camera_type, 
                                       extra_params=self.extra_params.expand(window_size+1, -1))
        
        ba_config = pycolmap.BundleAdjustmentConfig()
        for image_id in rec.reg_image_ids(): ba_config.add_image(image_id)

        # Fix frame 0, i.e, the end frame of the last window
        ba_config.set_constant_cam_pose(0)
        for fixp_idx in rec.point3D_ids():  
            ba_config.add_constant_point(fixp_idx)

        # Then, let's add more points to
        # 1. Predict tracks
        # 2. Do Triangulation
        # 3. Add them to rec and ba_config


        window_pred_track_NEW, window_pred_vis_NEW, window_pred_score_NEW = predict_tracks(
            self.cfg.query_method,  
            self.cfg.max_query_pts,
            self.track_predictor,
            window_images,
            window_masks,
            window_fmaps_for_tracker,
            [window_size//2, window_size],
            self.cfg.fine_tracking,
            self.bound_bboxes.expand(-1, window_images.shape[1], -1),
        )
        
        window_pred_track_NEW = window_pred_track_NEW.squeeze(0)
        window_pred_vis_NEW = window_pred_vis_NEW.squeeze(0)
        if window_pred_score_NEW is not None: window_pred_score_NEW = window_pred_score_NEW.squeeze(0)
        tracks_normalized_refined = cam_from_img(
            window_pred_track_NEW, self.intrinsics.expand(window_size+1, -1, -1), self.extra_params.expand(window_size+1, -1)
        )

        # Conduct triangulation to all the frames
        # We adopt LORANSAC here again
        (
            best_triangulated_points,
            best_inlier_num,
            best_inlier_mask,
        ) = triangulate_tracks(
            align_extri_window_plus_one,
            tracks_normalized_refined,
            track_vis=window_pred_vis_NEW,
            track_score=window_pred_score_NEW,
        )

        rec = add_triangulated_points_to_Reconstruction(rec, best_triangulated_points, window_pred_track_NEW, best_inlier_mask, best_inlier_num)
        
        # rec.points3D
        fixed_point3D_ids = ba_config.constant_point3D_ids
        for p_idx in rec.point3D_ids():  
            if p_idx not in fixed_point3D_ids:
                ba_config.add_variable_point(p_idx)
                
        # ba_config.variable_point3D_ids
        # Run BA

        import pdb;pdb.set_trace()

        # TODO: something is wrong here
        # TODO: enable to optimize only a part of the points
        
        summary = solve_bundle_adjustment(rec, ba_options, ba_config)

        log_ba_summary(summary)

        # from vggsfm.utils.triangulation_helpers import project_3D_points
        # projected_points2D, projected_points_cam = project_3D_points(
        #     last_end_visible_points_3D.cuda(),
        #     align_extri_window_plus_one.cuda(),
        #     self.intrinsics.cuda().expand(window_size+1, -1, -1),
        #     extra_params=self.extra_params.cuda().expand(window_size+1, -1),
        #     return_points_cam=True,
        # )

        # m = projected_points2D - window_pred_track.cuda()


        # import pdb;pdb.set_trace()

        # logging.info(summary.BriefReport())
        
        # ba_config.num_images()
        import pdb;pdb.set_trace()
        # ba_config_fix = copy.deepcopy(ba_config)
        obmanager = pycolmap.ObservationManager(rec)
        obmanager.filter_observations_with_negative_depth()
        obmanager.compute_mean_reprojection_error()


        # logging.info(summary.BriefReport())

        import pdb;pdb.set_trace()
        # ba_config


        # pycolmap.bundle_adjustment(rec, ba_options)

        (
            points3D_opt,
            extrinsics_opt,
            intrinsics_opt,
            extra_params_opt,
        ) = pycolmap_to_batch_matrix(
            rec, device=self.device, camera_type=self.cfg.camera_type
        )
        
        # solve_bundle_adjustment(reconstruction, ba_options, ba_config)

        # ConstantPoints
        import pdb;pdb.set_trace()


        
        import pdb;pdb.set_trace()

        
        import pdb;pdb.set_trace()
        
        
        m=1

        # from vggsfm.utils.utils import visual_query_points
        # from vggsfm.utils.visualizer import Visualizer

        # visual_query_points(window_images, 0, window_pred_track[0:1])

        # vis = Visualizer(save_dir="visual", linewidth=1)
        # vis.visualize(window_images * 255, window_pred_track[None], window_pred_vis[None][..., None], filename="track")


        """
        from vggsfm.utils.triangulation_helpers import project_3D_points, filter_all_points3D, cam_from_img

        projected_points2D, projected_points_cam = project_3D_points(
            last_end_visible_points_3D.cuda(),
            last_end_extri.cuda(),
            self.intrinsics.cuda(),
            extra_params=self.extra_params.cuda(),
            return_points_cam=True,
        )

        # projected_points2D - last_end_visible_points_2D.cuda()
        """




        # last_end_frame = last_end_idx - 1
        last_end_valid_2D = last_pred_track[:, -1][:, last_valid_2D_mask[-1]]
        
        window_images, window_masks = extract_window(last_end_idx - 1, end_idx, images, masks)
        window_T = window_images.shape[1]
        window_fmaps_for_tracker = self.track_predictor.process_images_to_fmaps(window_images)
    
        
        return None


    def align_next_window(self, extrinsics, tracks, inlier, points3D, use_pnp=False):
        # extrinsics: Sx3x4
        # tracks: SxPx2
        # inlier: SxP
        # points3D: P x 3

        S, _, _ = extrinsics.shape
        _, P, _ = tracks.shape

        # P' x 3
        points3D = points3D.cpu().numpy()
        # S x P' x 2
        tracks2D = tracks.cpu().numpy()
        inlier = inlier.cpu().numpy()
        refoptions = pycolmap.AbsolutePoseRefinementOptions()
        refoptions.refine_focal_length = False
        refoptions.refine_extra_params = False
        refoptions.print_summary = False

        refined_extrinsics = []
        if self.cfg.camera_type == "SIMPLE_RADIAL":
            pycolmap_intri = np.array(
                [
                    self.intrinsics[0][0, 0].cpu().numpy(),
                    self.intrinsics[0][0, 2].cpu().numpy(),
                    self.intrinsics[0][1, 2].cpu().numpy(),
                    self.extra_params[0][0].cpu().numpy(),
                ]
            )
        elif self.cfg.camera_type == "SIMPLE_PINHOLE":
            pycolmap_intri = np.array(
                [
                    self.intrinsics[0][0, 0].cpu().numpy(),
                    self.intrinsics[0][0, 2].cpu().numpy(),
                    self.intrinsics[0][1, 2].cpu().numpy(),
                ]
            )
        else:
            raise NotImplementedError(f"Camera type {self.cfg.camera_type} not implemented")

        # We assume the same camera for all frames in a video
        pycam = pycolmap.Camera(model=self.cfg.camera_type,
                    width=self.image_size[0],
                    height=self.image_size[1],
                    params=pycolmap_intri,
                    camera_id=0,
                )
        
        for ridx in range(S):
            if ridx==0:
                refined_extrinsics.append(extrinsics[ridx].cpu().numpy())
                continue
            cam_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(extrinsics[ridx][:3, :3].cpu()),
                extrinsics[ridx][:3, 3].cpu(),
            )  # Rot and Trans
            points2D = tracks2D[ridx]
            inlier_mask = inlier[ridx]

            if inlier_mask.sum() <= 50:
                # If too few inliers, ignore it
                # use all the points
                print("too small inliers")
                inlier_mask[:] = 1
                
            if use_pnp:
                estoptions = pycolmap.AbsolutePoseEstimationOptions()
                estoptions.ransac.max_error = 12


                estanswer = pycolmap.absolute_pose_estimation(
                    points2D[inlier_mask],
                    points3D[inlier_mask],
                    pycam,
                    estoptions,
                    refoptions,
                )
                cam_from_world = estanswer["cam_from_world"]

                
            answer = pycolmap.pose_refinement(
                cam_from_world,
                points2D,
                points3D,
                inlier_mask,
                pycam,
                refoptions,
            )
            

            cam_from_world = answer["cam_from_world"]
            refined_extrinsics.append(cam_from_world.matrix())

        # get the optimized cameras
        refined_extrinsics = torch.from_numpy(np.stack(refined_extrinsics)).to(
            tracks.device
        )
        return refined_extrinsics


def add_triangulated_points_to_Reconstruction(rec, points3D, tracks, inlier_mask, inlier_num, min_valid_track_length=3):

    # TODO: THIS MAY BE WRONG SOMEWHERE
    # CAREFUL
    valid_track_mask = inlier_num >= min_valid_track_length

    valid_points = points3D[valid_track_mask].cpu().numpy()
    valid_tracks = tracks[:, valid_track_mask].cpu().numpy()
    valid_inlier_masks = inlier_mask[valid_track_mask].transpose(0, 1).cpu().numpy()

    per_point_inlier_num = valid_inlier_masks.sum(0)
    valid_point_mask = per_point_inlier_num >= 2
    valid_point_idx = np.nonzero(valid_point_mask)[0]

    start_point3D_idx = max(rec.point3D_ids()) + 1
    
    end_point3D_idx = start_point3D_idx + len(valid_points)
    # Add 3D points to rec
    for vidx in valid_point_idx: 
        rec.add_point3D(valid_points[vidx], pycolmap.Track(), np.zeros(3))
        
    
    num_images = rec.num_images()
    
    for fidx in range(num_images):
        pyimg = rec.images[fidx]
        # Existing 2D points
        point2D_idx = len(pyimg.points2D)
        
        
        for point3D_id in range(start_point3D_idx, end_point3D_idx): 
            
            original_track_idx = valid_point_idx[point3D_id - start_point3D_idx]

            if valid_inlier_masks[fidx][original_track_idx]:
                # It seems we don't need +0.5 for BA
                point2D_xy = valid_tracks[fidx][original_track_idx]
                # Please note when adding the Point2D object
                # It not only requires the 2D xy location, but also the id to 3D point
                pyimg.points2D.append(
                    pycolmap.Point2D(point2D_xy, point3D_id)
                )

                # add element
                track = rec.points3D[point3D_id].track
                track.add_element(fidx, point2D_idx)
                point2D_idx += 1
                
    return rec
    
    
    
def log_ba_summary(summary):
    logging.info(f"Residuals : {summary.num_residuals_reduced}")
    logging.info(f"Parameters : {summary.num_effective_parameters_reduced}")
    logging.info(f"Iterations : {summary.num_successful_steps + summary.num_unsuccessful_steps}")
    logging.info(f"Time : {summary.total_time_in_seconds} [s]")
    logging.info(f"Initial cost : {np.sqrt(summary.initial_cost / summary.num_residuals_reduced)} [px]")
    logging.info(f"Final cost : {np.sqrt(summary.final_cost / summary.num_residuals_reduced)} [px]")


def solve_bundle_adjustment(reconstruction, ba_options, ba_config):
    bundle_adjuster = pycolmap.BundleAdjuster(ba_options, ba_config)
    # alternative equivalent python-based bundle adjustment (slower):
    # bundle_adjuster = PyBundleAdjuster(ba_options, ba_config)
    bundle_adjuster.set_up_problem(
        reconstruction, ba_options.create_loss_function()
    )
    solver_options = bundle_adjuster.set_up_solver_options(
        bundle_adjuster.problem, ba_options.solver_options
    )
    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, bundle_adjuster.problem, summary)
    return summary

            
            

def extract_window(start_idx, end_idx, *vars):
    """
    Extracts a window from start_idx to end_idx along dimension 1 for each variable in vars.
    """
    return [var[:, start_idx:end_idx, ...] if var is not None else None for var in vars]

def remove_query(*vars):
    """
    Removes the first element along dimension 1 for each variable in vars.
    """
    return [var[:, 1:, ...] if var is not None else None for var in vars]



# def move_to_device(tensor, device):
#     return tensor.to(device) if tensor is not None else None

# def add_batch_dimension(tensor):
#     return tensor.unsqueeze(0) if tensor is not None else None

        
        
        


            # window_pred_track, window_pred_vis, window_pred_score = predict_tracks(
            #     self.cfg.query_method,
            #     self.cfg.max_query_pts,
            #     self.track_predictor,
            #     window_images,
            #     window_masks,
            #     window_fmaps_for_tracker,
            #     [0],
            #     self.cfg.fine_tracking,
            #     bound_bboxes.expand(-1, window_T, -1),
            #     query_points_dict = {0:last_end_valid_2D}
            # )

            # # TODO: Do not optimize intrinsics here
            # from vggsfm.utils.triangulation import iterative_global_BA, batch_matrix_to_pycolmap
            
            
            # image_size = torch.tensor(
            #     [W, H], dtype=window_pred_track.dtype, device=self.device
            # )
            # valid_tracks = last_valid_2D_mask.sum(0)>=3
            # import pycolmap
            # ba_options = pycolmap.BundleAdjustmentOptions()


            # reconstruction = batch_matrix_to_pycolmap(
            #     BA_points,
            #     aligned_pred_extri,
            #     intrinsics,
            #     BA_tracks,
            #     BA_inlier_masks,
            #     image_size,
            #     shared_camera=shared_camera,
            #     camera_type=camera_type,
            #     extra_params=extra_params,  # Pass extra_params to batch_matrix_to_pycolmap
            # )
                


            # visual_query_points(window_images, 0, last_end_valid_2D)
            # from vggsfm.utils.utils import visual_query_points
            # import pdb;pdb.set_trace()
            # m=1
        
            #         ba_options = pycolmap.BundleAdjustmentOptions()
            # (
            #         points3D,
            #         extrinsics,
            #         intrinsics,
            #         extra_params,
            #         valid_tracks,
            #         BA_inlier_masks,
            #         reconstruction,
            #     ) = iterative_global_BA(
            #         window_pred_track[0,1:],
            #         last_intri[0:1].expand(window_T-1, -1, -1),
            #         aligned_pred_extri[last_end_idx:],
            #         window_pred_vis[0, 1:],
            #         window_pred_score[0, 1:],
            #         valid_tracks,
            #         last_points3D,
            #         image_size,
            #         lastBA=False,
            #         extra_params=last_extra[0:1].expand(window_T-1, -1, -1),
            #         shared_camera=self.cfg.shared_camera,
            #         min_valid_track_length=self.cfg.min_valid_track_length,
            #         max_reproj_error=self.cfg.max_reproj_error,
            #         ba_options=ba_options,
            #         camera_type=self.cfg.camera_type,
            # )
            
            