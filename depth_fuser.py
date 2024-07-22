import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from gsplat.rendering import rasterization

import torch.utils
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torchvision
from torch.cuda.amp import autocast

class DepthFuser(nn.Module):
    def __init__(self, device="cuda"):
        super(DepthFuser, self).__init__()
        
        self.init_depths = {}
        # TODO add a depth scale?
        self.depth_residuals = {}
        self.points3D = {}
        self.rgbs = {}
        self.cam_from_worlds = {}
        self.Ks = {}
        self.images = {}
        self.sizes = {}
        self.unproject_homo = {}
        self.depth_scales = {}
        self.device = device
        self.masks = {}
        self.sparse_depths = {}
        self.ssim_lambda = 0.2
        
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        
    def add_frame(self, img_name, cam_from_world, K, depth, unproject_points_homo, rgb, rgb_image, size, mask, sprase_depth):
        with torch.enable_grad():
            assert img_name not in self.depth_residuals
            
            homogenous_row = np.array([0, 0, 0, 1])
            cam_from_world_homo = torch.from_numpy(np.vstack((cam_from_world, homogenous_row))).to(self.device)

            unproject_points_homo = torch.from_numpy(unproject_points_homo).to(self.device) 

            depth_residual = torch.nn.Parameter(torch.zeros(depth.shape).to(self.device))
            
            depth_scale = torch.nn.Parameter(torch.ones_like(depth_residual).to(self.device))
            # depth_scale = torch.nn.Parameter(torch.ones(1).to(self.device))
            
            depth_map = torch.from_numpy(depth).to(self.device) * depth_scale + depth_residual
            
            unproject_points_withz = unproject_points_homo * depth_map[:, None]
            
            cam_to_world_homo = torch.linalg.inv(cam_from_world_homo)
            unproject_points_world = unproject_points_withz @ cam_to_world_homo[:3,:3].t() + cam_to_world_homo[:3,3][None]
            
            
            self.unproject_homo[img_name] = unproject_points_homo.float()
            self.init_depths[img_name] = torch.from_numpy(depth).to(self.device).float()
            self.rgbs[img_name] = torch.from_numpy(rgb).to(self.device).float()
            self.Ks[img_name] = torch.from_numpy(K).to(self.device).float()
            self.images[img_name] = torch.from_numpy(rgb_image).to(self.device).float()
            self.masks[img_name] = torch.from_numpy(mask).to(self.device)
            self.sparse_depths[img_name] = torch.from_numpy(sprase_depth).to(self.device)
            self.sizes[img_name] = size

            
            self.depth_residuals[img_name] = depth_residual.float()
            self.depth_scales[img_name]  = depth_scale.float()
            self.points3D[img_name] = unproject_points_world.float()
            self.cam_from_worlds[img_name] = cam_from_world_homo.float()



    def prepare_for_opt(self):
        with torch.enable_grad():
            self.frame_names = [key for key in self.points3D.keys()]

            self.points_all = []
            self.rgbs_all = []

            for img_name in self.frame_names:
                self.points_all.append(self.points3D[img_name])
                self.rgbs_all.append(self.rgbs[img_name])
            self.points_all = torch.cat(self.points_all)
            self.rgbs_all = torch.cat(self.rgbs_all)
            self.points_all_init = self.points_all.clone()
            self.rgbs_all_init = self.rgbs_all.clone()

    def update_points3D(self):
        
        self.points_all = []
        self.rgbs_all = []

        use_sparse_depth = False

        residual_ratio = 5

        for img_name in self.frame_names:
            
            unproject_points_homo = self.unproject_homo[img_name]

            depth_residual = self.depth_residuals[img_name]
            depth_scale = self.depth_scales[img_name]


            if use_sparse_depth:
                depth_map = self.sparse_depths[img_name].float() * depth_scale + depth_residual * residual_ratio
            else:
                depth_map = self.init_depths[img_name] * depth_scale + depth_residual * residual_ratio
            
            
            unproject_points_withz = unproject_points_homo * depth_map[:, None]
            cam_from_world_homo = self.cam_from_worlds[img_name]
            cam_to_world_homo = torch.linalg.inv(cam_from_world_homo)
            unproject_points_world = unproject_points_withz @ cam_to_world_homo[:3,:3].t() + cam_to_world_homo[:3,3][None]
            self.points3D[img_name] = unproject_points_world

            # concat
            self.points_all.append(self.points3D[img_name])
            self.rgbs_all.append(self.rgbs[img_name])
            
        self.points_all = torch.cat(self.points_all)
        self.rgbs_all = torch.cat(self.rgbs_all)
            
        
        
    def optimize_depth(self,):
        with torch.enable_grad():
            steps = 1000

            optimize_opc = True
            optimize_depth = True
            optimize_pts3d = False
            use_render_as_gt = True
            
            self.opacities = {}
            for tname in self.frame_names:
                curopa = torch.full((len(self.depth_residuals[tname]),), 0.50).to(self.device) 
                self.opacities[tname] = torch.nn.Parameter(curopa)

            optparams = []
            
            if optimize_pts3d:
                self.points_all = torch.nn.Parameter(self.points_all)
                optparams += [self.points_all]
            
            if optimize_depth:
                assert not optimize_pts3d
                optparams += [param for param in self.depth_residuals.values()]
                optparams += [param for param in self.depth_scales.values()]
                
            if optimize_opc:
                optparams += [param for param in self.opacities.values()]
                        
            lr = 0.001
            # Define the Adam optimizer
            # optimizer = torch.optim.Adam(optparams, lr=lr)
            optimizer = torch.optim.AdamW(optparams, lr=lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1.0 / steps))


            num = len(self.points_all)
            scales = torch.ones(num, 3).detach().to(self.device) * 1e-4
            quats = torch.ones((num, 4)).detach().to(self.device)
            quats[:, 1:] = 0             

            opa_all = torch.cat([self.opacities[tmp] for tmp in self.frame_names], dim=0)
            pcmask = opa_all > 0.0


            import time
            start_time = time.time()

            batch_size = 8
            for step in range(steps):
                if not optimize_pts3d:
                    self.update_points3D()

                opa_all = torch.cat([self.opacities[tmp] for tmp in self.frame_names], dim=0)
                
                loss = 0
                sampled = random.sample(self.frame_names, batch_size)

                ############################################################################################################################################
                viewmats = []
                Ks = []
                for sampled_name in sampled: 
                    viewmats.append(self.cam_from_worlds[sampled_name][None])
                    Ks.append(self.Ks[sampled_name][None])
                viewmats = torch.cat(viewmats)
                Ks = torch.cat(Ks)
                # Assume the same size for all the images
                width = self.sizes[sampled_name][0]
                height = self.sizes[sampled_name][1]
                ############################################################################################################################################


                render_colors, render_alphas, info = rasterization(
                    means=self.points_all[pcmask],
                    quats=quats[pcmask],
                    scales=scales[pcmask],
                    opacities=opa_all[pcmask],
                    colors=self.rgbs_all[pcmask],
                    viewmats = viewmats, 
                    Ks = Ks,   
                    width=width,
                    height=height,
                    packed=False,
                    absgrad=False,
                    sparse_grad=False,
                    rasterize_mode="classic",
                    radius_clip = 0.0,
                )

                curidx = 0
                for sampled_name in sampled: 
                    # print(sampled_name)
                    # TODO: convert K to gsplat K
                    # TODO: multiple gt 
                    
                    if use_render_as_gt:
                        with torch.no_grad():
                            num_on_one_frame = len(self.points3D[sampled_name])
                            image_gt, _, _ = rasterization(
                                means=self.points3D[sampled_name],
                                quats=quats[:num_on_one_frame],
                                scales=scales[:num_on_one_frame],
                                opacities=torch.ones_like(self.opacities[sampled_name]),
                                colors=self.rgbs[sampled_name],
                                viewmats = viewmats[curidx][None], 
                                Ks = Ks[curidx][None],   
                                width=width,
                                height=height,
                                packed=False,
                                absgrad=False,
                                sparse_grad=False,
                                rasterize_mode="classic",
                                radius_clip = 0.0,
                            )
                    else:
                        image_gt = self.images[sampled_name][None]
                    
                    valid_mask = self.masks[sampled_name][None]

                    image_gt[~valid_mask] = 0
                    
                    if False:
                        render_colors[curidx] = render_colors[curidx] * valid_mask[0][...,None]
                        cur_render_colors = render_colors[curidx][None]
                    else:
                        cur_render_colors = render_colors[curidx][None].clone()
                        cur_render_colors[~valid_mask] = 0
                        render_colors[curidx] = render_colors[curidx] * valid_mask[0][...,None]



                    if self.ssim_lambda>0:
                        l1loss = F.l1_loss(cur_render_colors, image_gt)
                        ssimloss = 1.0 - self.ssim(
                            cur_render_colors.permute(0, 3, 1, 2), image_gt.permute(0, 3, 1, 2)
                        )
                        tmploss = l1loss * (1.0 - self.ssim_lambda) + ssimloss * self.ssim_lambda
                    else:
                        tmploss = F.l1_loss(cur_render_colors, image_gt)

                    tmploss = l1loss
                    loss += tmploss
                    curidx += 1
                    
                loss = loss/batch_size
                print("Step: " , step, " Loss: ", loss.item(), " Lr: ", optimizer.param_groups[0]["lr"])

                loss.backward()
                
                if step%100==0:
                    if step!=0:
                        pcmask = opa_all > 0.48
                    print("pcmask num: ", pcmask.sum())
                    torchvision.utils.save_image(render_colors.permute(0, 3, 1, 2), f'tmp/debug_step_{step:06d}.png', nrow=1, padding=0)
                    if optimize_depth:  
                        for mname in self.frame_names: print(f"max grad: {self.depth_residuals[mname].grad.max()}")
                    else:
                        print(f"max grad: {self.points_all.grad.max()}")
                                        
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        
        
            end_time = time.time()
            elapsed_time_sec = end_time - start_time
            elapsed_time_min = elapsed_time_sec / 60
            print(f"Elapsed time: {elapsed_time_sec:.6f} seconds ({elapsed_time_min:.6f} minutes)")


            pcmask = opa_all >= 0.48

            render_colors, render_alphas, info = rasterization(
                means=self.points_all,
                quats=quats,
                scales=scales,
                opacities=torch.ones_like(opa_all),
                colors=self.rgbs_all,
                viewmats = viewmats, 
                Ks = Ks,   
                width=width,
                height=height,
                packed=False,
                absgrad=False,
                sparse_grad=False,
                rasterize_mode="classic",
                radius_clip = 0.0,
            )
            torchvision.utils.save_image(render_colors.permute(0, 3, 1, 2), f'tmp/final_unmasked.png', nrow=1, padding=0)


            render_colors, render_alphas, info = rasterization(
                means=self.points_all[pcmask],
                quats=quats[pcmask],
                scales=scales[pcmask],
                opacities=torch.ones_like(opa_all[pcmask]),
                colors=self.rgbs_all[pcmask],
                viewmats = viewmats, 
                Ks = Ks,   
                width=width,
                height=height,
                packed=False,
                absgrad=False,
                sparse_grad=False,
                rasterize_mode="classic",
                radius_clip = 0.0,
            )
            torchvision.utils.save_image(render_colors.permute(0, 3, 1, 2), f'tmp/final_masked.png', nrow=1, padding=0)



            from pytorch3d.structures import Pointclouds
            from pytorch3d.vis.plotly_vis import plot_scene
            from pytorch3d.renderer.cameras import PerspectiveCameras as PerspectiveCamerasVisual

            from pytorch3d.implicitron.tools import model_io, vis_utils
            viz = vis_utils.get_visdom_connection(server=f"http://10.200.188.27", port=10088)
    
            pcl = Pointclouds(points=self.points_all[pcmask][None], features = self.rgbs_all[pcmask][None])
        
            pcl_all = Pointclouds(points=self.points_all[None], features = self.rgbs_all[None])
            
            pcl_all_init = Pointclouds(points=self.points_all_init[None], features = self.rgbs_all_init[None])
            visual_dict = {"scenes": {"points_opt_masked": pcl, "points_opt":pcl_all, "points_raw": pcl_all_init}}
            fig = plot_scene(visual_dict, camera_scale=0.05)
            env_name = "opt_depth"
            viz.plotlyplot(fig, env=env_name, win="3D")

            import pdb;pdb.set_trace()

            m=1
        return 
        
        
        
        
        
    # def optimizer_by_pair(self,):
    #     with torch.enable_grad():
            
    #         opt_pts3d = False
    #         if opt_pts3d:
    #             for img_name in self.points3D:
    #                 self.points3D[img_name] = torch.nn.Parameter(self.points3D[img_name].detach())
            
    #         for img_name in self.frame_names:
    #             steps = 5000
                
    #             optimize_opc = True

    #             if opt_pts3d:
    #                 raw_pts3D = self.points3D[img_name].detach().clone()
    #                 optparams = [self.points3D[img_name]]
    #                 if optimize_opc:
    #                     opacities = torch.full((len(self.points_all),), 0.50).to(self.device) 
    #                     opacities = torch.nn.Parameter(opacities)
    #                     optparams += [opacities]
    #             else:
    #                 optparams = [self.depth_residuals[img_name]]
    #                 optparams += [self.depth_scales[img_name]]
    #                 # optimize_opc = False
                    
    #                 if optimize_opc:
    #                     self.opacities = {}
    #                     for tname in self.frame_names:
    #                         curopa = torch.full((len(self.depth_residuals[tname]),), 0.80).to(self.device) 
    #                         self.opacities[tname] = torch.nn.Parameter(curopa)
                            
    #                     optparams += [self.opacities[img_name]]

    #                     # opacities = torch.full((len(self.depth_residuals[img_name]),), 0.80).to(self.device) 
    #                     # opacities = torch.nn.Parameter(opacities)
    #                     # optparams += [opacities]
                    
                
    #             lr = 0.001
    #             # Define the Adam optimizer
    #             optimizer = torch.optim.Adam(optparams, lr=lr)
    #             # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    #             scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1.0 / steps))

                
                
    #             for step in range(steps):
    #                 ref_img_name = self.frame_names[step % len(self.frame_names)]
                    
    #                 if not opt_pts3d:
    #                     self.update_points3D()
                    
    #                 points_all = torch.cat([self.points3D[img_name], self.points3D[ref_img_name].detach()], dim=0)
    #                 rgbs_all = torch.cat([self.rgbs[img_name], self.rgbs[ref_img_name].detach()], dim=0)
    #                 opa_all = torch.cat([self.opacities[img_name], self.opacities[ref_img_name].detach()], dim=0)
                    
    #                 num = len(points_all)
    #                 scales = torch.ones_like(points_all).detach() * 1e-4
                    
    #                 quats = torch.ones((num, 4)).detach().to(self.device)
    #                 quats[:, 1:] = 0             # should this be [1,0,0,0]?

            
    #                 if not optimize_opc:
    #                     opacities = torch.full((num,), 1.00).to(self.device) 


    #                 with torch.no_grad():
    #                     render_colors_GT = self.images[ref_img_name][None]



    #                 render_colors, render_alphas, info = rasterization(
    #                     means=points_all,
    #                     quats=quats,
    #                     scales=scales,
    #                     opacities=opa_all,
    #                     colors=rgbs_all,
    #                     viewmats = self.cam_from_worlds[ref_img_name][None],
    #                     Ks = self.Ks[ref_img_name][None],
    #                     width=self.sizes[ref_img_name][0],
    #                     height=self.sizes[ref_img_name][1],
    #                     packed=False,
    #                     absgrad=False,
    #                     sparse_grad=False,
    #                     rasterize_mode="classic",
    #                     radius_clip = 0.0,
    #                 )
                    
    #                 # absloss = (render_colors_GT - render_colors).abs().mean(-1)
    #                 valid_mask = self.masks[ref_img_name]
    #                 mask = valid_mask[None]
    #                 # valuemask = (absloss > 0.05)
    #                 # mask = torch.logical_and(valid_mask[None], valuemask)
    #                 # l1loss = F.l1_loss(render_colors[mask], render_colors_GT[mask])
    #                 # 
    #                 # l2loss = F.mse_loss(render_colors[mask], render_colors_GT[mask])
    #                 loss = F.l1_loss(render_colors[mask], render_colors_GT[mask])

    #                 print("Step: " , step, " Loss: ", loss.item(), " Lr: ", optimizer.param_groups[0]["lr"])

    #                 loss.backward()
                    
    #                 # if step%10==0:
    #                 optimizer.step()
    #                 optimizer.zero_grad(set_to_none=True)
    #                 scheduler.step()

    #                 # print(f"max value: {self.depth_residuals[img_name].max()}")

    #                 if opt_pts3d:
    #                     diff = (raw_pts3D - self.points3D[img_name]).max()
    #                     print(diff)

    #                 if step%50==1:

    #                     torchvision.utils.save_image(render_colors_GT.permute(0, 3, 1, 2), f'visualtmp/step_{step}_GT.png', nrow=1, padding=0)
    #                     torchvision.utils.save_image(render_colors.permute(0, 3, 1, 2), f'visualtmp/step_{step}_render.png', nrow=1, padding=0)

    #             torchvision.utils.save_image(render_colors_GT.permute(0, 3, 1, 2), f'visualtmp/step_{step}_GT.png', nrow=1, padding=0)
    #             torchvision.utils.save_image(render_colors.permute(0, 3, 1, 2), f'visualtmp/step_{step}_render.png', nrow=1, padding=0)

    #             import pdb;pdb.set_trace()
    #             m=1
        