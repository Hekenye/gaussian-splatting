import torch
import torch.nn.functional as F
from tqdm import tqdm

from typing import (
    Optional,
    Tuple,
    Union,
)

def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
):
    """
    Get ray directions for all pixels in camera coordinate.
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )
    
    # nerf OpenGL
    # directions = torch.stack(
    #     [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    # )
    
    # OpenGL -> Opencv
    directions = torch.stack(
        [(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1
    )

    return directions


def get_rays(
    directions: torch.Tensor,
    c2w: torch.Tensor,
):
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3
    assert c2w.ndim == 2  # (4, 4)

    # c2w[:3, :3] = R  c2w[:3, 3] = t
    rays_d = directions[:, None, :].matmul(c2w[:3, :3].transpose(0, 1)).squeeze()  # (N_rays, 3)
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (N_rays, 3)
    rays_d = F.normalize(rays_d, dim=-1)
    return rays_o, rays_d

def trace_gaussians(
        points,
        segmap,
        H: int,
        W: int,
        focal,
        principal,
        c2w,
        batch_size,
        device,
):
    # flatten_segmap.shape = [H*W, 6], 0-5: background, head, body-skin, face-skin, clothes, others
    flatten_segmap = torch.from_numpy(segmap).to(device).permute(1, 2, 0).reshape(-1, 6)
    background_mask = flatten_segmap[:, 0] > 0.5
    valid_rays_mask = ~background_mask

    # get rays from image pixel
    directions = get_ray_directions(H, W, focal, principal).to(device)
    directions = directions.reshape(-1, 3)   # [H*W, 3]
    valid_directions = directions[valid_rays_mask]
    rays_logits = flatten_segmap[valid_rays_mask]     # [H*W, 6] -> [N_r, 6]

    rays_o, rays_d = get_rays(valid_directions, c2w)
    rays_o = rays_o.unsqueeze(0)  # [N_r, 3] --> [1, N_r, 3]
    rays_d = rays_d.unsqueeze(0)

    num_points = points.shape[0]

    progress_bar = tqdm(range(0, num_points), desc="Tracing gaussians")
    indices = []
    for i in range(0, num_points, batch_size):
        if i + batch_size > num_points:
            batch_points = points[i:]
        else:
            batch_points = points[i:i+batch_size]

        batch_points = batch_points.unsqueeze(1)      # [bs, 3] --> [bs, 1, 3]
        dist = (batch_points - rays_o).cross(rays_d, dim=-1).norm(dim=-1)    # [bs, N_r]
        indice = torch.argmin(dist, dim=1)    # [bs]
        indices.append(indice)
        progress_bar.update(batch_size)

    indice = torch.cat(indices, dim=0)    # [N_p], value in [0, H*W]

    # accumulated segmap for each point
    points_seg_logits = rays_logits[indice]
    # head_points_mask = points_seg_logits[:, [1, 3, 5]].sum(dim=1) > 0.5
    head_points_mask = points_seg_logits[:, [1, 3]].sum(dim=1) > 0.5
    # torso_points_mask =  points_seg_logits[:, [2, 4]].sum(dim=1) > 0.5
    # return head_points_mask, torso_points_mask
    return head_points_mask
