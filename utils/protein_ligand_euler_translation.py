import torch

def align_point_clouds(A, B, pivot):
    """
    A: point cloud A, of shape (N, 3)
    B: point cloud B, of shape (N, 3)
    pivot: the pivot point, of shape (3,)
    """
    # Convert pivot point to a 1x3 tensor
    pivot = torch.tensor(pivot, dtype=torch.float32).unsqueeze(0)

    # Translate A and B to pivot
    A -= pivot
    B -= pivot

    # Compute rotation matrix
    cov = torch.matmul(B.T, A)
    U, S, V = torch.svd(cov)
    R = torch.matmul(V, U.T)

    # Rotate A into alignment with B
    A_aligned = torch.matmul(R, A.T).T

    # Translate A back to original position
    A_aligned += pivot

    return A_aligned, R

def compute_euler_angles(R):
    """
    R: rotation matrix, of shape (3, 3)
    """
    # Extract rotation angles from rotation matrix
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0

    # Convert rotation angles to degrees
    x = x * 180 / torch.pi
    y = y * 180 / torch.pi
    z = z * 180 / torch.pi

    return x, y, z

def get_rotation_matrix(A, B, pivot):
    """
    A: point cloud A, of shape (N, 3)
    B: point cloud B, of shape (N, 3)
    pivot: the pivot point, of shape (3,)
    """
    # Convert point clouds to tensors
    A = torch.tensor(A, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)

    # Compute center of mass for A and B
    com_A = A.mean(dim=0, keepdim=True)
    com_B = B.mean(dim=0, keepdim=True)

    # Translate A and B to their respective centers of mass
    A -= com_A
    B -= com_B

    # Rotate A into alignment with B
    A_aligned, R = align_point_clouds(A, B, pivot)

    return R

def get_euler_angles_from_point_clouds(A, B, pivot):
    """
    A: point cloud A, of shape (N, 3)
    B: point cloud B, of shape (N, 3)
    pivot: the pivot point, of shape (3,)
    """
    # Compute rotation matrix from A to B
    R = get_rotation_matrix(A, B, pivot)

    # Compute Euler angles from rotation matrix
    x, y, z = compute_euler_angles(R)

    return x, y, z



def get_euclidean_kabsch(pos, ref, pos_mask):
    # pos [N,M,3]
    # ref [N,M,3]
    # pos_mask [N,M]
    # N : number of examples
    # M : number of atoms
    # R,T maps local reference onto global pos
    if pos_mask is None:
        pos_mask = torch.ones(pos.shape[:2], device=pos.device)
    else:
        if pos_mask.shape[0] != pos.shape[0]:
            raise ValueError("pos_mask should have same number of rows as number of input vectors.")
        if pos_mask.shape[1] != pos.shape[1]:
            raise ValueError("pos_mask should have same number of cols as number of input vector dimensions.")
        if pos_mask.ndim != 2:
            raise ValueError("pos_mask should be 2 dimensional.")
    # Center point clouds
    denom = torch.sum(pos_mask, dim=1, keepdim=True)
    denom[denom == 0] = 1.
    pos_mu = torch.sum(pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
    ref_mu = torch.sum(ref * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
    pos_c = pos - pos_mu
    ref_c = ref - ref_mu
    # Covariance matrix
    H = torch.einsum('bji,bjk->bik', ref_c, pos_mask[:, :, None] * pos_c)
    U, S, Vh = torch.linalg.svd(H)
    # Decide whether we need to correct rotation matrix to ensure right-handed coord system
    locs = torch.linalg.det(U @ Vh) < 0
    S[locs, -1] = -S[locs, -1]
    U[locs, :, -1] = -U[locs, :, -1]
    # Rotation matrix
    R = torch.einsum('bji,bkj->bik', Vh, U)
    # Translation vector
    T = pos_mu - torch.einsum('bij,bkj->bki', R, ref_mu)
    return R, T.squeeze(1)




A = torch.rand([5,3])
B = A + torch.tensor([0.5,0.5,-1.0])
pivot = A[0]

center = torch.mean(A, dim=0, keepdim=True)
    

# 计算点云之间的欧拉角
# 生成随机欧拉角
# A矩阵通过欧拉角做旋转得到



# 旋转
rot_mat = axis_angle_to_matrix(rot_update.squeeze()).to(torch.float32)
# 所有原子减去中心原子坐标,进行旋转 再平移 再加中心原子
rigid_new_pos = (data['ligand'].pos - lig_center.to(device)) @ rot_mat.to(device).T + lig_center.to(device)