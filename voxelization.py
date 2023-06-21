import open3d as o3d
import numpy as np

def np2Pcd(pts):
    pts = np.asarray(pts)
    color = np.random.rand(pts.shape[0], pts.shape[1])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]


def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                        [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
                        [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x)


def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0, 0, 2]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans


def preprocess(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= center
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model


def voxel_carving(mesh, cubic_size, voxel_resolution, w=300, h=300):
    mesh.compute_vertex_normals()
    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)

    # Setup dense voxel grid.
    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size,
        height=cubic_size,
        depth=cubic_size,
        voxel_size=cubic_size / voxel_resolution,
        origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],
        color=[0.0, 0.2, 1.0])

    # Rescale geometry.
    camera_sphere = preprocess(camera_sphere)
    mesh = preprocess(mesh)

    # Setup visualizer to render depthmaps.
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # Carve voxel grid.
    for xyz in camera_sphere.vertices:
        # Get new camera pose.
        trans = get_extrinsic(xyz)
        param.extrinsic = trans
        ctr.convert_from_pinhole_camera_parameters(param)

        # Capture depth image and make a point cloud.
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)

        # Depth map carving method.
        voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)
    vis.destroy_window()

    return voxel_carving

def getVoxelPC(path, cubic_size = 2.0, voxel_resolution = 128.0, thresholds = None):
    mesh = o3d.io.read_triangle_mesh(path)
    carved_voxels = voxel_carving(mesh, cubic_size, voxel_resolution)
    carved_voxels_pts = np.stack(list(vx.grid_index for vx in carved_voxels.get_voxels()))
    return carved_voxels, carved_voxels_pts