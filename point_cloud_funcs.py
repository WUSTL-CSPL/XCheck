import open3d as o3d
import numpy as np
import math
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import point_cloud_utils as pcu
from vtkplotter import *


def loadObject(file_path, thresholds=None):
    
    #load object from file_path
    load_object = load(file_path)

    object_mesh = mesh.Mesh()
    
    #if object is loaded from a volume image or dicom series, isosurface the volume
    if isinstance(load_object, volume.Volume):
        
        #load_object = load_object.gaussianSmooth(sigma=(.6, .6, .6)).medianSmooth(neighbours=(1,1,1))
        
        #extract surface from given threshold values OR use automatic thresholding if no threshold is specified
        if thresholds is not None:
            object_mesh = load_object.isosurface(threshold= thresholds).extractLargestRegion()
        else:
            object_mesh = load_object.isosurface().extractLargestRegion()   
        
        if len(object_mesh.points()) > 1000000:
            object_mesh = object_mesh.decimate(N=100000, method='pro', boundaries=False)

    else:
        object_mesh = load_object.triangulate()
    return object_mesh
    

def generateSamples(mesh, num):
    vertices = mesh.points()
    faces = np.asarray(mesh.faces())
    samples = pcu.sample_mesh_lloyd(vertices, faces, num)
    return samples

def pt_dist(pt_a, pt_b):
    return(np.sqrt(np.sum((pt_a-pt_b)**2, axis=0)))

def load_mesh(file_path, thresholds=None):
    
    #load object from file_path
    load_object = load(file_path)
    object_mesh = mesh.Mesh()
    
    #if object is loaded from a volume image or dicom series, isosurface the volume
    if isinstance(load_object, volume.Volume):

        #load_object = load_object.gaussianSmooth(sigma=(.6, .6, .6)).medianSmooth(neighbours=(1,1,1))
        #extract surface from given threshold values OR use automatic thresholding if no threshold is specified
        if thresholds is not None:
            object_mesh = load_object.isosurface(threshold= thresholds).extractLargestRegion()#.extractLargestRegion()
        else:
            object_mesh = load_object.isosurface().extractLargestRegion()#.extractLargestRegion()

        if len(object_mesh.points()) > 1000000:
            object_mesh = object_mesh.decimate(N=100000, method='pro', boundaries=False)

    else:
        object_mesh = load_object.triangulate()
    
    return object_mesh

def Gamma_sD1D2(dists1, dists2, scale, s_threshold, d_threshold):
    
    d1_list = []
    d2_list = []
    dists1_list = np.where(dists1 > d_threshold)[0]
    dists2_list = np.where(dists2 > d_threshold)[0]
    
    if dists1_list is not None:
        for i in dists1_list:
            d1_list.append(dists1[i])
    if dists2_list is not None:
        for i in dists2_list:
            d2_list.append(dists2[i])
    
    if d1_list:
        max_d1 = max(d1_list)
    else:
        max_D = max(d2_list)

    if d2_list:
        max_d2 = max(d2_list)
    else:
        max_D = max(d1_list)

    if d1_list and d2_list:
        if max_d1 < max_d2:
            max_D = max_d2
        else:
            max_D = max_d1
    factor_D = max_D / d_threshold

    delta_scale = abs(1-scale)
    if delta_scale <= s_threshold:
        factor_s = 1
    else:
        factor_s = delta_scale/s_threshold

    factor_Pv = 1
    Gamma = math.sqrt(factor_s**2 + factor_D**2 + factor_Pv**2)
    print("factor s is: %s"%factor_s)
    print("factor D is: %s"%factor_D)
    print("factor Pv is: %s"%factor_Pv)
    print("Gamma value is: %s"%Gamma)

def Gamma_sDP(dists1, dists2, scale, P, s_threshold, d_threshold, P_threshold):
    
    D1 = np.max(dists1)
    D2 = np.max(dists2)
    factor_D = (D_1+D_2) / d_threshold

    delta_scale = abs(1-scale)
    factor_s = delta_scale/s_threshold

    factor_P = P / P_threshold

    Gamma = math.sqrt(factor_s**2 + factor_D**2 + factor_P**2)
    print("factor s is: %s"%factor_s)
    print("factor D is: %s"%factor_D)
    print("factor P is: %s"%factor_P)
    print("Gamma value is: %s"%Gamma)

def spherical(pt):
    x,y,z       = pt
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arccos(z/r)*(180/ np.pi)
    phi     =  np.arctan2(y,x)*(180/ np.pi)
    return [r,theta,phi]

def delete_origin(intersect_points_all, origin):
    intersect_points = []
    for point in intersect_points_all:
        if pt_dist(point, origin) != 0:
            intersect_points.append(point)
    #print(intersect_points)
    return intersect_points

def point_set_leastdist(origin, intersect_points):
    dists = []
    for intersect_point in intersect_points:
        dist = pt_dist(origin, intersect_point)
        dists.append(dist)
    if dists is not None:
        closest_dist = np.array(dists).min()
        closest_point = intersect_points[dists.index(closest_dist)]

    return closest_point, closest_dist

def fibonacci_sphere(origin, r, samples):

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))
    
    points = origin + np.array(points) * r
    
    return points

def get_boundary(src_mesh, tgt_mesh):

    src_max = np.max(src_mesh.points(), axis=0)
    src_min = np.min(src_mesh.points(), axis=0)
    tgt_max = np.max(tgt_mesh.points(), axis=0)
    tgt_min = np.min(tgt_mesh.points(), axis=0)
    
    cube_max = np.max(np.concatenate([src_mesh.points(), tgt_mesh.points()]), axis=0)
    cube_min = np.min(np.concatenate([src_mesh.points(), tgt_mesh.points()]), axis=0)
    cube_dis = np.sqrt( np.sum( (cube_max-cube_min)**2 ))


    return cube_dis

def furthest_pts(points):
    hullpoints = points
    hdist = []
    if len(points) < 400:
        hdist = cdist(points, points, metric='euclidean')
    else:
        hull = ConvexHull(points)
        hullpoints = points[hull.vertices,:]
        hdist = cdist(hullpoints, hullpoints, metric='euclidean')
    
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

    hull_pts = np.asarray([hullpoints[bestpair[1]],hullpoints[bestpair[0]]])
    hull_pts = hull_pts.astype('float64') 
    hull_pts.view('f8,f8,f8').sort(order=['f1'], axis=0)
    hull_pt1 = hull_pts[0]
    hull_pt2 = hull_pts[1]
    return hull_pt1, hull_pt2

# for overall distance calculation
def calculate_scale(src_samples):
    hull_pt1, hull_pt2 = furthest_pts(src_samples)
    src_dist = pt_dist(hull_pt1, hull_pt2)
    return src_dist

# for global registration
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 5
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_point_clouds(sourcePts, targtPts, voxel_size):

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(sourcePts)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(targtPts)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size, iterations = None):
    distance_threshold = voxel_size * 0.5
    if iterations != None:
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching( source_down, 
            target_down, source_fpfh, 
            target_fpfh,o3d.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold, 
                iteration_number= iterations))
        #iteration_number= 15 modified to iteration_number=iterations

    else:
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching( source_down, 
            target_down, source_fpfh, 
            target_fpfh,o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        
    return result


def perform_global_registration(source, target, source_points, spacing):

    size = spacing/24

    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_point_clouds(source, target, size)

    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh, size)
    global_transformation = result_fast.transformation

    pts_cloud = o3d.geometry.PointCloud()
    pts_cloud.points = o3d.utility.Vector3dVector(source_points)
    
    return_samples = source.transform(global_transformation)
    return_pts = pts_cloud.transform(global_transformation)
    src_samples = np.asarray(return_samples.points)
    src_points = np.asarray(return_pts.points)
    
    return src_samples, src_points, global_transformation
