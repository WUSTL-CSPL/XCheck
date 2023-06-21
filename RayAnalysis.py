import os
import numpy as np
from scipy.spatial import distance
from vtkplotter import *
from vtkplotter.plotter import *
from point_cloud_funcs import *


def rays_normalization(origin, points, threshold):
    ray_dist = dict()
    ray_sum = 0
    points_new = list()
    ray_number = int(points.size/3)
    origin_new = np.mean(points, axis=0)

    if threshold is not None:
        for i in range(ray_number):
            ray_dist[i] = pt_dist(points[i],origin_new)
            if ray_dist[i] <= threshold:
                points_new.append(points[i])
    else:
        for i in range(ray_number):
            ray_dist[i] = pt_dist(points[i],origin_new)
            ray_sum = ray_sum + ray_dist[i]
        ray_average = ray_sum/ray_number
        for i in range(ray_number):
            if ray_dist[i] <= ray_average:
                points_new.append(points[i])
    return points_new


def RayAnalysis(src_mesh, tgt_mesh, errorToleranceDist, errorToleranceVolume, outName):

    src_points = src_mesh.points(copy=True)
    tgt_points = tgt_mesh.points(copy=True)
    
    #get sample points from surface of mesh
    src_samples = generateSamples(src_mesh, 6000) 
    tgt_samples = generateSamples(tgt_mesh, 6000) 

    ratio = 2000/min(len(tgt_points), len(src_points))
    spacing = np.mean(distance.pdist(src_samples))
    spacing = 3*spacing*ratio
    dists1 = distance.cdist(src_points,tgt_samples).min(axis=1)
    dists2 = distance.cdist(tgt_points,src_samples).min(axis=1)

    cube_dis = get_boundary(src_mesh, tgt_mesh)

    origins_list = np.where(dists2 > errorToleranceDist)[0]
    origins = []

    print('-- Areas of interest identified, iteratively analyzing now...')
    counter = 0

    for origin_list in origins_list:

        counter += 1
        print('---- Analyzing {} of {}'.format(counter, len(origins_list)))

        points_twomesh = []

        origin = tgt_points[origin_list]
        origins.append(origin)

        points = fibonacci_sphere(origin, 0.5*cube_dis, 100)

        for point in points:

            src_intersect_points_all = src_mesh.intersectWithLine(origin, point)
            tgt_intersect_points_all = tgt_mesh.intersectWithLine(origin, point)
            src_intersect_points = delete_origin(src_intersect_points_all, origin)
            tgt_intersect_points = delete_origin(tgt_intersect_points_all, origin)

            if np.array(src_intersect_points).size == 0:
                src_closest_point = origin
                src_closest_dist = 0
            elif np.array(src_intersect_points).size == 3:
                src_closest_point = src_intersect_points[0]
                src_closest_dist = pt_dist(src_intersect_points[0], origin)
            else:               
                src_closest_point, src_closest_dist = point_set_leastdist(origin, src_intersect_points)

            if np.array(tgt_intersect_points).size == 0:
                tgt_closest_point = origin
                tgt_closest_dist = 0
            elif np.array(tgt_intersect_points).size == 3:
                tgt_closest_point = tgt_intersect_points[0]
                tgt_closest_dist = pt_dist(tgt_intersect_points[0], origin)
            else:               
                tgt_closest_point, tgt_closest_dist = point_set_leastdist(origin, tgt_intersect_points)

            if abs(src_closest_dist-tgt_closest_dist) != 0:
                if src_closest_dist < tgt_closest_dist:
                    points_twomesh.append(src_closest_point)
                elif src_closest_dist > tgt_closest_dist:
                    points_twomesh.append(tgt_closest_point)

        number = np.where(origins_list==origin_list)[0][0]

        np.savetxt('Results/' + outName + '/shape_feature/origins/origins.txt', origins)
        np.savetxt('Results/' + outName + '/shape_feature/twomesh_shape/points_'+str(number)+'.txt', points_twomesh)

    tgt_mesh = load('Results/'+outName+'/tgt_mesh.stl').alpha(0.2).c('ivory') 

    if not os.path.isfile('Results/' + outName + '/shape_feature/origins/origins.txt'):
        print("-- Percentage of malicious region: 0")
        return 0

    print('-- Mapping areas of interest to target...')

    origins = np.loadtxt('Results/' + outName + '/shape_feature/origins/origins.txt')
    origin_number = int(np.size(origins)/3)

    hull = dict()
    Merged_list = list()
    volume_all = 0
    
    for i in range(origin_number):
        if i in Merged_list:
            continue
        else:
            points = np.loadtxt('Results/' + outName + '/shape_feature/twomesh_shape/points_%s'%i+'.txt')
            if points.size >= 240: # More than 80/100 rays. Wipe out the surface discrepancy
                for j in range(i+1,origin_number):
                    points_2BMerge = np.loadtxt('Results/' + outName + '/shape_feature/twomesh_shape/points_%s'%j+'.txt')
                    if pt_dist(origins[i], origins[j])<3:   #threshold here
                        points = np.append(points, points_2BMerge, axis=0)  #merge two point clouds
                        Merged_list.append(j)
                points = rays_normalization(origins[i], points, 1.6)   #rays normalization for better visualization
                hull[i] = shapes.convexHull(points).c('red') #color of discrepancy here
                volume_all = volume_all + hull[i].volume()

                write(hull[i], "Results/"+outName+"/RayMesh/mesh"+str(i)+".stl")

    print("-- Percentage of malicious region: ", volume_all/tgt_mesh.volume())
    malicious_percentage = volume_all/tgt_mesh.volume()
    gamma_v = malicious_percentage / errorToleranceVolume
    return gamma_v