import os
import argparse
import copy
import numpy as np
import open3d as o3d
from point_cloud_funcs import *
from voxelization import *
from vis_open3d import *
from vedo import *
from vtkplotter import *
from RayAnalysis import *
import shutil
from rigid_registration import rigid_registration
import pydicom as dicom
import seaborn as sb

def cleanup(argv):
    # If the output direct already exist under ./Results/, clear its content
    print("Cleaning previous record of {}...".format(argv.outName))

    if os.path.isdir('Results/' + argv.outName):
        shutil.rmtree('Results/' + argv.outName)
    os.mkdir('Results/' + argv.outName)
    os.mkdir('Results/' + argv.outName + '/RayMesh/')
    os.mkdir('Results/' + argv.outName + '/shape_feature/')
    os.mkdir('Results/' + argv.outName + '/shape_feature/origins/')
    os.mkdir('Results/' + argv.outName + '/shape_feature/twomesh_shape/')

def registration(argv):

    cleanup(argv)

    # ========== Registration ========== #
    print('Starting registration...')
    print('-- Source: {}'.format(argv.filePath1))
    print('-- Target: {}'.format(argv.filePath2))

    src_mesh = load_mesh(argv.filePath1, thresholds = argv.thresholds1)
    tgt_mesh = load_mesh(argv.filePath2, thresholds = argv.thresholds2)

    src_mesh_points = src_mesh.points(copy=True)
    tgt_mesh_points = tgt_mesh.points(copy=True)
    
    # Get sample points from surface of mesh
    src_mesh_samples = generateSamples(src_mesh, 2000) 
    tgt_mesh_samples = generateSamples(tgt_mesh, 2000) 

    scale_dist = calculate_scale(src_mesh_samples)

    print('-- Performing Fast Global Registration...')

    spacing = np.mean(distance.pdist(src_mesh_samples))
    src_mesh_samples, src_mesh_points, global_transformation = perform_global_registration(src_mesh_samples, tgt_mesh_samples, src_mesh_points, spacing)

    print('-- Performing Coherent Point Drift...')

    cpd_ittr = 60
    reg = rigid_registration(max_iterations = cpd_ittr, **{ 'X': tgt_mesh_samples, 'Y': src_mesh_samples })

    src_mesh_samples, [s,R,t] = reg.register()
    src_mesh_points = s*np.dot(src_mesh_points, R) + t # Applying transformation

    scale_dist2 = calculate_scale(src_mesh_samples)
    scale = scale_dist2/scale_dist

    gamma_s = abs(1-scale) / float(argv.errorToleranceScale)

    src_mesh.points(src_mesh_points) # Construct aligned src mesh
    tgt_mesh.points(tgt_mesh_points) # Construct aligned tgt mesh

    write(src_mesh, 'Results/'+argv.outName+'/src_mesh.stl')
    write(tgt_mesh, 'Results/'+argv.outName+'/tgt_mesh.stl')

    verify_mesh_src = o3d.io.read_triangle_mesh('Results/'+argv.outName+'/src_mesh.stl')
    verify_mesh_tgt = o3d.io.read_triangle_mesh('Results/'+argv.outName+'/tgt_mesh.stl')
    verify_mesh_src.paint_uniform_color([1, 0, 0])
    verify_mesh_tgt.paint_uniform_color([0, 0, 1])

    print('')
    print('=================================================================================')
    print("IMPORTANT: Registration complete. Press ENTER to visually verify the registration.")
    print("If the registration is correct, close the window to start voxel and ray analysis.")
    print('=================================================================================')
    input()

    o3d.visualization.draw_geometries([verify_mesh_src, verify_mesh_tgt])

    print('')
    print('==========================================================================')
    print("If registration is correct, press ENTER to proceed voxel and ray analysis.")
    print("Otherwise, enter (r) to restart registration.")
    print('==========================================================================')
    val = input()

    return val, gamma_s, src_mesh, tgt_mesh

def load_scan(path):
    slices = [dicom.read_file(path+'/'+s, force=True) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Set outlier to be 0
    image[image == -2000] = 0

    # Transform to HU
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def filter_voxels_hu(value_array):
    value_array = value_array.flatten()
    min_value = -800
    max_value = 800
    del_index_1 = np.where(value_array < min_value)
    value_array = np.delete(value_array, del_index_1)
    del_index_2 = np.where(value_array > max_value)
    value_array = np.delete(value_array, del_index_2)
    return value_array

def find_maximum(name):
    x,y = sb.distplot(name).get_lines()[0].get_data()
    index_max = np.argmax(y)
    x = np.asarray(x)
    y = np.asarray(y)
    sum_array = x*y
    return [x[index_max], y[index_max], sum_array.sum()]

def get_refernce_data(name):
    if 'bone' in name.lower():
        return 1, [-699.02427384315600000, 0.00290340242208494, -116.38059884631300000]
    elif 'chip' in name.lower():
        return 1, [155.53092459487700000, 0.00305289586959092, -10.15296808637010000]
    elif 'screw' in name.lower():
        return 1, [122.385352302532, 0.00261942987517301, -21.6053416746393]
    elif 'guide' in name.lower():
        return 5, [137.4120214940133,0.0031574425132645627,-11.119509504055227]
    else:
        print("Cannot compare to existing data, please make sure outName contains 'Bone', 'Screw', 'Chip', or 'Guide'")
        exit()
 
def material_analysis(path, outname):
    if os.path.isdir(path):
        model = load_scan(path)
        model_pixels = get_pixels_hu(model)
        model_pixels = filter_voxels_hu(model_pixels)
        np.savetxt("./Results/{}/CTValues.txt".format(outname), model_pixels)

        v1 = find_maximum(model_pixels)
        scale, v2 = get_refernce_data(outname)

        for i in range(0,3):
            v1[i] = v1[i] / v2[i]
        distance = np.linalg.norm(v1 - np.array((1,1,1))) * scale

        return distance
    else:
        print("Cannot conduct material analysis, f1 is not a directory.")
        return -1

def check_args(argv):
    # Check arguments are entered either for basic or complete test mode
    if argv.basicTest is not None:
        return True
    else:
        if argv.filePath1 is not None\
            and argv.filePath2 is not None\
                and argv.outName is not None\
                    and argv.errorToleranceDist is not None\
                        and argv.errorToleranceScale is not None\
                            and argv.errorToleranceVolume is not None\
                                and argv.errorToleranceMaterial is not None:
            return True
    return False

def aggregate_gamma(gamma_s, gamma_d, gamma_v, gamma_m):
    gamma_s = max(1, gamma_s)
    gamma_d = max(1, gamma_d)
    gamma_v = max(1, gamma_v)
    gamma_m = max(1, gamma_m)
    gamma = math.sqrt(gamma_s ** 2 + gamma_d ** 2 + gamma_v ** 2 + gamma_m ** 2)
    return gamma

def main(argv):

    if not check_args(argv):
        print('')
        print("Checking arguments failed, please make sure the arguments are entered correctly.")
        print("- For basic test mode please run:")
        print("-- python3 run.py -basic <result_dir>")
        print("- For complete test mode please run:")
        print("-- python3 run.py -f1 <CT_scan_dir> -f2 <original_model_dir> -o <result_dir> -etdist <error_tolerance_distance> -ets <error_tolerance_scale> -etg <error_tolerance_group> -etm <error_tolerance_material>")
        exit()
    
    if argv.basicTest is not None:
        # Running basic test 
        if not os.path.isdir("Results/"+argv.basicTest):
            print("Basic test mode cannot find the entered result name, please check under ./Results to see if the entered result name exist.")
            exit()

        # Load existing record of analysis
        src = o3d.io.read_point_cloud("Results/"+argv.basicTest+"/src.pcd")
        tgt = o3d.io.read_point_cloud("Results/"+argv.basicTest+"/tgt.pcd")

        src_dist = np.load("Results/"+argv.basicTest+"/src_dist.npy")
        tgt_dist = np.load("Results/"+argv.basicTest+"/tgt_dist.npy")

        alpha_shape = []

        if not os.path.exists("Results/"+argv.basicTest+"/RayMesh/"):
            os.mkdir("Results/"+argv.basicTest+"/RayMesh/")

        for file in os.listdir("Results/"+argv.basicTest+"/RayMesh/"):
            alpha_shape += [o3d.io.read_triangle_mesh("Results/"+argv.basicTest+"/RayMesh/"+file)]
        alpha_shape += [o3d.io.read_triangle_mesh("Results/"+argv.basicTest+"/tgt_mesh.stl")]
        
        with open("Results/"+argv.basicTest+"/gamma.txt", 'r') as f:
            for line in f.readlines():
                print(line.rstrip())

        # Visualize
        visualizer_app(src, tgt, src_dist, tgt_dist, alpha_shape)

    else:    
        # Running complete test 
        val = 'r'

        while val == 'r':
            val, gamma_s, src_mesh, tgt_mesh = registration(argv)

        print("-- Registration is verified, checkpoint saved.")

        # ========== Voxel Analysis ========== #

        print('Starting voxel analysis, this may take a while...')

        src_voxels_carved, src_carved_voxels_pc = getVoxelPC('Results/'+argv.outName+'/src_mesh.stl', cubic_size = 2.0, voxel_resolution = 400.0, thresholds = None)
        tgt_voxels_carved, tgt_carved_voxels_pc = getVoxelPC('Results/'+argv.outName+'/tgt_mesh.stl', cubic_size = 2.0, voxel_resolution = 400.0, thresholds = None)

        src_voxel_points = copy.deepcopy(np2Pcd(src_carved_voxels_pc))
        tgt_voxel_points = copy.deepcopy(np2Pcd(tgt_carved_voxels_pc))

        print('-- Voxel carved, voxelized point cloud acquired')
        
        voxel_size = 4.0 
        src_voxel_down = src_voxel_points.voxel_down_sample(voxel_size)
        tgt_voxel_down = tgt_voxel_points.voxel_down_sample(voxel_size)

        print('-- Computing Hausdorff Distance...')

        # Hausdorff distance
        src_dist = np.asarray( src_voxel_down.compute_point_cloud_distance(tgt_voxel_down) )
        tgt_dist = np.asarray( tgt_voxel_down.compute_point_cloud_distance(src_voxel_down) )

        # Concatenate the two point clouds and find the largest distance
        all_dist = np.concatenate((src_dist, tgt_dist))
        max_dist = np.amax(all_dist)

        gamma_d = max_dist / float(argv.errorToleranceDist) / voxel_size / 2

        src_dist = src_dist / voxel_size
        tgt_dist = tgt_dist / voxel_size

        print("-- Voxel analysis complete.")

        # ========== Ray Analysis ========== #
        print('Starting ray analysis...')

        gamma_v = RayAnalysis(src_mesh, tgt_mesh, float(argv.errorToleranceDist), float(argv.errorToleranceVolume), argv.outName)

        alpha_shape = []
        for file in os.listdir("Results/"+argv.outName+"/RayMesh/"):
            alpha_shape += [o3d.io.read_triangle_mesh("Results/"+argv.outName+"/RayMesh/"+file)]
        alpha_shape += [o3d.io.read_triangle_mesh(argv.filePath2)]

        o3d.io.write_point_cloud("Results/"+argv.outName+"/src.pcd", src_voxel_down)
        o3d.io.write_point_cloud("Results/"+argv.outName+"/tgt.pcd", tgt_voxel_down)
        np.save("Results/"+argv.outName+"/src_dist.npy", src_dist)
        np.save("Results/"+argv.outName+"/tgt_dist.npy", tgt_dist)

        print("-- Ray analysis complete.")

        # ========== Material Analysis ========== #
        print('Starting Material analysis...')
        material_dist = material_analysis(argv.filePath1, argv.outName)
        if material_dist == -1:
            gamma_m = 0
        else:
            gamma_m = material_dist / float(argv.errorToleranceMaterial) / 0.8

        print("-- Material analysis complete.")

        gamma = aggregate_gamma(gamma_s, gamma_d, gamma_v, gamma_m)

        print("All analysis complete")
        print("[Computed Gamma_s: {}]".format(gamma_s))
        print("[Computed Gamma_d: {}]".format(gamma_d))
        print("[Computed Gamma_v: {}]".format(gamma_v))
        print("[Computed Gamma_m: {}]".format(gamma_m))
        print("[Total Computed Gamma: {}]".format(gamma))
        print("[Final Decision: {}]".format('Malicious' if gamma > 2 else 'Benign'))

        with open('Results/' + argv.outName + '/gamma.txt', 'w') as f:
            f.write("[Computed Gamma_s: {}]\n".format(gamma_s))
            f.write("[Computed Gamma_d: {}]\n".format(gamma_d))
            f.write("[Computed Gamma_v: {}]\n".format(gamma_v))
            f.write("[Computed Gamma_m: {}]\n".format(gamma_m))
            f.write("[Total Computed Gamma: {}]\n".format(gamma))
            f.write("[Final Decision: {}]".format('Malicious' if gamma > 2 else 'Benign'))

        input("Press ENTER to visualize...")


        visualizer_app(src_voxel_down, tgt_voxel_down, src_dist, tgt_dist, alpha_shape)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compares two objects to each other for geometric similarity. Accepts inputs of volume images, dicom series or mesh object. Input objects do not have to be the same type.')
    
    parser.add_argument('-f1', '--filePath1', type=str, help = "File path to input object: Accepts volume image (.tiff, .vti, .slc etc...), directory containing Dicom series, or mesh objects (.stl, .obj, .ply etc...)")
    
    parser.add_argument('-f2', '--filePath2', type=str, help = "File path to original object: Accepts volume image (.tiff, .vti, .slc etc...), directory containing Dicom series, or mesh objects (.stl, .obj, .ply etc...)")

    parser.add_argument('-o', '--outName', type=str)
    
    parser.add_argument('-t1', '--thresholds1', nargs='+', type=int, help = "Optional: Dual threshold values for feature extraction for input object. Ex: -200, 100, -500, 200")
    
    parser.add_argument('-t2', '--thresholds2', nargs='+', type=int, help = "Optional: Dual threshold values for feature extraction for original object. Ex: -200, 100, -500, 200")
    
    parser.add_argument('-etdist', '--errorToleranceDist', type=float, help = "Provide a maximum error tolerance for distance.")

    parser.add_argument('-ets', '--errorToleranceScale', type=float, help = "Provide a maximum error tolerance for delta scale.")

    parser.add_argument('-etg', '--errorToleranceVolume', type=float, help = "Provide a maximum error tolerance for delta group.")

    parser.add_argument('-etm', '--errorToleranceMaterial', type=float, help = "Provide a maximum error tolerance for delta material.")

    parser.add_argument('-basic', '--basicTest', type=str, help = "Optional: Run basic tests without running the entire system.")

    main(parser.parse_args())