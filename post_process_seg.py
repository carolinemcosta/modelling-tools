import os.path
import numpy as np


def smooth_seg(segsmooth_bin, original_seg, smooth_seg, tags_list, target_res, fuzz_factor = 1.0, voxel_padding = 10):
  
  ''' Smooth segmentation using segsmooth. Segsmooth is part of the segtools package (MUG). Expects NRRD files.
      
      Parameters: 
        segsmooth_bin (string): path+name of segsmooth binary
        original_seg (string): path+name of original segmentation's NRRD file
        smooth_seg (string): path+name of output smooth segmentation's NRRD file
        tags_list (list): list of tags to be smoothed. Example: [2, 4, 6]
        target_res (float): target resolution in milimeters. Ex. 0.25
        fuzz_factor (float): factor defining how much smoothing is applied. Default = 1.0 (Optional)
        voxel_padding (int): number of voxels to pad the smooth segmentation with. Default = 10 (Optional)
      Returns:
        Writes out a new smooth segmentation to NRRD file
  ''' 
  
  if os.path.isfile(original_seg) and not os.path.isfile(smooth_seg):
    print("Smoothing segmentation...")

    cmd = "%s %s %s -l %s --fuzz-factor %1.1f --voxel-units --padding %d %d %d -r %1.2f %1.2f %1.2f"% 
          (segsmooth_bin, 
           original_seg, 
           smooth_seg, 
           " ".join(str(t) for t in tags_list), 
           fuzz_factor,
           "".join((str(voxel_padding)+' ')*3),
           "".join((str(target_res)+' ')*3))
           
    print(cmd)
    os.system(cmd)      


def tag_base_with_plane(segclip_bin, original_seg, base_seg, plane_name, tags_list, base_tag):
  
  ''' Tag base of segmentation using a plane. 
      Uses segclip, which is part of the segtools package (MUG). 
      Expects NRRD files and a text file with plane definition.
      
      Parameters: 
        segclip_bin (string): path+name of segclip binary
        original_seg (string): path+name of original segmentation's NRRD file
        base_seg (string): path+name of output tagged segmentation's NRRD file
        plane_name (string): path+name of text file containing a plane's origin and normal. 
                             Format: line 1 contains the origin point and line 2 contains the normal vector ("x y z")
        tags_list (list): list of tags to be labeled as base. Example: [2, 4, 6]
        base_tag (int): tag to label the base
      Returns:
        Writes out a new segmentation file (NRRD) with the tagged base
  ''' 
  
  # read plane
  plane = np.loadtxt(plane_name,dtype=float)
  p0 = plane[0,:]
  v  = plane[1,:]

  # tag image
  if os.path.isfile(original_seg) and not os.path.isfile(base_seg):	
    for tag in tags_list:
      print("Tagging base of tag %d..."%tag)

      cmd = "%s %s %s --clip-tag %d --to-tag %d --origin %s --normal %s"%
            (segclip_bin,
            original_seg,
            base_seg,
            tag,
            base_tag,
            " ".join(str(e) for e in p0)
            " ".join(str(e) for e in v))
      
      print(cmd)
      os.system(cmd)
        
          
def add_bath_to_seg(segchangetag_bin, original_seg, bath_seg, bath_tag):

  ''' Add bath to segmentation by changing the "0" tag to a user-defined number. 
      Uses segchangetag, which is part of the segtools package (MUG). 
      Expects NRRD files.
      
      Parameters: 
        segchangetag_bin (string): path+name of segchangetag binary
        original_seg (string): path+name of original segmentation's NRRD file
        bath_seg (string): path+name of output tagged segmentation's NRRD file
        bath_tag (int): tag to label the bath
      Returns:
        Writes out a new segmentation file (NRRD) with a bath with tag bath_tag
  ''' 
  
  # add bath by changing tag
  if os.path.isfile(original_seg) and not os.path.isfile(bath_seg):
    print("Adding bath to original image...")
    cmd = "%s %s %s 0 %d" % (segchangetag_bin, original_seg, bath_seg, bath_tag)
    os.system(cmd)          

