import os
from easy_mesh_vtk import *

if __name__ == "__main__":

  num_samples = 30 # need to define
  vtk_path = '' # need to define
  output_save_path = './augmentation_vtk_data/'
  if not os.path.exists(output_save_path):
      os.mkdir(output_save_path)

  # sample 1-24 as train and validation samples, do augmentation
  sample_list = list(range(1, num_samples+1))
  num_augmentations = 20

  for i_sample in sample_list:
      for i_aug in range(num_augmentations):

          file_name = 'Sample_0{0}_d.vtp'.format(i_sample)
          output_file_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample)
          vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                  translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                  scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]) #use default random setting
          mesh = Easy_Mesh(os.path.join(vtk_path, file_name))
          mesh.mesh_transform(vtk_matrix)
          mesh.to_vtp(os.path.join(output_save_path, output_file_name))

      # flipped meshes
      for i_aug in range(num_augmentations):
          file_name = 'Sample_0{0}_d.vtp'.format(i_sample+1000)
          output_file_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample+1000)
          vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                  translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                  scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]) #use default random setting
          mesh = Easy_Mesh(os.path.join(vtk_path, file_name))
          mesh.mesh_transform(vtk_matrix)
          mesh.to_vtp(os.path.join(output_save_path, output_file_name))
