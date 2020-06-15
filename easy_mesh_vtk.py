import sys
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from scipy.spatial import distance_matrix
from sklearn import svm
import math

class Easy_Mesh(object):
    def __init__(self, filename = None, warning=False):
        #initialize
        self.warning = warning
        self.reader = None
        self.vtkPolyData = None
        self.cells = np.array([])
        self.cell_ids = np.array([])
        self.points = np.array([])
        self.point_attributes = dict()
        self.cell_attributes = dict()
        self.filename = filename
        if self.filename != None:
            if self.filename[-3:].lower() == 'vtp':
                self.read_vtp(self.filename)
            elif self.filename[-3:].lower() == 'stl':
                self.read_stl(self.filename)
            elif self.filename[-3:].lower() == 'obj':
                self.read_obj(self.filename)
            else:
                if self.warning:
                    print('Not support file type')

    
    def get_mesh_data_from_vtkPolyData(self):
        data = self.vtkPolyData
        
        n_triangles = data.GetNumberOfCells()
        n_points = data.GetNumberOfPoints()
        mesh_triangles = np.zeros([n_triangles, 9], dtype='float32')
        mesh_triangle_ids = np.zeros([n_triangles, 3], dtype='int32')
        mesh_points = np.zeros([n_points, 3], dtype='float32')
    
        for i in range(n_triangles):
            mesh_triangles[i][0], mesh_triangles[i][1], mesh_triangles[i][2] = data.GetPoint(data.GetCell(i).GetPointId(0))
            mesh_triangles[i][3], mesh_triangles[i][4], mesh_triangles[i][5] = data.GetPoint(data.GetCell(i).GetPointId(1))
            mesh_triangles[i][6], mesh_triangles[i][7], mesh_triangles[i][8] = data.GetPoint(data.GetCell(i).GetPointId(2))
            mesh_triangle_ids[i][0] = data.GetCell(i).GetPointId(0)
            mesh_triangle_ids[i][1] = data.GetCell(i).GetPointId(1)
            mesh_triangle_ids[i][2] = data.GetCell(i).GetPointId(2)
    
        for i in range(n_points):
            mesh_points[i][0], mesh_points[i][1], mesh_points[i][2] = data.GetPoint(i)
        
        self.cells = mesh_triangles
        self.cell_ids = mesh_triangle_ids
        self.points = mesh_points
        
        #read point arrays
        for i_attribute in range(self.vtkPolyData.GetPointData().GetNumberOfArrays()):
#            print(self.vtkPolyData.GetPointData().GetArrayName(i_attribute))
#            print(self.vtkPolyData.GetPointData().GetArray(i_attribute).GetNumberOfComponents())
            self.load_point_attributes(self.vtkPolyData.GetPointData().GetArrayName(i_attribute), self.vtkPolyData.GetPointData().GetArray(i_attribute).GetNumberOfComponents())
        
        #read cell arrays
        for i_attribute in range(self.vtkPolyData.GetCellData().GetNumberOfArrays()):
#            print(self.vtkPolyData.GetCellData().GetArrayName(i_attribute))
#            print(self.vtkPolyData.GetCellData().GetArray(i_attribute).GetNumberOfComponents())
            self.load_cell_attributes(self.vtkPolyData.GetCellData().GetArrayName(i_attribute), self.vtkPolyData.GetCellData().GetArray(i_attribute).GetNumberOfComponents())
        
        
    def read_stl(self, stl_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.cells
            self.cell_ids
            self.points
            self.cell_attributes
            self.point_attributes
        '''
#        self.filename = stl_filename
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_filename)
        reader.Update()
        self.reader = reader
        
        data = reader.GetOutput()
        self.vtkPolyData = data        
        self.get_mesh_data_from_vtkPolyData()
        
        
    def read_obj(self, obj_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.cells
            self.cell_ids
            self.points
            self.cell_attributes
            self.point_attributes
        '''
#        self.filename = obj_filename
        reader = vtk.vtkOBJReader()
        reader.SetFileName(obj_filename)
        reader.Update()
        self.reader = reader
        
        data = reader.GetOutput()
        self.vtkPolyData = data
        self.get_mesh_data_from_vtkPolyData()
    
    
    def read_vtp(self, vtp_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.cells
            self.cell_ids
            self.points
            self.cell_attributes
            self.point_attributes
        '''
#        self.filename = vtp_filename
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_filename)
        reader.Update()
        self.reader = reader
    
        data = reader.GetOutput()
        self.vtkPolyData = data        
        self.get_mesh_data_from_vtkPolyData()
        
    
    def load_point_attributes(self, attribute_name, dim):
        self.point_attributes[attribute_name] = np.zeros([self.points.shape[0], dim])
        try:
            if dim == 1:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetValue(i)
            elif dim == 2:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 0)
                    self.point_attributes[attribute_name][i, 1] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 1)
            elif dim == 3:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 0)
                    self.point_attributes[attribute_name][i, 1] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 1)
                    self.point_attributes[attribute_name][i, 2] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 2)
        except:
            if self.warning:
                print('No cell attribute named "{0}" in file: {1}'.format(attribute_name, self.filename))
        
    
    def get_point_curvatures(self, method='mean'):
        curv = vtk.vtkCurvatures()
        curv.SetInputData(self.vtkPolyData)
        if method == 'mean':
            curv.SetCurvatureTypeToMean()
        elif method == 'max':
            curv.SetCurvatureTypeToMaximum()
        elif method == 'min':
            curv.SetCurvatureTypeToMinimum()
        elif method == 'Gaussian':
            curv.SetCurvatureTypeToGaussian()
        else:
            curv.SetCurvatureTypeToMean()
        curv.Update()
        
        n_points = self.vtkPolyData.GetNumberOfPoints()
        self.point_attributes['Curvature'] = np.zeros([n_points, 1])
        for i in range(n_points):
            self.point_attributes['Curvature'][i] = curv.GetOutput().GetPointData().GetArray(0).GetValue(i)
            
    
    def get_cell_curvatures(self, method='mean'):
        self.get_point_curvatures(method=method)
        self.cell_attributes['Curvature'] = np.zeros([self.cells.shape[0], 1])
        for i_cell in range(self.cells.shape[0]):
            p_idx = self.cell_ids[i_cell][:]
            p_curv = self.point_attributes['Curvature'][p_idx]
            self.cell_attributes['Curvature'][i_cell] = np.mean(p_curv)            
    
        
    def load_cell_attributes(self, attribute_name, dim):
        self.cell_attributes[attribute_name] = np.zeros([self.cells.shape[0], dim])
        try:
            if dim == 1:
                for i in range(self.cells.shape[0]):
                    self.cell_attributes[attribute_name][i, 0] = self.vtkPolyData.GetCellData().GetArray(attribute_name).GetValue(i)
            elif dim == 2:
                for i in range(self.cells.shape[0]):
                    self.cell_attributes[attribute_name][i, 0] = self.vtkPolyData.GetCellData().GetArray(attribute_name).GetComponent(i, 0)
                    self.cell_attributes[attribute_name][i, 1] = self.vtkPolyData.GetCellData().GetArray(attribute_name).GetComponent(i, 1)
            elif dim == 3:
                for i in range(self.cells.shape[0]):
                    self.cell_attributes[attribute_name][i, 0] = self.vtkPolyData.GetCellData().GetArray(attribute_name).GetComponent(i, 0)
                    self.cell_attributes[attribute_name][i, 1] = self.vtkPolyData.GetCellData().GetArray(attribute_name).GetComponent(i, 1)
                    self.cell_attributes[attribute_name][i, 2] = self.vtkPolyData.GetCellData().GetArray(attribute_name).GetComponent(i, 2)
        except:
            if self.warning:
                print('No cell attribute named "{0}" in file: {1}'.format(attribute_name, self.filename))
         
            
#    def set_cell_labels(self, given_labels):
#        '''
#        update:
#            self.cell_attributes['Label']
#        '''
#        self.cell_attributes['Label'] = np.zeros([self.cell_ids.shape[0], 1])
#        self.cell_attributes['Label'] = given_labels
        
    
    def set_cell_labels(self, label_dict, tol=0.01):
        '''
        update:
            self.cell_attributes['Label']
        '''
        self.cell_attributes['Label'] = np.zeros([self.cell_ids.shape[0], 1])
        
        cell_centers = (self.cells[:, 0:3] + self.cells[:, 3:6] + self.cells[:, 6:9]) / 3.0
        for i_label in label_dict:
            i_label_cell_centers = (label_dict[i_label][:, 0:3] + label_dict[i_label][:, 3:6] + label_dict[i_label][:, 6:9]) / 3.0
            D = distance_matrix(cell_centers, i_label_cell_centers)
            
            if len(np.argwhere(D<=tol)) > i_label_cell_centers.shape[0]:
                sys.exit('tolerance ({0}) is too large, please adjust.'.format(tol))
            elif len(np.argwhere(D<=tol)) < i_label_cell_centers.shape[0]:
                sys.exit('tolerance ({0}) is too small, please adjust.'.format(tol))
            else:
                for i in range(i_label_cell_centers.shape[0]):
                    label_id = np.argwhere(D<=tol)[i][0]
                    self.cell_attributes['Label'][label_id, 0] = int(i_label)
        
            
    def get_cell_edges(self):
        '''
        update:
            self.cell_attributes['Edge']
        '''
        self.cell_attributes['Edge'] = np.zeros([self.cell_ids.shape[0], 3])
    
        for i_count in range(self.cell_ids.shape[0]):
            v1 = self.points[self.cell_ids[i_count, 0], :] - self.points[self.cell_ids[i_count, 1], :]
            v2 = self.points[self.cell_ids[i_count, 1], :] - self.points[self.cell_ids[i_count, 2], :]
            v3 = self.points[self.cell_ids[i_count, 0], :] - self.points[self.cell_ids[i_count, 2], :]
            self.cell_attributes['Edge'][i_count, 0] = np.linalg.norm(v1)
            self.cell_attributes['Edge'][i_count, 1] = np.linalg.norm(v2)
            self.cell_attributes['Edge'][i_count, 2] = np.linalg.norm(v3)
            
            
    def get_cell_normals(self):
        data = self.vtkPolyData
        n_triangles = data.GetNumberOfCells()
        #normal
        v1 = np.zeros([n_triangles, 3], dtype='float32')
        v2 = np.zeros([n_triangles, 3], dtype='float32')
        v1[:, 0] = self.cells[:, 0] - self.cells[:, 3]
        v1[:, 1] = self.cells[:, 1] - self.cells[:, 4]
        v1[:, 2] = self.cells[:, 2] - self.cells[:, 5]
        v2[:, 0] = self.cells[:, 3] - self.cells[:, 6]
        v2[:, 1] = self.cells[:, 4] - self.cells[:, 7]
        v2[:, 2] = self.cells[:, 5] - self.cells[:, 8]
        mesh_normals = np.cross(v1, v2)
        self.cell_attributes['Normal'] = mesh_normals
        
        
    def compute_guassian_heatmap(self, landmark, sigma = 10.0, height = 1.0):
        '''
        inputs:
            landmark: np.array [1, 3]
            sigma (default=10.0)
            height (default=1.0)
        update:
            self.cell_attributes['heatmap']
        '''
        cell_centers = (self.cells[:, 0:3] + self.cells[:, 3:6] + self.cells[:, 6:9]) / 3.0
        heatmap = np.zeros([cell_centers.shape[0], 1])
    
        for i_cell in range(len(cell_centers)):
            delx = cell_centers[i_cell, 0] - landmark[0]
            dely = cell_centers[i_cell, 1] - landmark[1]
            delz = cell_centers[i_cell, 2] - landmark[2]
            heatmap[i_cell, 0] = height*math.exp(-1*(delx*delx+dely*dely+delz*delz)/2.0/sigma/sigma)
        self.cell_attributes['Heatmap'] = heatmap
        
    
    def compute_displacement_map(self, landmark):
        '''
        inputs:
            landmark: np.array [1, 3]
        update:
            self.cell_attributes['Displacement map']
        '''
        cell_centers = (self.cells[:, 0:3] + self.cells[:, 3:6] + self.cells[:, 6:9]) / 3.0
        displacement_map = np.zeros([cell_centers.shape[0], 3])
    
        for i_cell in range(len(cell_centers)):
            delx = cell_centers[i_cell, 0] - landmark[0]
            dely = cell_centers[i_cell, 1] - landmark[1]
            delz = cell_centers[i_cell, 2] - landmark[2]
            displacement_map[i_cell, 0] = delx
            displacement_map[i_cell, 1] = dely
            displacement_map[i_cell, 2] = delz
        self.cell_attributes['Displacement_map'] = displacement_map
        
    
    def compute_cell_attributes_by_svm(self, given_cells, given_cell_attributes, attribute_name):
        '''
        inputs:
            given_cells: [n, 9] numpy array
            given_cell_attributes: [n, 1] numpy array
        update:
            self.cell_attributes[attribute_name]
        '''
        if given_cell_attributes.shape[1] == 1:
            self.cell_attributes[attribute_name] = np.zeros([self.cells.shape[0], 1])
            clf = svm.SVC()
            clf.fit(given_cells, given_cell_attributes)
            self.cell_attributes[attribute_name][:, 0] = clf.predict(self.cells)
        else:
            print('Only support 1D attribute')
            
            
    def update_cell_ids_and_points(self):
        '''
        call when self.cells is modified
        update
            self.cell_ids
            self.points
        '''
        self.points = self.cells.reshape([int(self.cells.shape[0]*3), 3])
        self.points = np.unique(self.points, axis=0)
        self.cell_ids = np.zeros([self.cells.shape[0], 3], dtype='int64')
    
        for i_count in range(self.cells.shape[0]):
            counts0 = np.bincount(np.where(self.points==self.cells[i_count, 0:3])[0])
            counts1 = np.bincount(np.where(self.points==self.cells[i_count, 3:6])[0])
            counts2 = np.bincount(np.where(self.points==self.cells[i_count, 6:9])[0])
            self.cell_ids[i_count, 0] = np.argmax(counts0)
            self.cell_ids[i_count, 1] = np.argmax(counts1)
            self.cell_ids[i_count, 2] = np.argmax(counts2)
        
        if self.warning:
            print('Warning! self.cell_attributes are reset and need to be updated!')
        self.cell_attributes = dict() #reset
        self.point_attributes = dict() #reset
        self.update_vtkPolyData()
            
            
    def update_vtkPolyData(self):
        '''
        call this function when manipulating self.cells, self.cell_ids, or self.points
        '''
        vtkPolyData = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
    
        points.SetData(numpy_to_vtk(self.points))
        cells.SetCells(len(self.cell_ids),
                       numpy_to_vtkIdTypeArray(np.hstack((np.ones(len(self.cell_ids))[:, None] * 3,
                                                          self.cell_ids)).astype(np.int64).ravel(),
                                               deep=1))
        vtkPolyData.SetPoints(points)
        vtkPolyData.SetPolys(cells)
        
        #update point_attributes
        for i_key in self.point_attributes.keys():
            point_attribute = vtk.vtkDoubleArray()
            point_attribute.SetName(i_key);
            if self.point_attributes[i_key].shape[1] == 1:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple1(i_attribute)
                vtkPolyData.GetPointData().AddArray(point_attribute)
#                vtkPolyData.GetPointData().SetScalars(cell_attribute)
            elif self.point_attributes[i_key].shape[1] == 2:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple2(i_attribute[0], i_attribute[1])
                vtkPolyData.GetPointData().AddArray(point_attribute)
#                vtkPolyData.GetPointData().SetVectors(cell_attribute)
            elif self.point_attributes[i_key].shape[1] == 3:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple3(i_attribute[0], i_attribute[1], i_attribute[2])
                vtkPolyData.GetPointData().AddArray(point_attribute)
#                vtkPolyData.GetPointData().SetVectors(cell_attribute)
            else:
                if self.warning:
                    print('Check attribute dimension, only support 1D, 2D, and 3D now')
        
        #update cell_attributes
        for i_key in self.cell_attributes.keys():
            cell_attribute = vtk.vtkDoubleArray()
            cell_attribute.SetName(i_key);
            if self.cell_attributes[i_key].shape[1] == 1:
                cell_attribute.SetNumberOfComponents(self.cell_attributes[i_key].shape[1])
                for i_attribute in self.cell_attributes[i_key]:
                    cell_attribute.InsertNextTuple1(i_attribute)
                vtkPolyData.GetCellData().AddArray(cell_attribute)
#                vtkPolyData.GetCellData().SetScalars(cell_attribute)
            elif self.cell_attributes[i_key].shape[1] == 2:
                cell_attribute.SetNumberOfComponents(self.cell_attributes[i_key].shape[1])
                for i_attribute in self.cell_attributes[i_key]:
                    cell_attribute.InsertNextTuple2(i_attribute[0], i_attribute[1])
                vtkPolyData.GetCellData().AddArray(cell_attribute)
#                vtkPolyData.GetCellData().SetVectors(cell_attribute)
            elif self.cell_attributes[i_key].shape[1] == 3:
                cell_attribute.SetNumberOfComponents(self.cell_attributes[i_key].shape[1])
                for i_attribute in self.cell_attributes[i_key]:
                    cell_attribute.InsertNextTuple3(i_attribute[0], i_attribute[1], i_attribute[2])
                vtkPolyData.GetCellData().AddArray(cell_attribute)
#                vtkPolyData.GetCellData().SetVectors(cell_attribute)
            else:
                if self.warning:
                    print('Check attribute dimension, only support 1D, 2D, and 3D now')
        
        vtkPolyData.Modified()
        self.vtkPolyData = vtkPolyData
       
    
    def mesh_decimation(self, reduction_rate):
        decimate_reader = vtk.vtkQuadricDecimation()
        decimate_reader.SetInputData(self.vtkPolyData)
        decimate_reader.SetTargetReduction(reduction_rate)
        decimate_reader.VolumePreservationOn()
        decimate_reader.Update()
        self.vtkPolyData = decimate_reader.GetOutput()
        self.get_mesh_data_from_vtkPolyData()
        if self.warning:
            print('Warning! self.cell_attributes are reset and need to be updated!')
        self.cell_attributes = dict() #reset
        self.point_attributes = dict() #reset
    
    
    def mesh_subdivision(self, num_subdivisions, method='loop'):
        if method == 'loop':
            subdivision_reader = vtk.vtkLoopSubdivisionFilter()
        elif method == 'butterfly':
            subdivision_reader = vtk.vtkButterflySubdivisionFilter()
        else:
            if self.warning:
                print('Not a valid subdivision method')
            
        subdivision_reader.SetInputData(self.vtkPolyData)
        subdivision_reader.SetNumberOfSubdivisions(num_subdivisions)
        subdivision_reader.Update()
        self.vtkPolyData = subdivision_reader.GetOutput()
        self.get_mesh_data_from_vtkPolyData()
        if self.warning:
            print('Warning! self.cell_attributes are reset and need to be updated!')
        self.cell_attributes = dict() #reset 
        self.point_attributes = dict() #reset
        
    
    def mesh_transform(self, vtk_matrix):
        Trans = vtk.vtkTransform()
        Trans.SetMatrix(vtk_matrix)
        
        TransFilter = vtk.vtkTransformPolyDataFilter()
        TransFilter.SetTransform(Trans)
        TransFilter.SetInputData(self.vtkPolyData)
        TransFilter.Update()
        
        self.vtkPolyData = TransFilter.GetOutput()
        self.get_mesh_data_from_vtkPolyData()
    
    
    def mesh_reflection(self, ref_axis='x'):
        '''
        This function is only for tooth arch model,
        it will flip the label (n=15 so far) as well.
        input:
            ref_axis: 'x'/'y'/'z'
        '''
        xmin = np.min(self.points[:, 0])
        xmax = np.max(self.points[:, 0])
        ymin = np.min(self.points[:, 1])
        ymax = np.max(self.points[:, 1])
        zmin = np.min(self.points[:, 2])
        zmax = np.max(self.points[:, 2])
        center = np.array([np.mean(self.points[:, 0]), np.mean(self.points[:, 1]), np.mean(self.points[:, 2])])
        
        if ref_axis == 'x':
            point1 = [xmin, ymin, zmin]
            point2 = [xmin, ymax, zmin]
            point3 = [xmin, ymin, zmax]
        elif ref_axis == 'y':
            point1 = [xmin, ymin, zmin]
            point2 = [xmax, ymin, zmin]
            point3 = [xmin, ymin, zmax]
        elif ref_axis == 'z':
            point1 = [xmin, ymin, zmin]
            point2 = [xmin, ymax, zmin]
            point3 = [xmax, ymin, zmin]
        else:
            if self.warning:
                print('Invalid ref_axis!')
            
        #get equation of the plane by three points
        v1 = np.zeros([3,])
        v2 = np.zeros([3,])

        for i in range(3):
            v1[i] = point1[i] - point2[i]
            v2[i] = point1[i] - point3[i]

        normal_vec = np.cross(v1, v2)/np.linalg.norm(np.cross(v1, v2))

        flipped_cells = np.copy(self.cells)
        flipped_points = np.copy(self.points)

        #flip cells
        for idx in range(len(self.cells)):
            tmp_p1 = self.cells[idx, 0:3]
            tmp_p2 = self.cells[idx, 3:6]
            tmp_p3 = self.cells[idx, 6:9]
    
            tmp_v1 = tmp_p1 - point1
            dis_v1 = np.dot(tmp_v1, normal_vec)*normal_vec
    
            tmp_v2 = tmp_p2 - point1
            dis_v2 = np.dot(tmp_v2, normal_vec)*normal_vec
            
            tmp_v3 = tmp_p3 - point1
            dis_v3 = np.dot(tmp_v3, normal_vec)*normal_vec
            
            flipped_p1 = tmp_p1 - 2*dis_v1
            flipped_p2 = tmp_p2 - 2*dis_v2
            flipped_p3 = tmp_p3 - 2*dis_v3
    
            flipped_cells[idx, 0:3] = flipped_p1
            flipped_cells[idx, 3:6] = flipped_p3 #change order p3 and p2
            flipped_cells[idx, 6:9] = flipped_p2 #change order p3 and p2
    
        #flip points
        for idx in range(len(self.points)):
            tmp_p1 = self.points[idx, 0:3]
            
            tmp_v1 = tmp_p1 - point1
            dis_v1 = np.dot(tmp_v1, normal_vec)*normal_vec
                    
            flipped_p1 = tmp_p1 - 2*dis_v1
            flipped_points[idx, 0:3] = flipped_p1

        #move flipped_cells and flipped_points back to the center
        flipped_center = np.array([np.mean(flipped_points[:, 0]), np.mean(flipped_points[:, 1]), np.mean(flipped_points[:, 2])])
        displacement = center - flipped_center
        
        flipped_cells[:, 0:3] += displacement
        flipped_cells[:, 3:6] += displacement
        flipped_cells[:, 6:9] += displacement
        
        original_cell_labels = self.cell_attributes['Label'].copy()
        flipped_cell_labels = self.cell_attributes['Label'].copy()
        
        self.cells = flipped_cells
        self.update_cell_ids_and_points() # all cell and point attributes are gone, include label
        
        self.cell_attributes['Label'] = original_cell_labels # add original cell label back
        for i in range(1, 15):
            if len(self.cell_attributes['Label']==i) > 0:
                flipped_cell_labels[self.cell_attributes['Label']==i] = 15-i #1 -> 14, 2 -> 13, ..., 14 -> 1
        self.cell_attributes['Label'] = flipped_cell_labels # update flipped label
    
    
    def to_vtp(self, vtp_filename):
        self.update_vtkPolyData()
        
        if vtk.VTK_MAJOR_VERSION <= 5:
            self.vtkPolyData.Update()
     
        writer = vtk.vtkXMLPolyDataWriter();
        writer.SetFileName("{0}".format(vtp_filename));
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(self.vtkPolyData)
        else:
            writer.SetInputData(self.vtkPolyData)
        writer.Write()
        
        
#------------------------------------------------------------------------------
def GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                               translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                               scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]):
    '''
    get transformation matrix (4*4)
    
    return: vtkMatrix4x4
    '''
    Trans = vtk.vtkTransform()
    
    ry_flag = np.random.randint(0,2) #if 0, no rotate
    rx_flag = np.random.randint(0,2) #if 0, no rotate
    rz_flag = np.random.randint(0,2) #if 0, no rotate
    if ry_flag == 1:
        # rotate along Yth axis
        Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))
    if rx_flag == 1:
        # rotate along Xth axis
        Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))
    if rz_flag == 1:
        # rotate along Zth axis
        Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))

    trans_flag = np.random.randint(0,2) #if 0, no translate
    if trans_flag == 1:
        Trans.Translate([np.random.uniform(translate_X[0], translate_X[1]),
                         np.random.uniform(translate_Y[0], translate_Y[1]),
                         np.random.uniform(translate_Z[0], translate_Z[1])])

    scale_flag = np.random.randint(0,2)
    if scale_flag == 1:
        Trans.Scale([np.random.uniform(scale_X[0], scale_X[1]),
                     np.random.uniform(scale_Y[0], scale_Y[1]),
                     np.random.uniform(scale_Z[0], scale_Z[1])])
    
    matrix = Trans.GetMatrix()
    
    return matrix

    
if __name__ == '__main__':
    
    # create a new mesh by loading a VTP file
    mesh = Easy_Mesh('Sample_010.vtp')
    mesh.get_cell_edges()
    mesh.get_cell_normals()
    mesh.get_point_curvatures()
    mesh.get_cell_curvatures()
    mesh.to_vtp('example.vtp')
#    
#    # create a new mesh by loading a STL/OBJ file
#    mesh = Easy_Mesh('Test5.stl')
#    mesh.set_cell_labels(np.ones([mesh.cells.shape[0], 1]))
#    mesh.get_cell_edges()
#    mesh.get_cell_normals()
#    mesh.to_vtp('example2.vtp')
#    
#    # create a new mesh by loading a main STL file and label it with other STL files
#    mesh = Easy_Mesh('Sample_01_d.stl')
#    mesh1 = Easy_Mesh('Sample_01_T2_d.stl')
#    mesh2 = Easy_Mesh('Sample_01_T3_d.stl')   
#    ## make a label dict in which key=str(label_ID), value=cells (i.e., [n, 9] array)
#    label_dict = {'1': mesh1.cells, '2': mesh2.cells} 
#    mesh.set_cell_labels(label_dict)
#    mesh.to_vtp('example_with_labels.vtp')
#    mesh.mesh_reflection('x')
#    mesh.to_vtp('example_with_labels_fliped.vtp')
#    
#    # decimation
#    mesh_d = Easy_Mesh('A0_Sample_01.vtp')
#    mesh_d.mesh_decimation(0.5)
#    print(mesh_d.cells.shape)
#    print(mesh_d.points.shape)
#    mesh_d.get_cell_edges()
#    mesh_d.get_cell_normals()
#    mesh_d.compute_cell_attributes_by_svm(mesh.cells, mesh.cell_attributes['Label'], 'Label')
#    mesh_d.to_vtp('decimation_example.vtp')
#    
#    # subdivision
#    mesh_s = Easy_Mesh('A0_Sample_01.vtp')
#    mesh_s.mesh_subdivision(2, method='butterfly')
#    print(mesh_s.cells.shape)
#    print(mesh_s.points.shape)
#    mesh_s.get_cell_edges()
#    mesh_s.get_cell_normals()
#    mesh_s.compute_cell_attributes_by_svm(mesh.cells, mesh.cell_attributes['Label'], 'Label')
#    mesh_s.to_vtp('subdivision_example.vtp')
#    
#    # flip mesh for augmentation
#    mesh_f = Easy_Mesh('A0_Sample_01.vtp')
#    mesh_f.mesh_reflection(ref_axis='x')
#    mesh_f.to_vtp('flipped_example.vtp')
#
#    # create a new mesh from cells
#    mesh2 = Easy_Mesh()
#    mesh2.cells = mesh.cells[np.where(mesh.cell_attributes['Label']==1)[0]]
#    mesh2.update_cell_ids_and_points()
#    mesh2.set_cell_labels(mesh.cell_attributes['Label'][np.where(mesh.cell_attributes['Label']==1)[0]])
#    mesh2.to_vtp('part_example.vtp')
#    
#    # downsampled UR3 (label==5) and compute heatmap
#    tooth_idx = np.where(mesh.cell_attributes['Label']==5)[0]
#    print(len(tooth_idx))
#    mesh2 = Easy_Mesh()
#    mesh2.cells = mesh.cells[tooth_idx]
#    mesh2.update_cell_ids_and_points()
#    target_cells = 400
#    rate = 1.0 - target_cells/len(tooth_idx) - 0.005
#    mesh2.mesh_decimation(rate)
#    mesh2.get_cell_normals()
#    mesh2.compute_guassian_heatmap(mesh2.points[3])
#    mesh2.to_vtp('Canine_d.vtp')
#    
#    # downsampled UR3 (label==5) and compute heatmap
#    tooth_idx = np.where(mesh.cell_attributes['Label']==5)[0]
#    mesh2 = Easy_Mesh()
#    mesh2.cells = mesh.cells[tooth_idx]
#    mesh2.update_cell_ids_and_points()
#    mesh2.mesh_subdivision(2, method='butterfly')
#    mesh2.get_cell_normals()
#    mesh2.compute_displacement_map(np.array([0, 0, 0]))
#    mesh2.to_vtp('Canine_s.vtp')
#    
#    # trnasform a mesh
#    matrix = GetVTKTransformationMatrix()
#    mesh.mesh_transform(matrix)
#    mesh.to_vtp('example_t.vtp')