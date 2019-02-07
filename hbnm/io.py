#! usr/bin/python
import numpy as np
from scipy.io import loadmat, savemat
import os
import pandas as pd
import nibabel as nib
import h5py

class Data:
    """
    An auxiliary class to facilitate loading and saving data
    """
    def __init__(self, input_dir, output_dir):
        """
        Parameters
        ----------
        input_dir : str
            Input directory to loed data
        output_dir : str
            Output directory to save results
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

    def load(self, filename, numeric=False, type=None, from_path=False, from_output=False, *args, **kwargs):
        """
        Load method
        
        Parameters
        ----------
        file_name : str
            The name of the file to load
        numeric : bool, optional
            This keyword is required only when load csv or txt files
        type : str, optional
            The type of the file. If None the method will guess the file type based on its extension
        from_output : bool, optional
            If True the file will be loaded from the output directory, otherwise it will be loaded
            from the input directory
             
        Returns
        -------
        object
            File object (depends on the extension of the file) 
             
        Notes
        -----
        This method supports files with extensions '.hdf5', '.mat', '.npy/.npz', '.xlsx', '.txt'. It loads
        the file using the appropriate method based on the extensions. E.g. uses pandas for '.xlsx'
        """
        if from_path:
            file_path = from_path + filename
        else:
            if from_output:
                file_path = self.output_dir + filename
            else:
                file_path = self.input_dir + filename

        if type is None:
            filename, file_extension = os.path.splitext(file_path)
        else:
            file_extension = type

        if file_extension == '.hdf5':
            return h5py.File(file_path,'r')

        if file_extension == '.mat':
            return loadmat(file_path)

        if (file_extension == '.npy') | (file_extension == '.npz'):
            return np.load(file_path)

        if file_extension == '.xlsx':
            return pd.read_excel(file_path, *args, **kwargs)

        if file_extension == '.csv':
            if numeric:
                return np.loadtxt(file_path)
            else:
                return pd.read_csv(file_path, *args, **kwargs)

        if file_extension == '.txt':
            if numeric:
                return np.loadtxt(file_path, *args, **kwargs)
            else:
                f = open(file_path, 'r')
                read_data = f.readlines()
                return [read_data[ii][:-1] for ii in range(len(read_data))]


    def save(self, filename, data=None, numeric=False, *args, **kwargs):
        """
        Save method
        
        Parameters
        ----------
        filename : str
            The name of the file to save
        data : ndarray, optional
            The data to be saved. It is not necessary if multiple data will be saved.
        numeric : bool, optional
            This keyword is required only when saving csv or txt files
             
        Returns
        -------
        object
            File object (depends on the extension of the file) 
             
        Notes
        -----
        This method supports files with extensions '.hdf5', '.mat', '.npy/.npz', '.xlsx', '.txt'. It loads
        the file using the appropriate method based on the extensions. E.g. uses pandas for '.xlsx'
        """

        file_path = self.output_dir + filename
        filename, file_extension = os.path.splitext(file_path)
        if file_extension == '.hdf5':
            return h5py.File(file_path, 'w')

        if file_extension == '.mat':
            savemat(file_path, data)

        if (file_extension == '.npy') | (file_extension == '.npz'):
            np.save(file_path, data)

        if file_extension == '.xlsx':
            data.to_excel(file_path, *args, **kwargs)

        if file_extension == '.csv':
            if numeric:
                np.savetxt(file_path, data)
            else:
                data.to_csv(file_path, *args, **kwargs)

        if file_extension == '.txt':
            np.savetxt(file_path, data)

    def load_nifti(self, filename):
        """
        NIFTI loader

        Parameters
        ----------
        filename : str
            The name of the file to load

        Returns
        -------
        ndarray
            The data from the NIFTI file 
        """
        file_path = self.input_dir + filename
        return np.array(np.squeeze(nib.load(file_path).get_data()))

    def save_nifti(self, output, filename, template):
        """
        NIFTI save

        Parameters
        ----------
        output : ndarray
            The data to save as NIFTI file
        filename : str
            The name of the file to save
        template : str
            The name of the template file to be used when generating the NIFIT 
        """
        of = nib.load(template)
        temp_data = np.array(of.get_data())
        data_to_write = output.reshape(np.shape(temp_data))
        new_img = nib.Nifti1Image(data_to_write, affine=of.get_affine(), header=of.get_header())
        nib.save(new_img, self.output_dir + filename)

    def load_cifti(self, filename, from_input=False):
        """
        CIFTI loader

        Parameters
        ----------
        filename : str
            The name of the file to load

        Returns
        -------
        ndarray
            The data from the CIFTI file 
        """
        if from_input: filename = self.input_dir + filename
        of = nib.load(filename)
        return np.array(np.squeeze(of.get_data()))

    def save_cifti(self, output, filename, template):
        """
        NIFTI save
        
        Parameters
        ----------
        output : ndarray
            The data to save as CIFTI file
        filename : str
            The name of the file to save
        template : str
            The name of the template file to be used when generating the CIFTI 
        """
        of = nib.load(template)
        data_to_write = output.reshape(np.shape(np.array(of.get_data())))
        new_img = nib.Nifti2Image(data_to_write, affine=of.get_affine(), header=of.get_header())
        nib.save(new_img, filename)