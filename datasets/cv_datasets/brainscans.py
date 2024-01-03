import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms

from torchvision import datasets
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data

mean, std = {}, {}
mean['brainscans'] = [0.5, 0.5, 0.5]
std['brainscans'] = [0.5, 0.5, 0.5]

#TO DO:
#update the mean and std func
#see the need for eval dataset

def read_brainscans_images( data_dir, folder_name = 'train'):
    
    #data_dir = os.path.join(data_dir, name.lower())
    print ('Reading brain scans dataset at location:', data_dir +"/"+folder_name)

    new_size = 300
    crop_size = 224

    baremin_transforms = transforms.Compose([transforms.Resize(new_size), 
                                             transforms.CenterCrop(crop_size),
                                             transforms.ToTensor()])
    
    ds_obj = datasets.ImageFolder(os.path.join(data_dir, folder_name ),
                                  transform = baremin_transforms)
    
    
    ds_len = len(ds_obj)
    
    X_ds = np.zeros((ds_len, crop_size, crop_size, 3))
    y_ds = np.zeros(ds_len)
    
    for ii in range(ds_len):
        X_ds[ii] = np.transpose(ds_obj[ii][0].numpy(), (1, 2, 0)) #(channels, height, width) to (height, width, channels)
        y_ds[ii] = ds_obj[ii][1]

    

    return X_ds,y_ds.astype(int)


def write_brainscans_images(data_dir):

    #data_dir = os.path.join(data_dir, name.lower())
    print ('Writing brain scans dataset at location:', data_dir)
    
    folders = ['train','test']
    for f in folders:
        X_ds,y_ds =  read_brainscans_images (data_dir, f)
        np.save( os.path.join(data_dir, 'X_'+ f ), X_ds)
        np.save( os.path.join(data_dir, 'y_'+ f ), y_ds)  
    return 0

def read_brainscans_dset(data_dir='./data', name='brainscans'):
    data_dir = os.path.join(data_dir, name.lower())

    if not (os.path.isfile(data_dir+'/X_train.npy') ):
        print ("Dataset does not exist, creating dataset")
        write_brainscans_images(data_dir)

    X_train = np.load(data_dir+'/X_train.npy')
    y_train = np.load(data_dir + '/y_train.npy')
    X_test = np.load(data_dir + '/X_test.npy')
    y_test = np.load(data_dir + '/y_test.npy')
    #X_eval = np.load(data_dir + '/X_eval.npy')
    #y_eval = np.load(data_dir + '/y_eval.npy')
    return X_train, y_train, X_test, y_test#, X_eval, y_eval

def get_brainscans(args, alg, name, num_labels, num_classes, labeled_percentage ,data_dir='./data', include_lb_to_ulb=True):
    
    #data, targets, test_data, test_targets, X_eval, y_eval = read_brainscans_dset(data_dir='./data', name='brainscans')
    data, targets, test_data, test_targets = read_brainscans_dset(data_dir='./data', name='brainscans')

    print ("data.shape", data.shape)
    print ("targets.shape", targets.shape)
    print ("test_data", test_data.shape)
    print ("test_targets", test_targets.shape)
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb, labeled_percentage=args.labeled_percentage)
    
    
    print ('lb_targets', lb_targets)
    print ('ulb_targets', ulb_targets)

    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))


    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)

    print(f"#Labeled: {len(lb_dset)} #Unlabeled: {len(ulb_dset)} #Val: {len(eval_dset)}")

    return lb_dset, ulb_dset, eval_dset



#read_brainscans_dset(data_dir='/home/ubunto/MasterThesis_old/Semi-supervised-learning2/data', name='brainscans')
#write_brainscans_images(data_dir='/home/ubunto/MasterThesis_old/Semi-supervised-learning2/data', name='brainscans')

#if __name__ == "__main__":
#    #print ("here")
#    write_brainscans_images(data_dir='/home/ubunto/MasterThesis_old/Semi-supervised-learning2/data/brainscans')
