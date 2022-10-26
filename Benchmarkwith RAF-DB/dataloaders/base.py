import torchvision
from torchvision import transforms
from .wrapper import CacheClassLabel, MyDataset, AppendName
from torch.utils.data import  DataLoader
import PIL


import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

def RafDB(train_root, val_root):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    val_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
   
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_root,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)
    val_dataset = torchvision.datasets.ImageFolder(
        root=val_root,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)
    return train_dataset, val_dataset


def get_indexes(attribute_path, attribute_list, num_of_test, num_of_train):
    mydict_train = {}
    mydict_test = {}
    for i, attr in enumerate(attribute_list):
        train, test = read_attribute(attribute_path, attr, num_of_test, num_of_train)
        for j in range(5):
            if(i != 2 and j == 3):
                break
            mydict_train[attr + "_" + str(j)] = np.where(train == j)[0]
            mydict_test[attr + "_" + str(j)] = np.where(test == j)[0]

    mydict_train["age_combined_0"] = np.concatenate((mydict_train["age_0"],mydict_train["age_1"]), axis=0)
    mydict_train["age_combined_1"] = mydict_train["age_2"]
    mydict_train["age_combined_2"] = np.concatenate((mydict_train["age_3"],mydict_train["age_4"]), axis=0)
    mydict_test["age_combined_0"] = np.concatenate((mydict_test["age_0"],mydict_test["age_1"]), axis=0)
    mydict_test["age_combined_1"] = mydict_test["age_2"]
    mydict_test["age_combined_2"] = np.concatenate((mydict_test["age_3"],mydict_test["age_4"]), axis=0) 
    return mydict_train, mydict_test


def discriminate(attribute, element_in_attribute, train_y, mydict):
    h = train_y
    for i in range(len(h)):
        for j in range(element_in_attribute):
            if(i in mydict[attribute + "_" + str(j)]):
                h[i] += 7*j    
    return h

def load_image(filename):
    """read an image and convert it to numpy array and returns it"""
    img = Image.open(filename)
    img.load()
    #img = img.resize((45,55)) 
    img = np.transpose(img,(2,0,1))
    data = np.asarray(img, dtype="int32")
    return data

def read_data(filepath, num_of_test, num_of_train, size):
    """read the images in the dataset and return 4d numpy array of them"""
    train_set = np.zeros((num_of_train, size[0], size[1], size[2]))
    test_set = np.zeros((num_of_test, size[0], size[1], size[2]))
    filenames = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    for file in filenames:
        if(file[1] == "e"): #if it's name is test
            index = int(file[5:9]) - 1
            test_set[index,:,:,:] = load_image(filepath + file)
        else:
            index = int(file[6:11]) - 1
            train_set[index,:,:,:] = load_image(filepath + file)
    return train_set, test_set

def read_classes(filepath, num_of_test, num_of_train):
    """read the classes of corresponding images and returns the numpy array"""
    test_y = np.zeros((num_of_test))
    train_y = np.zeros((num_of_train))
    with open(filepath, "r") as file:
        data = file.readlines()
        for line in data:
            if(line[1] == "e"):
                index = int(line[5:9]) - 1
                test_y[index] = int(line[-2])
            else:
                index = int(line[6:11]) - 1
                train_y[index] = int(line[-2])
                
    return train_y, test_y

def get_number_from_file(filename, line_number):
    with open(filename) as file:
        data = file.readlines()
        return int(data[line_number-1])

def read_attribute(filepath, attribute, num_of_test, num_of_train):
    filenames = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    index = 0
    test_y = np.zeros((num_of_test))
    train_y = np.zeros((num_of_train))
    if(attribute == "gender"):
        index = 6
    elif(attribute == "race"):
        index = 7
    elif(attribute == "age"):
        index = 8
    for line in filenames:
        num = get_number_from_file(filepath + line, index)
        if(line[1] == "e"):
            index2 = int(line[5:9]) - 1
            test_y[index2] = num
        else:
            index2 = int(line[6:11]) - 1
            train_y[index2] = num
    return train_y, test_y


def RafDB_perm(category, train_aug=False):
    # this function reads RAF-DB data and it separates the samples given category. Output as dictionary. (dict["1"] = male samples, dict["2"] = female samples for gender ..)
    
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_transform = val_transform
    
    if train_aug:
        train_transform = transforms.Compose([
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])
    
    attribute_list = ["gender", "race", "age"]
    data_path = "/home/ozgur/Documents/Rafdb/basic/Image/aligned/"
    classes_path = "/home/ozgur/Documents/Rafdb/basic/EmoLabel/list_partition_label.txt"
    attribute_path = "/home/ozgur/Documents/Rafdb/basic/Annotation/manual/"
    num_of_train = 12271
    num_of_test = 3068
    size = [3,100,100]
    a = category + "_"
    
    mydict_train, mydict_test = get_indexes(attribute_path, attribute_list, num_of_test, num_of_train)

    train_x, test_x = read_data(data_path, num_of_test, num_of_train, size)
    train_y, test_y = read_classes(classes_path, num_of_test, num_of_train)
    
    train_y -= 1
    test_y -= 1
    train_datasets = {}
    val_datasets = {}
    task_output_space = {}   
    arr = [1,2,3,4,5] if category == "age" else [1,2,3]
    for j in range(1, len(arr) + 1):
        i = arr[j-1]
        name = str(j)
        tensor_x = train_x[mydict_train[a + str(i-1)]]
        tensor_test_x = test_x[mydict_test[a + str(i-1)]]
        tensor_y = train_y[mydict_train[a + str(i-1)]]
        tensor_test_y = test_y[mydict_test[a + str(i-1)]]
        train_dataset = MyDataset(tensor_x, tensor_y, transform=train_transform, root="data")
        train_dataset = CacheClassLabel(train_dataset)
        test_dataset = MyDataset(tensor_test_x, tensor_test_y, transform=val_transform, root="data")
        test_dataset = CacheClassLabel(test_dataset)
        train_datasets[name] = AppendName(train_dataset, name)
        val_datasets[name] = AppendName(test_dataset, name)
        task_output_space[name] = 7
    return train_datasets, val_datasets, task_output_space
