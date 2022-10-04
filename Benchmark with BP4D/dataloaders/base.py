import torchvision
from torchvision import transforms
from .wrapper import CacheClassLabel, MyDataset, AppendName
from torch.utils.data import  DataLoader
from PIL import Image
import pickle
from os import listdir
from os.path import isfile, join
import xlrd
import numpy as np
import torch

# Paths for Labels and Images
BP4D_EXCEL = "BP4D/BP4D_data.xlsx"
BP4D_IMAGES = "BP4D/images_crop"
# Images paths and Occurrence labels used to create a pickle file.
BP4D_OCCURENCE = "BP4D/aus_bp4d_occurrence.pkl"

def load_image(filename):
    """read an image and convert it to numpy array and returns it"""
    img = Image.open(filename)
    img.load()
    img = img.resize((32,32)) 
    img = np.transpose(img,(2,0,1))
    data = np.asarray(img, dtype="int32")
    return data

def load_info(excel_file):
    data_dict = {}
    wb = xlrd.open_workbook(excel_file)
    sheet = wb.sheet_by_index(0)
    for row in range(1,sheet.nrows):
        if(sheet.cell_value(row ,0) == ""):
            continue
        data_dict[sheet.cell_value(row, 0)] = [sheet.cell_value(row, 1), sheet.cell_value(row, 3), sheet.cell_value(row, 4), sheet.cell_value(row, 5), sheet.cell_value(row, 6)]
    return data_dict
 
    
def split_data(label_names, data, label):
    data_dict = {}
    label_dict = {}
    for key in label:
        if key[:4] in label_names:
            label_dict[key] = label[key]
    for key in data:
        if key[:4] in label_names:
            data_dict[key] = data[key]
            
    return label_dict, data_dict

def split_gender(gender, data, label):
    data_dict = {}
    label_dict = {}
    for key in label:
        if key[:1] == gender:
            label_dict[key] = label[key]
    for key in data:
        if key[:1] == gender:
            data_dict[key] = data[key]
            
    return label_dict, data_dict



def train_test_split(x, y, fold, transform):
    train_x = {}
    train_y = {}
    test_x = {}
    test_y = {}
    for key in list(x.keys()):
        if key[:4] in fold:
            test_x[key] = x[key]
            test_y[key] = y[key]

        else:
            train_x[key] = x[key]
            train_y[key] = y[key]
            
    return MyDataset(train_x, train_y, "data", transform), MyDataset(test_x, test_y, "data", transform)


def BP4D_perm_gender(train_aug=False):
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_transform = val_transform
    
    if train_aug:
        train_transform = transforms.Compose([
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])    
    mypath = BP4D_IMAGES
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    file = BP4D_OCCURENCE
#    excel_file = "BP4D/BP4D_data.xlsx"
    with open(file, "rb") as f:
        label = pickle.load(f)
    
    datax = {}


    au_list = ["1", "2", "4", "6", "7", "10", "12", "14", "15", "17", "23", "24"]    
    for i,f in enumerate(onlyfiles):
        # if(i%5000 == 0):
        #     print(i)
        datax[f[:-4]] = load_image(mypath + "/" + f)
#    data_dict = load_info(excel_file)
    train_datasets = {}
    val_datasets = {}
    task_output_space = {}
    # Example train_fold. Update this to randomly select train-fold based on subject IDs from the BP4D dataset.
    train_fold = ['F001', 'F002', 'F008', 'F009', 'F010', 'F016', 'F018', 'F023', 'M001', 'M004', 'M007', 'M008', 'M012', 'M014']
   
    y_male, X_male = split_gender("M", datax, label)
    
    y_female, X_female = split_gender("F", datax, label)

    dataset_train_male, dataset_test_male = train_test_split(X_male, y_male, train_fold, train_transform)
    dataset_train_female, dataset_test_female = train_test_split(X_female, y_female, train_fold, train_transform)
    order = [1,2]

    train_datasets[f"{order[0]}"] = AppendName(dataset_train_male, f"{order[0]}")
    train_datasets[f"{order[1]}"] = AppendName(dataset_train_female, f"{order[1]}")
    
    val_datasets[f"{order[0]}"] = AppendName(dataset_test_male, f"{order[0]}")
    val_datasets[f"{order[1]}"] = AppendName(dataset_test_female, f"{order[1]}")

    task_output_space[f"{order[0]}"] = len(au_list)  
    task_output_space[f"{order[1]}"] = len(au_list)
    
    return train_datasets, val_datasets, task_output_space





def BP4D_perm_race(train_aug=False):
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_transform = val_transform
    excel_file = BP4D_EXCEL
    if train_aug:
        train_transform = transforms.Compose([
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])    
    mypath = BP4D_IMAGES
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    file = BP4D_OCCURENCE
#    excel_file = "BP4D/BP4D_data.xlsx"
    with open(file, "rb") as f:
        label = pickle.load(f)
    
    datax = {}
    data_dict = load_info(excel_file)    
    au_list = ["1", "2", "4", "6", "7", "10", "12", "14", "15", "17", "23", "24"]    
    train_datasets = {}
    val_datasets = {}
    task_output_space = {}   
    datasets = {}
    for i in range(1,5):
        datasets[str(i) + "_x"] = {}
        datasets[str(i) + "_y"] = {}
    for i,f in enumerate(onlyfiles):
        if(i%5000 == 0):
            print(i)
        img = load_image(mypath + "/" + f)
        datax[f[:-4]] = img
        category = data_dict[f[:4]].index(1.0)
        datasets[str(category) + "_x"][f[:-4]] = img

    
    for i,f in enumerate(label.keys()):
        category = data_dict[f[:4]].index(1.0)
        datasets[str(category) + "_y"][f] = label[f]
        
    # Example test_fold. Update this to select test-fold based on subject IDs from the BP4D dataset.
    test_fold = ['F001', 'F002', 'F008', 'F009', 'F010', 'F016', 'F018', 'F023', 'M001', 'M004', 'M007', 'M008', 'M012', 'M014']
    train_datasets = {}
    val_datasets = {}
    task_output_space = {}  
    order = [1,2,3,4]
    for ind,ord in enumerate(order):
        a,b = train_test_split(datasets[f"{ord}_x"], datasets[f"{ord}_y"], test_fold, train_transform)
        train_datasets[f"{ind+1}"] = AppendName(a, f"{ind+1}")
        val_datasets[f"{ind+1}"] = AppendName(b, f"{ind+1}")
        task_output_space[f"{ind+1}"] = len(au_list)  


    return train_datasets, val_datasets, task_output_space
