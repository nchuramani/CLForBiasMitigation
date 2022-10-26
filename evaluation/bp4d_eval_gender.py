import numpy as np
import os

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file
            
folder = "/home/ozgur/Documents/CLForBiasMitigation/Benchmark with BP4D/results/gender/" # folder of the result path that includes .txt files
methods = ["EWC_"]
regularization_list = ["10.0_","50.0_","100.0_","500.0_","1000.0_","5000.0_"]

mydict_acc = {}
mydict_2 = {}

def BWT(matrix):
    task_num = matrix.shape[0]
    sum1 = 0
    for i in range(1,task_num):
        for j in range(0,i):
            sum1 += matrix[i,j] - matrix[j,j]
    sum1 = sum1 / (task_num * (task_num-1) / 2)
    return sum1

def to_table(lis):
    acc = np.zeros((2,2))
    f1 = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            acc[i,j] = float(lis[5*i + 2*(j+1)][:7])
    for i in range(2):
        for j in range(2):
            f1[i,j] = float(lis[10 + 5*i + 2*(j+1)])
            
    return f1, acc/100
    

for file in files(folder):
    mydict_acc[file] = {}
    mydict_2[file] = {}
    with open(folder+file, "r") as f:
        data = f.readlines()
        data = data[1:]
        f1 = np.zeros((len(data),2,2))
        acc = np.zeros((len(data),2,2))
        for i, line in enumerate(data):
            line = line.replace(")","").replace("(","").replace("[","").replace("]","").replace(",","")
            l = line.split()
            
            f1[i],acc[i] = to_table(l)
            if l[20][-1] == "0":  
                mydict_acc[file][l[23] + "_" + l[22]] = [] 
            mydict_acc[file][l[23] + "_" + l[22]].append(acc[i])
        for keys in mydict_acc:
            for key in mydict_acc[keys]:
                a = np.zeros((len(mydict_acc[keys][key]),2,2))
                for i,lis in enumerate(mydict_acc[keys][key]):
                    a[i] = mydict_acc[keys][key][i]
                mydict_2[keys][key] = a

for keys in mydict_2:
    for key in mydict_acc[keys]:
        mydict_2[keys][key + "_meanstd"] = np.vstack((mydict_2[keys][key].mean(axis=0)[:,-1], mydict_2[keys][key].std(axis=0)[:,-1]))
        mydict_2[keys][key + "_avg"] = mydict_2[keys][key].mean(axis=0)
a = {}
b = {}  
print('BWT scores')
for met in methods:
    print(met)
    for par in regularization_list:
        a = mydict_2[list(mydict_2.keys())[0]]
        b = mydict_2[list(mydict_2.keys())[1]]

        x = b[met + par + "avg"]
        y = a[met + par + "avg"]
        
        print(par,"not augmented", BWT(y))
        print(par, "augmented", BWT(x))

print('accuracy scores')
for met in methods:
    for par in regularization_list:
        a = mydict_2[list(mydict_2.keys())[0]]
        b = mydict_2[list(mydict_2.keys())[1]]

        x = b[met + par + "meanstd"]
        y = a[met + par + "meanstd"]
        print(met+par)
        print("Without Augmentation: gender_1 acc , gender_2 acc, fairness\ With Augmentation: gender_1 acc, gender_2 acc , fairness")
        print("%.3f$\pm$ %.3f & %.3f$\pm$%.3f & %.3f & %.3f$\pm$%.3f & %.3f$\pm$%.3f & %.3f"  % (x[0,0], x[1,0], x[0,1], x[1,1], min(x[0,0] / x[0,1], x[0,1]/x[0,0] ), y[0,0], y[1,0], y[0,1], y[1,1], min(y[0,0] / y[0,1], y[0,1]/y[0,0] )))
        
        
        
