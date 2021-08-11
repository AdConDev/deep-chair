from __future__ import print_function, division

from torch.autograd import Variable
import torch
from PIL import Image
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import os
import copy

def CrearResnet18(pesos):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ResnetFacial = models.resnet18(pretrained=True)
    num_ftrs = ResnetFacial.fc.in_features
    ResnetFacial.fc = nn.Linear(num_ftrs, 5)
    ResnetFacial = ResnetFacial.to(device)
    ResnetFacial.load_state_dict(torch.load(pesos))
    pass

def Normalizar(x):
    normalized = (x-np.min(x))/(np.max(x)-np.min(x))
    return normalized

def Conversion(imagen):
    image = Image.open(imagen)
    Imagen = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = Imagen(image)
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.cuda()

def Prueba(direccion,imagen):
    clases = ['Abajo','Arriba','Derecha','Izquierda','Neutro']
    ruta = os.path.join(direccion,imagen)
    with torch.no_grad():
        ResnetFacial.eval()
        outputs = ResnetFacial(Conversion(ruta))
        _, preds = torch.max(outputs,1)
    Resultados = Normalizar(outputs.data.cpu().numpy())
    print(Resultados , clases[preds],'\n')
    pass

CrearResnet18('Resnet18Facial.pth')
Prueba(r'C:\Users\secan\Desktop\\Biomédica\TransferLearning\Training Seti',r'IMG_20180806_080759337_BURST000_COVER.jpg')
Prueba(r'C:\Users\secan\Desktop\\Biomédica\TransferLearning\Training Seti',r'IMG_20180806_080827898_BURST001.jpg')
Prueba(r'C:\Users\secan\Desktop\\Biomédica\TransferLearning\Training Seti',r'IMG_20180806_080844840_BURST025.jpg')
Prueba(r'C:\Users\secan\Desktop\\Biomédica\TransferLearning\Training Seti',r'IMG_20180806_080903315_BURST001.jpg')
Prueba(r'C:\Users\secan\Desktop\\Biomédica\TransferLearning\Training Seti',r'IMG_20180806_080922110_BURST023.jpg')
