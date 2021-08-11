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

import serial
import cv2 as OpenCV

def Resnet18(pesos,clases = 5):
    ResnetFacial = models.resnet18(pretrained=True)
    num_ftrs = ResnetFacial.fc.in_features
    ResnetFacial.fc = nn.Linear(num_ftrs, clases)
    ResnetFacial = ResnetFacial.to(device)
    ResnetFacial.load_state_dict(torch.load(pesos))
    return ResnetFacial

def Normalizar(x):
    normalized = (x-np.min(x))/(np.max(x)-np.min(x))
    return normalized

def Conversion(imagen):
    pil_image = OpenCV.cvtColor(imagen, OpenCV.COLOR_BGR2RGB)
    image = Image.fromarray(pil_image)
    trans = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = trans(image)
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.cuda()

def Prueba(ResnetFacial, imagen):
    with torch.no_grad():
        ResnetFacial.eval()
        outputs = ResnetFacial(Conversion(imagen))
        _, preds = torch.max(outputs,1)
        Resultados = Normalizar(outputs.data.cpu().numpy())
    return ''.join(str(clases[preds]))

def ControlPorGestos(inicio = True):
    elapsed = 0;
    font = OpenCV.FONT_HERSHEY_SIMPLEX
    cam = OpenCV.VideoCapture(0)
    ret_val, img = cam.read()
    prueba = Prueba(Modelo,img)
    inicio = time.time()
    while True:
        ret_val, img = cam.read()
        if elapsed >= 0.1:
            prueba = Prueba(Modelo,img)
            Accion(prueba)
            inicio = fin
        ImgDone = OpenCV.putText(img,prueba,(100,400), font, 4,(0,0,0),2,OpenCV.LINE_AA)
        OpenCV.imshow('Reconocimiento facial', ImgDone)
        if OpenCV.waitKey(1) == 27:
            break  # esc to quit
        fin = time.time()
        elapsed = fin - inicio
    OpenCV.destroyAllWindows()
    cam.release();
    arduino.close();

def Accion(Gesto):
    if Gesto == clases[0]:
        arduino.write(b'0')
    if Gesto == clases[1]:
        arduino.write(b'1')
    if Gesto == clases[2]:
        arduino.write(b'2')
    if Gesto == clases[3]:
        arduino.write(b'3')
    if Gesto == clases[4]:
        arduino.write(b'4s')

clases = ['Abajo','Arriba','Derecha','Izquierda','Neutro']
arduino = serial.Serial('COM1', 9600)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pesos = 'Resnet18Facial.pth'
Modelo = Resnet18(pesos)
Lectura = OpenCV.VideoCapture(0)

valido, foto = Lectura.read()
if valido == True:
    print('Foto lista')
else:
    print('Error al iniciar cámara')
Lectura.release()

ControlPorGestos();
arduino.close()
