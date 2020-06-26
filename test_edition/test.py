# -*- coding:utf-8 -*-
import codecs,sys
import numpy as np
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk;
import predict
import cv2
from PIL import Image, ImageTk
import threading
import time

import importlib,sys
importlib.reload(sys)

def detect_carnumber(img_bgr, box):  #r1, r2, r3, r4
    predictor = predict.CardPredictor()
    img_bgr = img_bgr[box[0]:box[1], box[2]:box[3]]
    r = predictor.predict(img_bgr)
    return r
