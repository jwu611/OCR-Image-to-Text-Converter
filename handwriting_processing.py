import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import shutil
import os
from PIL import Image, ImageDraw, ImageChops, ImageOps
#import matplotlib.pyplot as plt
import random
import cv2 as cv
import math

OUTPUT_FOLDER = './output'

device = torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu")

file = open('./model/handwriting_classifier34_finetuned.pkl', 'r')
model = torch.load('./model/handwriting_classifier34_finetuned.pkl', map_location=device)
file.close()
model.eval()

# Specify transforms that were executed on training samples using torchvision.transforms
transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Preprocessing - Using image processing to clean up noise and make input images resemble the format of training samples for better predictions
def preprocess_image(image_path):
  image = cv.imread(image_path)
  blurred = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  blurred = cv.GaussianBlur(gray, (3, 3), 0) #using gaussian blur as low pass filter to reduce noise
  _, thresholded = cv.threshold(blurred,90,255,cv.THRESH_BINARY)

  #resize if necessary and save thresholded image for display in web app
  thresholded_path = OUTPUT_FOLDER+"/thresholded.png"
  h, w = thresholded.shape
  if w > 500 or h > 600:    #max height = 600px, max width = 500px
    scale_factor = 500/w if h <= 600 else 600/h
    target_w = int(w * scale_factor)
    target_h = int(h * scale_factor)
    target_dim = (target_w, target_h)
    clean_img = cv.resize(thresholded, target_dim)
  cv.imwrite(thresholded_path, clean_img)

  # perform edge detection, find contours in the edge map, and sort the
  # resulting contours from left-to-right
  edged = cv.Canny(thresholded, 50, 150)
  contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  #findContours outputs contours list and hierarchy
  cv.drawContours(edged, contours, -1, (127,127,127), 1)

  # Remove contours overlapping w other contours - each letter should only have 1 associated contour
  i = 0
  while i < len(contours):
    x,y,w,h = cv.boundingRect(contours[i])

    for j in range(len(contours)):
      if j != i:
        x_j, y_j, w_j, h_j = cv.boundingRect(contours[j])
        x_j1,y_j1 = x_j + w_j, y_j + h_j 
        x1 = x+w
        y1 = y+h
        #Remove contour if it is overlapping with any other contour, ex. holes in letters (B, P, etc)
        if ((x > x_j and x < x_j1) or (x1 > x_j and x1 < x_j1)) and ((y > y_j and y < y_j1) or (y1 > y_j and y1 < y_j1)): 
          del contours[i]
          break
    i += 1

  #sort contours from top to bottom (rounded to nearest 100 for leeway), then left to right
  contours = sorted(contours, key=lambda c: (round(cv.boundingRect(c)[1]/100)*100, cv.boundingRect(c)[0]), reverse=False)

  letters_list = []   #entries of format (image, num spaces after curr letter, num lines after curr letter)

  for c in range(len(contours)):
    x,y,w,h = cv.boundingRect(contours[c])
    
    #cv.putText(blurred, str(c), (x, y-3), cv.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
    if w < 20 or h < 20:  #if contour is too small, it is probably noise and not a letter
      continue
    roi = thresholded[y:y+h, x:x+w]   #crop image to take only contour

    #edged_contoured = cv.rectangle(edged,(x,y),(x+w,y+h),(0,255,0),10)

    #resize letter such that its longer length (width or height) is 20 pix while preserving aspect ratio - this better resembles the training samples
    scale_factor = 20/w if w > h else 20/h
    target_w = int(w * scale_factor)
    target_h = int(h * scale_factor)
    target_dim = (target_w, target_h)
    roi = cv.resize(roi, target_dim)

    #add border to make the image 28x28 pix like the training samples
    width_border = 28-target_w    #border for left/right
    height_border = 28-target_h  #border for top/bottom
    roi = cv.copyMakeBorder(roi,height_border,height_border,width_border,width_border,cv.BORDER_CONSTANT,value=[255,255,255])   #add white border to better ressemble samples used for training
   
    num_spaces = 0
    num_lines = 0
    if c < (len(contours)-1):
      x_next,y_next,w_next,h_next = cv.boundingRect(contours[c+1])
      y_bottom = y+h
      x_right = x+w
      #check if contour c+1 is on a new line
      if y_next >= y_bottom:
        num_lines += (math.floor((y_next-y_bottom)/h)+1)  #if there is less than a space of height h b/w the letters -> there is 1 newline (1 \n); > space of h = 2 newlines

      #only add spaces if next letter is on same line, otherwise start at left margin
      if num_lines == 0:
        if (x_next-x_right) >= 0:
          num_spaces += round(((x_next-x_right)/w)-0.25)   #add space after letter if the space b/w this letter & next >= 0.75*w
    letters_list.append((roi, num_lines, num_spaces))
    #cv.imwrite('test_processed'+str(c)+'.png', roi)        
  #plt.imshow(thresholded)
  return letters_list


def predict_image(opencv_img, model):
    #image = Image.open(image_path).convert('RGB')
    pil_img = cv.cvtColor(opencv_img, cv.COLOR_BGR2RGB) #input img is in BGR format from opencv, convert to RGB for PIL
    pil_img = Image.fromarray(pil_img)  #opencv uses numpy array format
    pil_img = ImageChops.invert(pil_img) #Model was trained on white letters with black backgrounds, must make inputs consistent by inverting 
    image_tensor = transformations(pil_img)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    index = output.argmax().item()
    return index


label_letter_dict = {
    '0':'A',
    '1':'B',
    '2':'C',
    '3':'D',
    '4':'E',
    '5':'F',
    '6':'G',
    '7':'H',
    '8':'I',
    '9':'J',
    '10':'K',
    '11':'L',
    '12':'M',
    '13':'N',
    '14':'O',
    '15':'P',
    '16':'Q',
    '17':'R',
    '18':'S',
    '19':'T',
    '20':'U',
    '21':'V',
    '22':'W',
    '23':'X',
    '24':'Y',
    '25':'Z',
}

#predict letters in image and add to text file
def image_to_text(image, num_lines, num_spaces, model):
  prediction = predict_image(image, model)
  letter = label_letter_dict[str(prediction)]
  f = open("./output/text_output.txt", "a")
  text = []
  text.append(letter)
  for l in range(num_lines):
    text.append("\n")
  for s in range(num_spaces):
    text.append(" ")
  text_output = "".join(text)
  f.write(text_output)
  f.close()
  return text_output 

#preprocess image in path, predict, and write prediction to text_output.txt
def evaluate(image_path):
    letters = preprocess_image(image_path)
    output_str = []
    if os.path.exists("./output/text_output.txt"):
        os.remove("./output/text_output.txt")   #remove previous prediction, if any
    for letter_tuple in letters:
        #image_to_text(letter_tuple[0], letter_tuple[1], letter_tuple[2], model)
        curr_output = image_to_text(letter_tuple[0], letter_tuple[1], letter_tuple[2], model)
        output_str.append(curr_output)
    output = "".join(output_str)  #complete string
    return output