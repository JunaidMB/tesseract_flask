from PIL import Image
import pytesseract
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from skimage import measure
import skimage.metrics
from matplotlib import pyplot as plt
import cv2 
import numpy
from numpy import asarray
import math

# Configuration of Tesseract
#pytesseract.pytesseract.tesseract = './.virtualenvs/safa_prototype/lib/python3.8/site-packages'

# define pre-trainned model (SRCNN)
def model():
    # model type
    SRCNN = Sequential()

    #adding model layers
    SRCNN.add(Conv2D(filters=128, kernel_size=(9,9), kernel_initializer='glorot_uniform', activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='glorot_uniform', activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size=(5,5), kernel_initializer='glorot_uniform', activation='relu', padding='valid', use_bias=True))

    #define optimiser
    adam = Adam(lr=0.0003)

    #compile model
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return SRCNN

def shave(image, border):
    img = image[border: -border, border: -border]
    return img





def ocr_core(filename):
    """
    This function will handle the core OCR processing
    """

    text = pytesseract.image_to_string(Image.open(filename))

    return text

def ocr_srcnn(filename):
    """
    This function will handle the core OCR processing after srcnn preprocessing.
    """

    # Upload Image
    uploaded_image = Image.open(filename)
    uploaded_image_array = asarray(uploaded_image)

    # Define model
    srcnn = model()
    srcnn.load_weights('3051crop_weight_200.h5')

    #convert image to YCrCb - srcnn trained on Y channel
    temp = cv2.cvtColor(uploaded_image_array, cv2.COLOR_BGR2YCrCb)

    #create image slice and normalise
    Y = numpy.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float)/255

    #perform super resolution with srcnn
    pre = srcnn.predict(Y, batch_size=1)

    #post-process output
    pre *= 255
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)

    #copy Y changel back to image and convert to BGR
    temp = shave(temp, 6)
    temp[:, :, 0] = pre[0, :, :, 0]
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

    # Apply tesseract on the output image
    cleaned_image = Image.fromarray(output)
    text = pytesseract.image_to_string(cleaned_image)
    
    return text

