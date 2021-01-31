from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

interpreter = tf.lite.Interpreter(model_path="mobilenet_v1_224px_proba.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#print(input_details)
#print(output_details)
# Test the model on random input data.
input_shape = input_details[0]['index']
images = []
np_cnt = 0
for i in range(1,51):
    image = cv2.imread("personas/personas (" + str(i) + ").jpg",0)
    new_image = cv2.resize(image,(224,224))
    #new_image= np.int8(new_image)
    #grey_image.save("/home/jgorospe/car_detection/transform/gs_96_96_np_person" + str(i) + ".png")
    #pixel_values = list(grey_image.getdata())
    #input_data = tf.expand_dims(grey_image, 0).numpy()
    #print(input_shape)
    new_image = np.expand_dims(new_image, axis=2)
    images.append(new_image)
    #print(np.shape(images))
    #print(np.shape(new_image))
    #print(np.shape([new_image]))
    interpreter.set_tensor(input_details[0]['index'], [new_image])

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Person " + str(i) + ":")
    print(output_data)
    #output_data = np.uint8(output_data)
    print(output_data)
    if(output_data[0][1] > output_data[0][0]):
        np_cnt = np_cnt +1
    print(np_cnt)

    #np.savetxt("/home/jgorospe/car_detection/person.cc", pixel_values, fmt='0x%X, ')
    #f= open("/home/jgorospe/car_detection/np_person_image/person" + str(i) + ".cc","w+")

    #f.write('#include "tensorflow/lite/micro/examples/person_detection/person_image_data.h"')
    #f.write('\n#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"')
    #f.write('\nconst int size = kMaxImageSize;')
    #f.write('const unsigned char g_person' + str(i) + '[size] = {')

    #for by in range(pixel_values.size - 1):
    #    if(by % 20 == 0):
    #        f.write('\n')
    #    f.write('0x%02X, ' % pixel_values[by])

    #f.write('0x%02X' % pixel_values[pixel_values.size - 1])     
    #f.write('\n};')
    #f.close()
