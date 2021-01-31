from PIL import Image
import numpy as np
import cv2

for i in range(1,51):
    image = Image.open("/home/jgorospe/testbench/no_personas/np_person (" + str(i) + ").jpg")
    new_image = image.resize((224,224))
    grey_image= new_image.convert('L')
    grey_image.save("/home/jgorospe/testbench/cc_images/np_person_224_uint/np_person_224_uint" + str(i) + ".png")
    pixel_values = list(grey_image.getdata())
    pixel_values = np.array(pixel_values, dtype=np.uint8)
                        
    #image = cv2.imread("/home/jgorospe/testbench/personas/personas (" + str(i) + ").jpg",0)
    #grey_image = cv2.resize(image,(128,128))
    #grey_image= np.int8(grey_image)
    #grey_image= new_image.convert('L')
    #grey_image.save("/home/jgorospe/testbench/cc_images/person_128_uint/person_128_uint" + str(i) + ".png")
    #pixel_values = list(grey_image.getdata())
    #pixel_values = np.array(pixel_values, dtype=np.int8)
    #np.savetxt("/home/jgorospe/car_detection/person.cc", pixel_values, fmt='0x%X, ')
    #pixel_values = grey_image
    
    f= open("/home/jgorospe/testbench/cc_images/np_person_224_uint/np_person_224_uint" + str(i) + ".cc","w+")

    f.write('#include "tensorflow/lite/micro/examples/person_detection/np_person_image_data.h"')
    f.write('\n#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"')
    f.write('\nconst int size = kMaxImageSize;')
    f.write('const unsigned char g_np_person' + str(i) + '[size] = {')

    for by in range(pixel_values.size - 1):
        if(by % 20 == 0):
            f.write('\n')
        f.write('0x%02X, ' % pixel_values[by])

    f.write('0x%02X' % pixel_values[pixel_values.size - 1])
    f.write('\n};')
    f.close()
