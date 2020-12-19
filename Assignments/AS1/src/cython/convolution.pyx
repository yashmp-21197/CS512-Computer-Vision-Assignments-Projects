import numpy as np
cimport numpy as np

np.import_array()

cpdef convolution_2d(np.ndarray input_image, np.ndarray filter_xy):
    cdef np.ndarray output_image = np.zeros((input_image.shape[0],input_image.shape[1]), dtype=np.uint8)
    cdef int image_x, image_y, filter_x, filter_y, img_x, img_y, flt_x, flt_y
    cdef int pixel_val = 0

    for image_x in range(0, input_image.shape[0], 1):
        for image_y in range(0, input_image.shape[1], 1):

            pixel_val = 0

            for filter_x in range(-(filter_xy.shape[0]//2), (filter_xy.shape[0]//2)+1, 1):
                for filter_y in range(-(filter_xy.shape[1]//2), (filter_xy.shape[1]//2)+1, 1):
                    img_x = image_x + filter_x
                    img_y = image_y + filter_y
                    flt_x = filter_x + (filter_xy.shape[0]//2)
                    flt_y = filter_y + (filter_xy.shape[1]//2)
                    if img_x < 0 or img_y < 0 or img_x >= input_image.shape[0] or img_y >= input_image.shape[1]:
                        continue
                    pixel_val += (filter_xy[flt_x][flt_y] * input_image[img_x][img_y])

            output_image[image_x][image_y] = pixel_val

    return output_image
