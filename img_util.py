#Some Utility Functions used in tkinter window
import numpy as np

class img_util:

    def subsample(image):
        #Use array slicing to delete every other row
        #subsampled_image = image[::2, ::2]

        rows, columns = image.shape
        subsampled_image = np.zeros((rows // 2, columns // 2), dtype=np.uint8)
        for i in range(0, rows, 2):
            for j in range(0, columns, 2):
                subsampled_image[i // 2, j // 2] = image[i, j]
                #print(image[i, j])

        return subsampled_image
    
    def reshape_for_model(img_array):
        arr_for_prediction = np.array([img_array])
        arr_for_prediction = arr_for_prediction.reshape(arr_for_prediction.shape[0], arr_for_prediction.shape[1], arr_for_prediction.shape[2], 1)

        arr_for_prediction = np.array(arr_for_prediction, dtype=np.float32)
        arr_for_prediction = arr_for_prediction / 255.0

        arr_for_prediction = 1 - arr_for_prediction

        return arr_for_prediction