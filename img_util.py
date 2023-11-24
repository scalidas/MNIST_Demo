#Some Utility Functions used in tkinter window
import numpy as np

class img_util:

    def subsample(self, image):
        #Use array slicing to delete every other row
        subsampled_image = image[::2, ::2]

        return subsampled_image
    
    def reshape_for_model(self, img_array):
        arr_for_prediction = np.array([img_array])
        arr_for_prediction = arr_for_prediction.reshape(arr_for_prediction.shape[0], arr_for_prediction.shape[1], arr_for_prediction.shape[2], 1)

        arr_for_prediction = np.array(arr_for_prediction, dtype=np.float32)
        arr_for_prediction = arr_for_prediction / 255.0

        arr_for_prediction = 1 - arr_for_prediction

        return arr_for_prediction