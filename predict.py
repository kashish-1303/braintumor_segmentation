# #final code
# import numpy as np
# import random
# from glob import glob
# import os
# import SimpleITK as sitk
# from evaluation_metrics import *
# from model import Unet_model
# import traceback

# class Prediction(object):

#     def __init__(self, batch_size_test, load_model_path):
#         self.batch_size_test = batch_size_test
#         unet = Unet_model(img_shape=(240,240,4), load_model_weights=load_model_path)
#         self.model = unet.model
#         print('U-net CNN compiled!\n')

#     def predict_volume(self, filepath_image, show):
#         try:
#             flair = glob(os.path.join(filepath_image, '*Flair*.mha'))
#             t1 = glob(os.path.join(filepath_image, '*T1*.mha'))
#             t1c = glob(os.path.join(filepath_image, '*T1c*.mha'))
#             t2 = glob(os.path.join(filepath_image, '*T2*.mha'))
#             gt = glob(os.path.join(filepath_image, '*more*.mha'))

#             if not all([flair, t1, t1c, t2, gt]):
#                 print(f"Missing one or more modalities for patient: {filepath_image}")
#                 print(f"FLAIR: {len(flair)}, T1: {len(t1)}, T1c: {len(t1c)}, T2: {len(t2)}, GT: {len(gt)}")
#                 print("Files in directory:")
#                 print(os.listdir(filepath_image))
#                 return None, None

#             scans_test = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
#             test_im = [sitk.GetArrayFromImage(sitk.ReadImage(scans_test[i])) for i in range(len(scans_test))]

#             test_im = np.array(test_im).astype(np.float32)
#             test_image = test_im[0:4]
#             gt = test_im[-1]
#             gt[gt==4] = 3

#             print(gt.shape, test_im.shape, test_image.shape)

#             test_image = self.norm_slices(test_image)
            
#             print("test image size:")
#             print(test_image.shape)
#             test_image = test_image.swapaxes(0,1)
#             test_image = np.transpose(test_image, (0,2,3,1))

#             print(test_image.shape)

#             verbose = 1 if show else 0
#             prediction = self.model.predict(test_image, batch_size=self.batch_size_test, verbose=verbose)  
#             print(prediction.shape)
#             prediction = np.argmax(prediction, axis=-1)
#             prediction = prediction.astype(np.uint8)

#             print(prediction.shape)
#             prediction[prediction==3] = 4
#             gt[gt==3] = 4

#             print(prediction.shape, gt.shape)
            
#             return np.array(prediction), np.array(gt)

#         except Exception as e:
#             print(f"Error processing volume {filepath_image}: {str(e)}")
#             print(traceback.format_exc())
#             return None, None

#     def evaluate_segmented_volume(self, filepath_image, save, show, save_path):
#         predicted_images, gt = self.predict_volume(filepath_image, show)
        
#         if predicted_images is None or gt is None:
#             print(f"Skipping evaluation for {filepath_image} due to prediction failure.")
#             return None

#         if save:
#             # Create directories if they don't exist
#             os.makedirs('predictions', exist_ok=True)
#             os.makedirs('predictions/gt', exist_ok=True)

#             tmp = sitk.GetImageFromArray(predicted_images)
#             sitk.WriteImage(tmp, f'predictions/{save_path}.mha')

#             tmp1 = sitk.GetImageFromArray(gt)
#             sitk.WriteImage(tmp1, f'predictions/gt/{save_path}.mha')

#         Dice_complete = DSC_whole(predicted_images, gt)
#         Dice_enhancing = DSC_en(predicted_images, gt)
#         Dice_core = DSC_core(predicted_images, gt)

#         Sensitivity_whole = sensitivity_whole(predicted_images, gt)
#         Sensitivity_en = sensitivity_en(predicted_images, gt)
#         Sensitivity_core = sensitivity_core(predicted_images, gt)
        
#         Specificity_whole = specificity_whole(predicted_images, gt)
#         Specificity_en = specificity_en(predicted_images, gt)
#         Specificity_core = specificity_core(predicted_images, gt)

#         Hausdorff_whole = hausdorff_whole(predicted_images, gt)
#         Hausdorff_en = hausdorff_en(predicted_images, gt)
#         Hausdorff_core = hausdorff_core(predicted_images, gt)

#         if show:
#             print("************************************************************")
#             print(f"Dice complete tumor score : {Dice_complete:.4f}")
#             print(f"Dice core tumor score (tt sauf vert): {Dice_core:.4f}")
#             print(f"Dice enhancing tumor score (jaune): {Dice_enhancing:.4f}")
#             print("**********************************************")
#             print(f"Sensitivity complete tumor score : {Sensitivity_whole:.4f}")
#             print(f"Sensitivity core tumor score (tt sauf vert): {Sensitivity_core:.4f}")
#             print(f"Sensitivity enhancing tumor score (jaune): {Sensitivity_en:.4f}")
#             print("***********************************************")
#             print(f"Specificity complete tumor score : {Specificity_whole:.4f}")
#             print(f"Specificity core tumor score (tt sauf vert): {Specificity_core:.4f}")
#             print(f"Specificity enhancing tumor score (jaune): {Specificity_en:.4f}")
#             print("***********************************************")
#             print(f"Hausdorff complete tumor score : {Hausdorff_whole:.4f}")
#             print(f"Hausdorff core tumor score (tt sauf vert): {Hausdorff_core:.4f}")
#             print(f"Hausdorff enhancing tumor score (jaune): {Hausdorff_en:.4f}")
#             print("***************************************************************\n\n")

#         return np.array((Dice_complete, Dice_core, Dice_enhancing, Sensitivity_whole, Sensitivity_core, Sensitivity_en, Specificity_whole, Specificity_core, Specificity_en, Hausdorff_whole, Hausdorff_core, Hausdorff_en))
    
#     def predict_multiple_volumes(self, filepath_volumes, save, show):
#         results, Ids = [], []
#         print(f"Number of volumes to process: {len(filepath_volumes)}")

#         for patient in filepath_volumes:
#             patient_name = os.path.basename(patient)
#             print(f"Processing Volume ID: {patient_name}")
#             try:
#                 tmp = self.evaluate_segmented_volume(patient, save=save, show=show, save_path=patient_name)
#                 if tmp is not None:
#                     results.append(tmp)
#                     Ids.append(patient_name)
#                     print(f"Successfully processed {patient_name}")
#                 else:
#                     print(f"Skipped {patient_name} due to processing failure")
#             except Exception as e:
#                 print(f"Error processing {patient_name}: {str(e)}")
#                 print(traceback.format_exc())
            
#         res = np.array(results)
#         print(f"Number of successfully processed volumes: {len(results)}")
        
#         if len(results) == 0:
#             print("No volumes were successfully processed.")
#             return
        
#         print("mean : ", np.mean(res, axis=0))
#         print("std : ", np.std(res, axis=0))
#         print("median : ", np.median(res, axis=0))
#         print("25 quantile : ", np.percentile(res, 25, axis=0))
#         print("75 quantile : ", np.percentile(res, 75, axis=0))
#         print("max : ", np.max(res, axis=0))
#         print("min : ", np.min(res, axis=0))

#         np.savetxt('Results.out', res)
#         np.savetxt('Volumes_ID.out', Ids, fmt='%s')

#     def norm_slices(self, slice_not):
#         normed_slices = np.zeros((4, 155, 240, 240))
#         for slice_ix in range(4):
#             normed_slices[slice_ix] = slice_not[slice_ix]
#             for mode_ix in range(155):
#                 normed_slices[slice_ix][mode_ix] = self._normalize(slice_not[slice_ix][mode_ix])
#         return normed_slices    

#     def _normalize(self, slice):
#         b = np.percentile(slice, 99)
#         t = np.percentile(slice, 1)
#         slice = np.clip(slice, t, b)
#         image_nonzero = slice[np.nonzero(slice)]
        
#         if np.std(slice) == 0 or np.std(image_nonzero) == 0:
#             return slice
#         else:
#             tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
#             tmp[tmp == tmp.min()] = -9
#             return tmp

# if __name__ == "__main__":
#     # Set arguments
#     model_to_load = "brain_segmentation\ResUnet.01_1.394.hdf5"
    
#     # Paths for the testing data
#     base_path = os.path.abspath('BRATS2015/training')
#     hgg_path = os.path.join(base_path, 'HGG')
#     lgg_path = os.path.join(base_path, 'LGG')

#     print(f"Base path: {base_path}")
#     print(f"HGG path: {hgg_path}")
#     print(f"LGG path: {lgg_path}")
    
#     print(f"Checking if directories exist:")
#     print(f"Base path exists: {os.path.exists(base_path)}")
#     print(f"HGG path exists: {os.path.exists(hgg_path)}")
#     print(f"LGG path exists: {os.path.exists(lgg_path)}")
    
#     path_HGG = glob(os.path.join(hgg_path, '*'))
#     path_LGG = glob(os.path.join(lgg_path, '*'))
    
#     print(f"Number of HGG volumes found: {len(path_HGG)}")
#     print(f"Number of LGG volumes found: {len(path_LGG)}")
    
#     if len(path_HGG) > 0:
#         print(f"Sample HGG path: {path_HGG[0]}")
#         print("Files in sample HGG folder:")
#         print(os.listdir(path_HGG[0]))

#     if len(path_LGG) > 0:
#         print(f"Sample LGG path: {path_LGG[0]}")
#         print("Files in sample LGG folder:")
#         print(os.listdir(path_LGG[0]))

#     test_path = path_HGG + path_LGG
#     print(f"Total number of volumes: {len(test_path)}")

#     if len(test_path) == 0:
#         print("No volumes found. Please check the directory paths.")
#         exit()

#     np.random.seed(2022)
#     np.random.shuffle(test_path)

#     # Compile the model
#     brain_seg_pred = Prediction(batch_size_test=2, load_model_path=model_to_load)

#     # Predict each volume and save the results in np array
#     brain_seg_pred.predict_multiple_volumes(test_path[:50], save=True, show=True)






#8/8/24 code
import numpy as np 
import SimpleITK as sitk 
import os 
from model import TwoPathwayGroupCNN 
from losses import gen_dice_loss, dice_whole_metric, dice_core_metric, dice_en_metric 
from tensorflow.keras.models import load_model 
from skimage.transform import resize
class Prediction: 
    def __init__(self, model_path, batch_size_test=2): 
        self.model = self.load_model(model_path) 
        self.batch_size_test = batch_size_test 
   
   
   
    def load_model(self, model_path): 
        return load_model(model_path, custom_objects={ 
            'gen_dice_loss': gen_dice_loss, 
            'dice_whole_metric': dice_whole_metric, 
            'dice_core_metric': dice_core_metric, 
            'dice_en_metric': dice_en_metric }) 
   
   
   
    def predict_volume(self, filepath_image, show=False): 
        test_image = self.load_and_preprocess(filepath_image) 
        prediction = self.model.predict(test_image, batch_size=self.batch_size_test, verbose=1 if show else 0) 
        prediction = np.argmax(prediction, axis=-1) 
        prediction[prediction == 3] = 4 
        return prediction 
    
    


    def load_and_preprocess(self, filepath_image):
        image = sitk.ReadImage(filepath_image)
        image_array = sitk.GetArrayFromImage(image)
        
        # Ensure the input has the correct shape (1, 128, 128, 4)
        image_array = np.expand_dims(image_array, axis=0)
        if image_array.shape[-1] != 4:
            image_array = np.repeat(image_array, 4, axis=-1)
        image_array = image_array[:, :128, :128, :]
        
        # Normalize the image
        image_array = (image_array - np.mean(image_array)) / np.std(image_array)
        
        return image_array
   
   
    def save_prediction(self, prediction, output_path): 
        prediction_sitk = sitk.GetImageFromArray(prediction.squeeze().astype(np.uint8))
        sitk.WriteImage(prediction_sitk, output_path) 

if __name__ == "__main__":
    model_path = "brain_segmentation\TwoPathwayGroupCNN.01_12.302.keras"
    predictor = Prediction(model_path)
    
    # Specify the input image file path
    input_path = "imgs/Flair.png"
    output_path = "predictions_twopath/prediction_Flair.png"
    
    print(f"Processing {input_path}...")
    prediction = predictor.predict_volume(input_path, show=True)
    predictor.save_prediction(prediction, output_path)
    print(f"Prediction saved to {output_path}")
