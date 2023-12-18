# Machine Learning
For the Dataset: [Link Dataset Makara](https://drive.google.com/file/d/1BOenc8e-p3nTCeXxEPUTYzcqKZqnpuOt/view?usp=drive_link)

We make a machine learning model architecture using CNN Model to classify 7 class of Indonesian Traditional Food such as Molen, Seblak, Sate Padang, Papeda, Nasi Goreng, Kerak Telor, and Bika Ambon. The images consist 1087 files belonging to 7 classes with 870 files for Training and 217 files for Validation. We save the model in h5 file type and we deploy it to Cloud Computing Team.

# Build Image Classification using Transfer Learning (MobileNetV2)
The following are the steps to build image classification using MobileNetV2 transfer learning :

## Steps
1. Download makara.zip dataset from google drive
   ```
   !gdown 1BOenc8e-p3nTCeXxEPUTYzcqKZqnpuOt
   ```
2. Extracting the makara.zip file
   ```
   import os
   import zipfile

   local_zip = './makara.zip'
   zip_ref = zipfile.ZipFile(local_zip, 'r')
   zip_ref.extractall('tmp/dataset')
   zip_ref.close()

   base_dir = 'tmp/dataset/makara/'
   ```
