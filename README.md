# Machine Learning 🤖📊
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
3. Defines the path to the base directory for various food classes.
   ```
   bikbon_dir = os.path.join(base_dir, 'bika ambon')
   kelor_dir = os.path.join(base_dir, 'kerak telor')
   molen_dir = os.path.join(base_dir, 'molen')
   nasgor_dir = os.path.join(base_dir, 'nasi goreng')
   papeda_dir = os.path.join(base_dir, 'papeda maluku')
   sate_dir = os.path.join(base_dir, 'sate padang')
   seblak_dir = os.path.join(base_dir, 'seblak')
   ```
4. Get a list of files from the base directory for each food class.
   ```
   bikbon_names = os.listdir(bikbon_dir)
   print(f'BIKA AMBON: {bikbon_names[:10]}')

   kelor_names = os.listdir(kelor_dir)
   print(f'KERAK TELOR: {kelor_names[:10]}')

   molen_names = os.listdir(molen_dir)
   print(f'MOLEN: {molen_names[:10]}')

   nasgor_names = os.listdir(nasgor_dir)
   print(f'NASI GORENG: {nasgor_names[:10]}')

   papeda_names = os.listdir(papeda_dir)
   print(f'PAPEDA: {papeda_names[:10]}')

   sate_names = os.listdir(sate_dir)
   print(f'SATE PADANG: {sate_names[:10]}')

   seblak_names = os.listdir(seblak_dir)
   print(f'SEBLAK: {seblak_names[:10]}')
   ```
5. Display the total number of images for each food class.
   ```
   print(f'total bika ambon images: {len(os.listdir(bikbon_dir))}')
   print(f'total kerak telor images: {len(os.listdir(kelor_dir))}')
   print(f'total molen images: {len(os.listdir(molen_dir))}')
   print(f'total nasi goreng images: {len(os.listdir(nasgor_dir))}')
   print(f'total papeda maluku images: {len(os.listdir(papeda_dir))}')
   print(f'total sate padang images: {len(os.listdir(sate_dir))}')
   print(f'total seblak images: {len(os.listdir(seblak_dir))}')
   ```
6. Display the image plotting
   ```
   # Use %matplotlib inline to display image plots inside the notebook.
   %matplotlib inline

   # Import the matplotlib library for image plotting.
   import matplotlib.pyplot as plt
   import matplotlib.image as mpimg

   # Specifies the number of rows and columns for the image plot layout.
   nrows = 14
   ncols = 14

   # Specify the index of the start image or other values that correspond to the number of pictures in each class.
   pic_index = 0
   ```
   ```
   # Set the size of the plot image according to the predefined number of rows and columns.
   fig = plt.gcf()
   fig.set_size_inches(ncols * 4, nrows * 4)

   # Add an image index to view the next image on the plot.
   pic_index += 14

   # Collect image paths for specific food classes from the training dataset.
   next_bikbon_pix = [os.path.join(bikbon_dir, fname)
                for fname in bikbon_names[pic_index-8:pic_index]]

   next_kelor_pix = [os.path.join(kelor_dir, fname)
                for fname in kelor_names[pic_index-8:pic_index]]

   next_molen_pix = [os.path.join(molen_dir, fname)
                for fname in molen_names[pic_index-8:pic_index]]

   next_nasgor_pix = [os.path.join(nasgor_dir, fname)
                for fname in nasgor_names[pic_index-8:pic_index]]

   next_papeda_pix = [os.path.join(papeda_dir, fname)
                for fname in papeda_names[pic_index-8:pic_index]]

   next_sate_pix = [os.path.join(sate_dir, fname)
                for fname in sate_names[pic_index-8:pic_index]]

   next_seblak_pix = [os.path.join(seblak_dir, fname)
                for fname in seblak_names[pic_index-8:pic_index]]

   # Display the next images on the plot.
   for i, img_path in enumerate(next_bikbon_pix+next_kelor_pix+next_molen_pix+
                             next_nasgor_pix+next_papeda_pix+next_sate_pix+next_seblak_pix):
   sp = plt.subplot(nrows, ncols, i + 1)
   sp.axis('Off')

   img = mpimg.imread(img_path)
   plt.imshow(img)

   # Display the image plot.
   plt.show()
   ```
7. Import the library needed to process the model
   ```
   import tensorflow as tf
   from tensorflow.keras.models import Sequential, Model
   from tensorflow.keras import layers
   from tensorflow.keras.layers import Dense, Flatten
   from tensorflow.keras.applications import MobileNetV2
   ```
8. Initialize the base model using transfer learning (MobileNetV2)
   ```
   base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
   base_model.trainable = False
   ```
9. Checking a summary of the model architecture of the transfer learning used.
    ```
    base_model.summary()
    ```
10. Add some layers in the output layer section
    ```
    model = Sequential([
       layers.Rescaling(1./255., input_shape=(224,224,3)),
       layers.RandomFlip(mode='horizontal'),
       layers.RandomRotation(0.2),
       base_model,
       layers.Flatten(),
       layers.Dense(128, activation='relu'),
       layers.Dropout(0.2),
       layers.Dense(7, activation='softmax')
    ])
    model.summary()
    ```
11. Initialize the optimizer, loss, and metrics in the compiled model
    ```
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    ```
12. Load image dataset from a directory
    ```
    import tensorflow as tf
    from tensorflow.keras.utils import image_dataset_from_directory

    # Load training data using image_dataset_from_directory.
    training = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        color_mode='rgb',
        batch_size=128,
        image_size=(224, 224),
        shuffle=True,
        label_mode='categorical',
        subset = 'training',
        validation_split=0.2,
        seed=42
    )

    # Load validation data using image_dataset_from_directory.
    validation = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        color_mode='rgb',
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        label_mode='categorical',
        subset = 'validation',
        validation_split=0.2,
        seed=42
    )
    ```
13. Train the model
    ```
    history = model.fit(
       training,
       steps_per_epoch=7,
       validation_data=validation,
       validation_steps=7,
       epochs=10,
       verbose=2
    )
    ```
14. Checking the accuracy of training and validation as well as the loss of training and validation using line charts.
    ```
    import pandas as pd

    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.show()
    ```
15. Test or predict the Model by uploading an image to get the appropriate accuracy
    ```
    import numpy as np
    from google.colab import files
    from tensorflow.keras.utils import load_img, img_to_array
    import matplotlib.image as mpimg

    uploaded = files.upload()

    for fn in uploaded.keys():

    # predicting images
    path = fn
    img = load_img(path, target_size=(224, 224))

    imgplot = plt.imshow(img)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images, batch_size=8)
    print(fn)

    print('Bika Ambon', '{:.4f}'.format(classes[0,0]))
    print('Kerak Telor', '{:.4f}'.format(classes[0,1]))
    print('Molen', '{:.4f}'.format(classes[0,2]))
    print('Nasi Goreng', '{:.4f}'.format(classes[0,3]))
    print('Papeda Maluku', '{:.4f}'.format(classes[0,4]))
    print('Sate Padang', '{:.4f}'.format(classes[0,5]))
    print('Seblak', '{:.4f}'.format(classes[0,6]))
    ```
16. Save the model in .h5 format
    ```
    model.save('makara.h5')
    ```

## License
[Makara's MIT License](https://github.com/makaraapps/MachineLearning/blob/main/LICENSE)
