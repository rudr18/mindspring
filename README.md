# MindSpring (Skin Disease Detection)

## Dataset

The dataset used for training the model is a large collection of skin disease images. You can download the dataset from Kaggle using the following link:

[Kaggle Dataset Link](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)

After downloading the dataset, extract it to a directory of your choice. This directory will be used as the `data_dir` in the model training script.

Run `modelScript.py` to train the model `skin_mindspring.h5`.

After this, in `app.py`, which is a Flask Python script, load the model using the following command:

```
from tensorflow.keras.models import load_model
```

Also, make a folder named uploads where all the uploaded disease images will be stored.

Create  a `templates` folder where the `index.html` will be stored.

After completing these steps, run the script with the following command:

```
python app.py
```
