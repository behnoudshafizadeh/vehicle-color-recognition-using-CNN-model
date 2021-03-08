# vehicle-color-recognition-using-CNN-model

## discription
> in this project,we have used CNN architecture for detecting vehicle color,the colors are used in this project :

> Beige,Black,Blue,Brown,Gray,Green,Orange,Red,Silver,White,Yellow.

> Hint:dataset and training files are not available,Until the paper related this project will be published.Meanwhile you are able to use test models for predicting vehicle color by using model weight `color_model.h5` and prediction code in google colab `color_prediction.ipynb`. 

> building an input file `input` containing images.

## road map for testing code

> first-step : mount your google colab dirve by using below instructions:
>
```
from google.colab import drive
drive.mount('/content/drive/')
import os
os.chdir("/content/drive/MyDrive/train python/IMAGE-AI")
!ls
```
> seconde-step : install requirements in google colab
```
!pip install -r requirements.txt
```
> third-step : importing libraries
```
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
```
> fourth-step : identifying image input size and defining list of colors
```
img_width, img_height = 224, 224
CATEGORIES=['beige','black','blue', 'brown', 'gray' ,'green', 'orange','red', 'silver', 'white','yellow']
```
> fifth-step : load model weights
```
model = load_model('color_model.h5')
```
> last-step : after builing input images file,you must run below instructions:
```
path="/content/drive/MyDrive/train python/IMAGE-AI/input"
x=os.listdir(path)
print(x)
for img in x:
  test_image = image.load_img(os.path.join(path,img), target_size=(img_width, img_height,3))
  test_image = image.img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis=0)
  test_image = test_image.reshape(1,img_width, img_height,3)
  result = model.predict_classes(test_image, batch_size=1)
  print("this car in "+img+"is:", CATEGORIES[result[0]])
  print("---------------------")
 ```
 
 ## Result :
 
![Capture1](https://user-images.githubusercontent.com/53394692/110308394-4d087300-8015-11eb-8c65-1f05d6c791d7.PNG)






