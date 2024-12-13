# cs163-finalproject

This is the final project for the course CS163.

to run the code, simply open up the ipynb file on colab and run it cell by cell. 

Since training and downloading dataset is very time consuming. We have also included the trained weight called test2 as part of the subbmission. 

To run test2, simply include test2 in the same directory and load the model with model.load_state_dict()

To run it on your own snowy/rainy images, run 
imName = "(name of the img)"
imSnow = cv2.imread(imageSnowName)
imSnow = cv2.cvtColor(imSnow, cv2.COLOR_BGR2RGB)
tensorImgSnow = tf.to_tensor(imSnow)
DesnowImage(model, imSnow, tensorImgSnow)

