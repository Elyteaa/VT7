from PIL import Image
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def labelsToWords(label):
    x = np.argmax(label)
    if x==0:
        return 'badminton'
    elif x==1:
        return 'bocce'
    elif x==2:
        return 'croquet'
    elif x==3:
        return 'polo'
    elif x==4:
        return 'rock climbing'
    elif x==5:
        return 'rowing'
    elif x==6:
        return 'sailing'
    else:
        return 'snowbording'




model = tf.keras.models.load_model('model.h5')
img_size = model.layers[0].output.shape[1]
predictions = []
i = 0
for img in os.listdir('./predictions'):
    path = os.path.join('./predictions', img)
    img = Image.open(path)
    img = np.array(img.resize((img_size, img_size), Image.ANTIALIAS))
    img = np.true_divide(np.expand_dims(img, 0), 255.0)
    predictedLabel = model.predict(img)
    predictions.append([img, labelsToWords(predictedLabel)])
    imgplot = plt.imshow(predictions[i][0].squeeze())
    plt.show()
    print('Label: ' + str(predictions[i][1]))
    i += 1
