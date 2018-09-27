import keras
import numpy as np

from keras.applications import resnet50
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions

from urllib.request import urlopen, Request

def process_url(url):
    req = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(req) as url:
        with open('temp.jpg', 'wb') as f:
            f.write(url.read())


def model_resnet50(image_batch):
    resnet_model = resnet50.ResNet50(weights='imagenet')
    # prepare the image for the ResNet50 model
    processed_image = resnet50.preprocess_input(image_batch.copy())
    # get the predicted probabilities for each class
    predictions = resnet_model.predict(processed_image)
    # convert the probabilities to class labels
    # output top 3 predictions
    label_resnet = decode_predictions(predictions, top=3)
    return label_resnet

def main():
    # taking input in the form of path or image URL
    path = input("Enter Image path or Image url:\n")
    #first check if it's an URL or not
    temp = path[:4]
    if temp =='http':
        process_url(path)
        filename = 'temp.jpg'
    else:
        filename = path

    # load an image in PIL format
    try:
        original = load_img(filename, target_size=(224, 224))
    except OSError:
        print("please enter correct image path/url")
        return

    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)

    # Convert the image into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)
    result = model_resnet50(image_batch)
    print("{0:s} ({1:.2f}%)\n".format(result[0][0][1], result[0][0][2]*100))

if __name__ == '__main__':
    main()
