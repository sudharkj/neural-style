# Taken from https://www.tensorflow.org/lite/examples/style_transfer/overview
from pathlib import Path

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.autolayout'] = True

style_predict_url = 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1' \
                    '?lite-format=tflite '
style_transform_url = 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1' \
                      '?lite-format=tflite '
style_predict_path = tf.keras.utils.get_file('style_predict.tflite', style_predict_url)
style_transform_path = tf.keras.utils.get_file('style_transform.tflite', style_transform_url)


def load_image(path_to_img):
    """
    Function to load an image from a file, and add a batch dimension.
    :param path_to_img: tf path to the image
    :return: image as a tensor
    """
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]

    return img


def preprocess_image(image, target_dim):
    """
    Function to pre-process by resizing an central cropping it.
    :param image: image in the form of a tensor
    :param target_dim: size to which the image has to resized
    :return: preprocess image in the form of tensor
    """
    # Resize the image so that the shorter dimension becomes 256px.
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)

    # Central crop the image.
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

    return image


def save_image(image, title, path):
    """
    Save images at a path
    :param image: image to be saved
    :param title: title of the image
    :param path: path where to save
    :return: None
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.clf()
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(f"{path}/{title}.jpg")


def run_style_predict(preprocessed_style_image):
    """
    Function to run style prediction on preprocessed style image.
    :param preprocessed_style_image: preprocessed style image
    :return: style bottleneck (prediction on preprocessed style image)
    """
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_predict_path)

    # Set model input.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

    # Calculate style bottleneck.
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
    )()

    return style_bottleneck


def run_style_transform(style_bottleneck, preprocessed_content_image):
    """
    Run style transform on preprocessed style image
    :param style_bottleneck: style bottleneck (prediction of the preprocessed style image)
    :param preprocessed_content_image: preprocessed content image
    :return: styled image
    """
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_transform_path)

    # Set model input.
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()

    # Set model inputs.
    interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()

    # Transform content image.
    stylized_image = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
    )()

    return stylized_image


def create_blended_images(content_url, style_url, count=5, tag=None):
    """
    Creates a blended images and returns the path of the related folder.
    :param content_url: url for content and style image
    :param style_url: url for style image
    :param count: number of images to return
    :param tag: optional tag that is to be used, if not provided uses timestamp as a tag
    :return: path where the blended images are present
    """
    content_image_size = 384
    style_image_size = 256
    if tag is None:
        tag = time.time()

    # create base folder
    base_path = f"data/{tag}"
    Path(f"{base_path}").mkdir(parents=True, exist_ok=True)

    content_path = tf.keras.utils.get_file(f'{tag}_content.jpg', content_url)
    style_path = tf.keras.utils.get_file(f'{tag}_style.jpg', style_url)
    # Load the input images.
    content_image = load_image(content_path)
    style_image = load_image(style_path)

    # Preprocess the input images.
    preprocessed_content_image = preprocess_image(content_image, content_image_size)
    preprocessed_style_image = preprocess_image(style_image, style_image_size)
    print('Style Image Shape:', preprocessed_style_image.shape)
    print('Content Image Shape:', preprocessed_content_image.shape)

    # Save the preprocessed images
    save_image(preprocessed_content_image, 'content_image', base_path)
    save_image(preprocessed_style_image, 'style_image', base_path)

    # Calculate style bottleneck for the preprocessed style image.
    style_bottleneck = run_style_predict(preprocessed_style_image)
    print('Style Bottleneck Shape:', style_bottleneck.shape)

    # Stylize the content image using the style bottleneck.
    stylized_image = run_style_transform(style_bottleneck, preprocessed_content_image)

    # Visualize the output.
    save_image(stylized_image, 'stylized_image', base_path)

    # Calculate style bottleneck of the content image.
    style_bottleneck_content = run_style_predict(
        preprocess_image(content_image, style_image_size)
    )

    start_ratio = 0.0
    end_ratio = 1.0
    base_path = f"{base_path}/blended"
    Path(f"{base_path}").mkdir(parents=True, exist_ok=True)
    for index, content_blending_ratio in enumerate(np.linspace(start_ratio, end_ratio, count)):
        # Blend the style bottleneck of style image and content image
        style_bottleneck_blended = content_blending_ratio * style_bottleneck_content + (
            1 - content_blending_ratio) * style_bottleneck

        # Stylize the content image using the style bottleneck.
        stylized_image_blended = run_style_transform(style_bottleneck_blended, preprocessed_content_image)

        # plt.subplot(images, 1, index+1)
        # Visualize the output.
        save_image(stylized_image_blended, f'image_{index + 1}', base_path)
    return base_path


if __name__ == '__main__':
    print(f"tensorflow version: {tf.__version__}")
    _content_url = 'https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/belfry' \
                   '-2611573_1280.jpg '
    _style_url = 'https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/style23.jpg'
    _blended_images_path = create_blended_images(_content_url, _style_url)
    print(f"Generated images path: {_blended_images_path}")
