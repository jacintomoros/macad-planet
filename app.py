from flask import Flask, render_template, request
# from flask_ngrok import run_with_ngrok
import os

import tensorflow as tf
import numpy as np

# import matplotlib.pyplot as plt
import os

import cv2 as cv

import base64

from PIL import Image

#Initialize the useless part of the base64 encoded image.
init_Base64 = 21

app = Flask(__name__)

image_folder = os.path.join('static', 'images')
app.config["UPLOAD_FOLDER"] = image_folder

@app.route('/', methods=['GET'])
def home():
  my_path_prediction = './static/images/'
  my_file_prediction = 'blank_space.png'
  pic = os.path.join(my_path_prediction, my_file_prediction)
  
  return render_template('index.html', user_pixel=pic, user_image=pic)
  # return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
  #Preprocess the image : set the image to 28x28 shape
  #Access the image
  draw = request.form['url']
  #Removing the useless part of the url.
  draw = draw[init_Base64:]
  #Decoding
  draw_decoded = base64.b64decode(draw)
  image = np.asarray(bytearray(draw_decoded), dtype="uint8")
  imagefile = cv.imdecode(image, cv.IMREAD_COLOR)
  # imagefile = Image.fromarray(imagefile)

  image_path = './static/images/'
  image_path_file = 'example.png'
  cv.imwrite(os.path.join(image_path, image_path_file), imagefile)
  # imagefile.save(image_path)

  my_path = './static/images/example.png'
  large_image_stack = cv.imread(my_path)
  width, height = 256, 256
  imgResize = cv.resize(large_image_stack,(width,height))

  new_my_path = './static/resize/'
  new_my_file = 'graph.png'
  cv.imwrite(os.path.join(new_my_path, new_my_file), imgResize)

  def load(image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image 
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

  # Each image is 256x256 in size
  IMG_WIDTH = 256
  IMG_HEIGHT = 256

  def resize(input_image, real_image, height, width):
      input_image = tf.image.resize(input_image, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      real_image = tf.image.resize(real_image, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      return input_image, real_image


  # Normalizing the images to [-1, 1]
  def normalize(input_image, real_image):
      input_image = (input_image / 127.5) - 1
      real_image = (real_image / 127.5) - 1
      return input_image, real_image


  def load_image_test(image_file):
      input_image, real_image = load(image_file)
      input_image, real_image = resize(input_image, real_image,IMG_HEIGHT, IMG_WIDTH)
      input_image, real_image = normalize(input_image, real_image)
      return input_image, real_image

  OUTPUT_CHANNELS = 3

  def downsample(filters, size, apply_batchnorm=True):
      initializer = tf.random_normal_initializer(0., 0.02)

      result = tf.keras.Sequential()
      result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                            kernel_initializer=initializer, use_bias=False))

      if apply_batchnorm:
          result.add(tf.keras.layers.BatchNormalization())

      result.add(tf.keras.layers.LeakyReLU())

      return result

  def upsample(filters, size, apply_dropout=False):
      initializer = tf.random_normal_initializer(0., 0.02)

      result = tf.keras.Sequential()
      result.add(
          tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  use_bias=False))

      result.add(tf.keras.layers.BatchNormalization())

      if apply_dropout:
          result.add(tf.keras.layers.Dropout(0.5))

      result.add(tf.keras.layers.ReLU())

      return result

  def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
      downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
      downsample(128, 4),  # (batch_size, 64, 64, 128)
      downsample(256, 4),  # (batch_size, 32, 32, 256)
      downsample(512, 4),  # (batch_size, 16, 16, 512)
      downsample(512, 4),  # (batch_size, 8, 8, 512)
      downsample(512, 4),  # (batch_size, 4, 4, 512)
      downsample(512, 4),  # (batch_size, 2, 2, 512)
      downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
      upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
      upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
      upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
      upsample(512, 4),  # (batch_size, 16, 16, 1024)
      upsample(256, 4),  # (batch_size, 32, 32, 512)
      upsample(128, 4),  # (batch_size, 64, 64, 256)
      upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                          strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
      x = down(x)
      skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

  generator = Generator()

  def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

  discriminator = Discriminator()

  generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  my_path_checkpoint = './static/checkpoint/'

  checkpoint_path = my_path_checkpoint
  checkpoint_dir = checkpoint_path
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)

  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  def add_margin(pil_img, top, right, bottom, left, color):
      width, height = pil_img.size
      new_width = width + right + left
      new_height = height + top + bottom
      result = Image.new(pil_img.mode, (new_width, new_height), color)
      result.paste(pil_img, (left, top))
      return result

  im = Image.open(os.path.join(new_my_path, new_my_file))
  x,y = im.size
  im_new = add_margin(im, 0, 0, 0, x, (255,255,255))

  new_my_path_padding = './static/padding/'
  new_my_file_padding = 'graphpadding.png' 
  im_new.save(os.path.join(new_my_path_padding, new_my_file_padding))

  try:
      test_dataset_web = tf.data.Dataset.list_files(os.path.join(new_my_path_padding, new_my_file_padding), shuffle=False)
  except tf.errors.InvalidArgumentError:
      test_dataset_web = tf.data.Dataset.list_files(os.path.join(new_my_path_padding, new_my_file_padding), shuffle=False)
  test_dataset_web = test_dataset_web.map(load_image_test)
  test_dataset_web = test_dataset_web.batch(1)

  def generate_images2(model, test_input):
      prediction = model(test_input, training=True)
      # plt.figure(figsize=(15, 15))

      display_list = [test_input[0], prediction[0]]
      title = ['Input Image', 'Predicted Image']

      # for i in range(2):
      #     plt.subplot(1, 3, i+1)
      #     plt.title(title[i])
      #     # Getting the pixel values in the [0, 1] range to plot.
      #     plt.imshow(display_list[i] * 0.5 + 0.5)
      #     plt.axis('off')
      # # plt.show()
  
  all_predictions = []
  for inp, tar in test_dataset_web:
      generate_images2(generator, inp)
      prediction = generator(inp, training=True)
      all_predictions.append(prediction[0])
  
  my_path_prediction = './static/prediction/'
  my_file_prediction = 'prediction.png'

  for iteration, item in enumerate(all_predictions):
        tf.keras.preprocessing.image.save_img(os.path.join(my_path_prediction, my_file_prediction),item)

  pic1 = os.path.join(image_path, image_path_file)
  pic2 = os.path.join(my_path_prediction, my_file_prediction)
  
  return render_template('index.html', user_pixel=pic1, user_image=pic2)

if __name__== "__main__":
    app.run(debug=True, threaded=True)