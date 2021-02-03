import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs import modeling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from time import time

# solution #1
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

start = time()

# solution #2
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

AUTOTUNE = tf.data.experimental.AUTOTUNE
print("TensorFlow Version: ", tf.__version__)
print("Number of GPU available: ", len(tf.config.experimental.list_physical_devices("GPU")))

IMG_HEIGHT = 100
IMG_WIDTH = 100
BATCH_SIZE = 64
val_fraction = 30
max_epochs = 150
testbatchsize = 64

augment_degree = 0.10
samplesize = [1200, 1600] # old, young
shuffle_buffer_size = 15000  # take first 100 from dataset and shuffle and pick one.

def read_and_label(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    # img = occlude(img, file_path)
    return img, label

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return tf.reshape(tf.where(parts[-4] == CLASS_NAMES), [])

def occlude(image, file_path):
    maskpth = tf.strings.regex_replace(file_path, 'image', 'label')
    mask = tf.io.read_file(maskpth)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float16)
    mask = tf.image.resize(mask, [IMG_WIDTH, IMG_HEIGHT])
    mask = tf.math.greater(mask, 0.25)
    # comment below for cell only
    # mask = tf.math.logical_not(mask)
    maskedimg = tf.where(mask, image, tf.ones(tf.shape(image)))
    return maskedimg

def augment(image, label):
    degree = augment_degree
    if degree == 0:
        return image, label
    image = tf.image.random_hue(image, max_delta=degree, seed=5)
    image = tf.image.random_contrast(image, 1-degree, 1+degree, seed=5)  # tissue quality
    image = tf.image.random_saturation(image, 1-degree, 1+degree, seed=5)  # stain quality
    image = tf.image.random_brightness(image, max_delta=degree)  # tissue thickness, glass transparency (clean)
    image = tf.image.random_flip_left_right(image, seed=5)  # cell orientation
    image = tf.image.random_flip_up_down(image, seed=5)  # cell orientation
    # image = tf.image.random_crop(image, [96,96,3])
    return image, label


def balance(data_dir):
    tmp = [0]
    for CLASS, n in zip(CLASS_NAMES, samplesize):
        secs = [_ for _ in data_dir.glob(CLASS+'/*')]
        for idx, sec in enumerate(secs):
            sec = os.path.join(sec, 'image/*.jpg')
            list_ds = tf.data.Dataset.list_files(sec)
            # subsample
            list_ds = (list_ds
                       .shuffle(shuffle_buffer_size)
                       .take(n)
                       )
            labeled_ds_org = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)
            labeled_ds = (list_ds
                          .map(read_and_label, num_parallel_calls=AUTOTUNE)
                          .map(augment, num_parallel_calls=AUTOTUNE))
            # labeled_ds = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)
            # add augment
            sampleN = len(list(labeled_ds))
            while sampleN < n:
                labeled_ds_aug = (labeled_ds_org
                                  .shuffle(shuffle_buffer_size)
                                  .take(n-sampleN)
                                  .map(augment, num_parallel_calls=AUTOTUNE)
                                  )
                labeled_ds = labeled_ds.concatenate(labeled_ds_aug)
                sampleN = len(list(labeled_ds))
            # print('list_ds: ', len(list(labeled_ds)),CLASS)
            # append
            if tmp[0] == 0:
                tmp[idx] = labeled_ds
            else:
                labeled_ds = tmp[0].concatenate(labeled_ds)
                tmp[0] = labeled_ds
        # print(CLASS, ': sample size =', len(list(tmp[0])))
    return tmp[0].shuffle(shuffle_buffer_size)

# list location of all training images
# train_data_dir = os.path.join(*[os.environ['HOME'], 'Desktop', 'Synology/aging/data/cnn_dataset/train'])
train_data_dir = r'\\kukibox\research\aging\data\cnn_dataset\train'

train_data_dir = pathlib.Path(train_data_dir)
CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if item.name != ".DS_store"])
CLASS_NAMES = sorted(CLASS_NAMES, key=str.lower) #sort alphabetically case-insensitive


train_labeled_ds = balance(train_data_dir)
train_image_count = len(list(train_labeled_ds))
print('training set size : ', train_image_count)
val_image_count = train_image_count // 100 * val_fraction
print('validation size: ', val_image_count)
train_image_count2 = train_image_count-val_image_count
print('training set size after split : ', train_image_count2)

STEPS_PER_EPOCH = train_image_count2 // BATCH_SIZE
VALIDATION_STEPS = val_image_count // BATCH_SIZE

print('train step #',STEPS_PER_EPOCH)
print('validation step #',VALIDATION_STEPS)

plt.figure(figsize=(10,10))
for idx, elem in enumerate(train_labeled_ds.take(100)):
    img = elem[0]
    label = elem[1]
    ax = plt.subplot(10,10,idx+1)
    plt.imshow(img)
    plt.title(CLASS_NAMES[label].title())
    plt.axis('off')
target = 'cnn'
if not os.path.exists(target): os.mkdir(target)
plt.savefig(target + '/aug'+str(np.around(augment_degree*100,decimals=-1))+'_training data.png')
plt.show()


train_ds = (train_labeled_ds
            .skip(val_image_count)
            .shuffle(buffer_size=shuffle_buffer_size)
            .repeat()
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=AUTOTUNE)
            )


val_ds = (train_labeled_ds
          .take(val_image_count)
          .repeat()
          .batch(BATCH_SIZE)
          .prefetch(buffer_size=AUTOTUNE))

testdir = r'\\kukibox\research\aging\data\cnn_dataset\test'


def get_callbacks(name):
    return [
        modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_categorical_crossentropy',
                                         patience=50, restore_best_weights=True),
        # tf.keras.callbacks.TensorBoard(log_dir/name, histogram_freq=1),
        # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/{}/cp.ckpt".format(name),
        #                                    verbose=0,
        #                                    monitor='val_sparse_categorical_crossentropy',
        #                                    save_weights_only=True,
        #                                    save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_crossentropy',
                                             factor=0.1, patience=10, verbose=0, mode='auto',
                                             min_delta=0.0001, cooldown=0, min_lr=0),
    ]

def compilefit(model, name, max_epochs, train_ds, val_ds):
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.losses.CategoricalCrossentropy(from_logits=True), 'accuracy'])
    model_history = model.fit(train_ds,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              epochs=max_epochs,
                              verbose=1,
                              validation_data=val_ds,
                              callbacks=get_callbacks(name),
                              validation_steps=VALIDATION_STEPS,
                              use_multiprocessing=True
                              )
    namename = os.path.dirname(name)
    if not os.path.isdir(os.path.abspath(namename)):
        os.mkdir(os.path.abspath(namename))
    if not os.path.isdir(os.path.abspath(name)):
        os.mkdir(os.path.abspath(name))
    if not os.path.isfile(pathlib.Path(name) / 'full_model.h5'):
        try:
            model.save(pathlib.Path(name) / 'full_model.h5')
        except:
            print('model not saved?')
    return model_history


def plotdf(dfobj, condition, repeat='',lr=None):
    # pd.DataFrame(dfobj).plot(title=condition+repeat)
    dfobj.pop('loss')
    dfobj.pop('val_loss')
    dfobj1 = dfobj.copy()
    dfobj2 = dfobj.copy()
    dfobj.pop('lr')
    dfobj.pop('categorical_crossentropy')
    dfobj.pop('val_categorical_crossentropy')
    pd.DataFrame(dfobj).plot(title=condition+repeat)
    plt.savefig('cnn/' + condition + '/' + repeat + '_accuracy.png')
    dfobj1.pop('lr')
    dfobj1.pop('accuracy')
    dfobj1.pop('val_accuracy')
    pd.DataFrame(dfobj1).plot(title=condition+repeat)
    plt.savefig('cnn/' + condition + '/' + repeat + '_loss.png')
    if lr is not 'decay':
        dfobj2.pop('categorical_crossentropy')
        dfobj2.pop('val_categorical_crossentropy')
        dfobj2.pop('accuracy')
        dfobj2.pop('val_accuracy')
        pd.DataFrame(dfobj2).plot(title=condition+repeat)
        plt.savefig('cnn/' + condition + '/' + repeat + '_lr.png')
    plt.show()


histories = {}
model_dir = 'cnn'


def load_compile(net):
    model = tf.keras.models.load_model(os.path.join(*[model_dir,net,'full_model.h5']),
                                    custom_objects={'KerasLayer': hub.KerasLayer},
                                    compile=False)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def evalmodels(path, model,accuracies):
    datasett, datasettsize = load_dataset(path)
    print('folder : ', os.path.basename(path), ' dataset size : ',datasettsize)
    results = model.evaluate(datasett.batch(testbatchsize).prefetch(buffer_size=AUTOTUNE), verbose=0)
    accuracies.append(np.around(results[-1] * 100, decimals=1))
    return accuracies

# def ds_resize(image,label):
#     # image = tf.image.resize(image,[96,96])
#     image = tf.image.central_crop(image,0.96)
#     return image, label

def load_dataset(dataset_dir):
    dataset_dir = pathlib.Path(dataset_dir)
    test_image_count2 = len(list(dataset_dir.glob('image/*.jpg')))
    list_ds = tf.data.Dataset.list_files(str(dataset_dir / 'image/*.jpg')).shuffle(10000)
    labeled_ds = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)
    # labeled_ds = labeled_ds.map(ds_resize, num_parallel_calls=AUTOTUNE)
    return labeled_ds, test_image_count2

def validateit(mm,t):
    duration = []
    start = time()
    accuracies = []
    print(mm, t)
    m = load_compile(os.path.join(mm, t))
    print('young train')
    accuracies = evalmodels(os.path.join(train_data_dir, 'young/sec001'), m,accuracies)
    accuracies = evalmodels(os.path.join(train_data_dir, 'young/sec003'), m,accuracies)
    accuracies = evalmodels(os.path.join(train_data_dir, 'young/sec007'), m,accuracies)
    accuracies = evalmodels(os.path.join(train_data_dir, 'young/sec010'), m,accuracies)
    accuracies = evalmodels(os.path.join(train_data_dir, 'young/sec016'), m,accuracies)
    accuracies = evalmodels(os.path.join(train_data_dir, 'young/sec019'), m,accuracies)
    print('young test')
    accuracies = evalmodels(os.path.join(testdir, 'young/sec023'), m,accuracies)
    accuracies = evalmodels(os.path.join(testdir, 'young/sec025'), m,accuracies)
    accuracies = evalmodels(os.path.join(testdir, 'young/sec029'), m,accuracies)
    print('old train')
    accuracies = evalmodels(os.path.join(train_data_dir, 'old/sec031'), m,accuracies)
    accuracies = evalmodels(os.path.join(train_data_dir, 'old/sec037'), m,accuracies)
    accuracies = evalmodels(os.path.join(train_data_dir, 'old/sec041'), m,accuracies)
    accuracies = evalmodels(os.path.join(train_data_dir, 'old/sec045'), m,accuracies)
    accuracies = evalmodels(os.path.join(train_data_dir, 'old/sec049'), m,accuracies)
    accuracies = evalmodels(os.path.join(train_data_dir, 'old/sec062'), m,accuracies)
    accuracies = evalmodels(os.path.join(train_data_dir, 'old/sec068'), m,accuracies)
    accuracies = evalmodels(os.path.join(train_data_dir, 'old/sec070'), m,accuracies)
    print('old test')
    accuracies = evalmodels(os.path.join(testdir, 'old/sec076'), m,accuracies)
    accuracies = evalmodels(os.path.join(testdir, 'old/sec078'), m,accuracies)
    accuracies = evalmodels(os.path.join(testdir, 'old/sec082'), m,accuracies)
    accuracies = evalmodels(os.path.join(testdir, 'old/sec088'), m,accuracies)
    end = time()
    print('duration: ', end - start)
    duration.append(end - start)
    accuracies.append(np.around(np.average(accuracies[0:6] + accuracies[9:17]), decimals=1))
    accuracies.append(np.around(np.average(accuracies[6:9] + accuracies[17:21]), decimals=1))
    df.loc[os.path.join(mm, t)] = accuracies
    df.to_csv(csvname)
    print('saved')
    print(df)
    print('validation duration : ', duration)


def evaluateit(network,networkname,repeat, train_ds, val_ds):
    histories[networkname] = compilefit(network, 'cnn/'+networkname+'/'+repeat, max_epochs, train_ds, val_ds)
    plotdf(histories[networkname].history, networkname, repeat)
    validateit(networkname, repeat)


csvname = 'hub.csv'
# csvname = os.path.join(*[os.environ['HOME'], 'Desktop', 'Synology/aging/data/cnn_models', csvname])
csvname = os.path.join(r'\\kukibox\research\aging\data\cnn_models',csvname)


if os.path.exists(csvname):
    print('reading :', csvname)
    df = pd.read_csv(csvname,header=0,index_col=0)
else:
    print('empty')
    df = pd.DataFrame([],columns=[1,3,7,10,16,19,23,25,29,31,37,41,45,49,62,68,70,76,78,82,88,'Train','Test'])


trials = ['t'+str(_)+'_24003200_aug10_cell' for _ in range(1,2)]

# def ds_resize(image,label):
#     # image = tf.image.resize(image,[96,96])
#     image = tf.image.central_crop(image,0.96)
#     return image, label
# train_ds_96 = train_ds.map(ds_resize, num_parallel_calls=AUTOTUNE)
# val_ds_96 = val_ds.map(ds_resize, num_parallel_calls=AUTOTUNE)

duration=[]
# for trial in trials:
#     start = time()
#     # #min input size 76x76
#     MobileNetV2_base = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3),
#                                                 pooling=None,
#                                                 include_top=False,
#                                                 weights='imagenet'
#                                                 )
#     MobileNetV2 = tf.keras.Sequential([
#         MobileNetV2_base,
#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dense(2, activation='softmax')
#     ])
#     evaluateit(MobileNetV2,'MobileNetV2', trial, train_ds, val_ds)
#     end = time()
#     duration.append(end-start)
#     print('train+valid duration : ', end-start)
end = time()

print('temp_immune time: ', np.around(end-start,decimals=1))
for trial in trials:
    start = time()
    # #min input size 76x76
    IncV3_base = tf.keras.applications.InceptionV3(input_shape=(100, 100, 3),
                                                pooling=None,
                                                include_top=False,
                                                weights='imagenet'
                                                )
    IncV3 = tf.keras.Sequential([
        IncV3_base,
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    evaluateit(IncV3,'IncV3',trial,train_ds,val_ds)
    end = time()
    duration.append(end-start)
    print('duration : ', end-start)

# for trial in trials:
#     start = time()
#     # #min input size 76x76
#     ResNet101V2_base = tf.keras.applications.ResNet101V2(input_shape=(96, 96, 3),
#                                                 pooling=None,
#                                                 include_top=False,
#                                                 weights='imagenet'
#                                                 )
#     ResNet101V2 = tf.keras.Sequential([
#         ResNet101V2_base,
#         tf.keras.layers.Dense(2, activation='softmax')
#     ])
#     evaluateit(ResNet101V2,'ResNet101V2_keras_imagenet_col',trial,train_ds,val_ds,test_ds)
#     end = time()
#     duration.append(end-start)
#     print('duration : ', end-start)
print('duration : ', duration)
print('total duration :',np.sum(duration))