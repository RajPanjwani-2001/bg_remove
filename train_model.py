from mrcnn_demo.m_rcnn import *
import tensorflow as tf
from mrcnn_demo import visualize
import imageio

device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])

dataset = "C:/Users/Raj/MyProg/s20/dataset"
annotations_path = "C:/Users/Raj/MyProg/s20/car_bg.json"

dataset_train = load_image_dataset(annotations_path, dataset,"train")
print('Train: %d' % len(dataset_train.image_ids))
class_number = dataset_train.count_classes()
print("Classes: {}".format(class_number))

#display_image_samples(dataset_train)
# Load Configuration
config = CustomConfig(class_number)
# config.display()

'''model = load_training_model(config)
train_head(model, dataset_train, dataset_train, config)'''


dataset_val = "C:/Users/Raj/MyProg/s20/test_images"
inf_model = "C:/Users/Raj/MyProg/s20/mask_rcnn_object_0014.h5"

class_names = ['BG','car']
test_model, inference_config = load_inference_model(class_number, inf_model)

file_names = next(os.walk(dataset_val))[2]
image = imageio.imread(os.path.join(dataset_val, random.choice(file_names)))
print(file_names)
# Run detection
results = test_model.detect([image], verbose=1)

# Visualize results
r = results[0]
print(len(r['class_ids']))
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])