import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# CALIBRATION LIST OF FILES
calibration_list_of_files = "ILSVRC2012_CalibrationSet.txt"
calibration_images_folder = "./images/ILSVRC2012_img_cal"
calibration_groundtruth = "ILSVRC2012_calibration_ground_truth.txt"

# IMAGENET LABELS
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = open(labels_path).read().splitlines()[1:]

# LOAD CALIBRATION FILES AND GROUND TRUTH
with open(calibration_list_of_files, 'r') as f:
    cal_files = [line.strip() for line in f]

with open(calibration_groundtruth, 'r') as f:
    cal_gt = [int(line.strip()) for line in f]

# NUMBER OF IMAGES TO EVALUATE
limit = None

# LIMIT IS OPTIONAL
if limit is not None:
    cal_files = cal_files[:limit]
    cal_gt = cal_gt[:limit]

# MODELS TO EVALUATE
models = [
    ("VGG19", tf.keras.applications.VGG19(weights="imagenet"), 224), # VGG19
    ("ResNet50", tf.keras.applications.ResNet50(weights="imagenet"), 224), # ResNet50
    ("VGG16", tf.keras.applications.VGG16(weights="imagenet"), 224), # VGG16
    ("InceptionV3", tf.keras.applications.InceptionV3(weights="imagenet"), 299), # InceptionV3
    ("EfficientNetB7", tf.keras.applications.EfficientNetB7(weights="imagenet"), 600), # EfficientNetB7
]

# PREPROCESS IMAGE
def preprocess_image(image_path, model_name, image_res):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_res, image_res))
    x = tf.keras.preprocessing.image.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    
    if model_name == "InceptionV3":
        x = tf.keras.applications.inception_v3.preprocess_input(x)
    elif model_name == "EfficientNetB7":
        x = tf.keras.applications.efficientnet.preprocess_input(x)
    elif model_name == "ResNet50":
        x = tf.keras.applications.resnet50.preprocess_input(x)
    elif model_name == "VGG16":
        x = tf.keras.applications.vgg16.preprocess_input(x)
    else:
        x = tf.keras.applications.vgg19.preprocess_input(x)
    
    return x

# STORES METRICS
model_names = []
precisions = []
recalls = []
fscores = []
accuracies = []
inference_times = []

# CYCLE TO EVALUATE EACH MODEL
for model_name, model, image_res in models:
    cal_pred = []
    start_time = time.time()

    # # Determine the grid size
    # grid_size = int(np.ceil(np.sqrt(limit)))
    # fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    # fig.suptitle(f"Model: {model_name}", fontsize=20)

    # # Flatten the axes array for easier iteration
    # axes = axes.flatten()

    # Perform inference on calibration set
    for i, file in enumerate(cal_files):
        print(f"------------------------")
        print(f"Processing image {i+1}/{len(cal_files)} with {model_name}: {file}")
        image_path = os.path.join(calibration_images_folder, file)
        x = preprocess_image(image_path, model_name, image_res)
        
        result = model.predict(x, verbose=0)
        predicted_class = np.argmax(result[0], axis=-1)
        cal_pred.append(predicted_class)

        # Display images with predictions
        predicted_class_name = imagenet_labels[predicted_class]
        gt_class_name = imagenet_labels[cal_gt[i]]
        img = tf.keras.preprocessing.image.load_img(image_path)

        # axes[i].imshow(img)
        # axes[i].axis('off')
        # axes[i].set_title(f"GT: {gt_class_name} ({cal_gt[i]})\nPred: {predicted_class_name} ({predicted_class})", fontsize=12, pad=20)

        # PRINT PREDICTED CLASS NAME AND GROUND TRUTH CLASS NAME
        print(f"GT: {gt_class_name} ({cal_gt[i]})")
        print(f"Pred: {predicted_class_name} ({predicted_class})")
        print(f"------------------------")

    # HIDE UNUSED SUBPLOTS
    # for j in range(i + 1, len(axes)):
    #     axes[j].axis('off')

    # plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
    # plt.show()

    end_time = time.time()
    inference_time = end_time - start_time
    fps = len(cal_files) / inference_time

    # CALCULATE METRICS
    precision, recall, fscore, _ = precision_recall_fscore_support(cal_gt, cal_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(cal_gt, cal_pred)

    # STORE METRICS
    model_names.append(model_name)
    precisions.append(precision)
    recalls.append(recall)
    fscores.append(fscore)
    accuracies.append(accuracy)
    inference_times.append(fps)

    # PRINT RESULTS
    print(f"Evaluation Results for {model_name} on ImageNet Calibration Set")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-Score: {fscore:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Inference Frame Rate: {fps:.2f} fps\n")

# PLOT THE METRICS
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# PRECISION
ax[0, 0].bar(model_names, precisions, color='blue')
ax[0, 0].set_title('Precision')
ax[0, 0].set_ylim([0, 1])

# RECALL
ax[0, 1].bar(model_names, recalls, color='green')
ax[0, 1].set_title('Recall')
ax[0, 1].set_ylim([0, 1])

# F-SCORE
ax[1, 0].bar(model_names, fscores, color='red')
ax[1, 0].set_title('F-Score')
ax[1, 0].set_ylim([0, 1])

# ACCURACY
ax[1, 1].bar(model_names, accuracies, color='purple')
ax[1, 1].set_title('Accuracy')
ax[1, 1].set_ylim([0, 1])

# plt.tight_layout()
plt.show()
