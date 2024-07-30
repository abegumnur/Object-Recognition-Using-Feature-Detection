import os
import cv2
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_images(object_name):
    base_name = '_'.join(object_name.split('_')[:-1])
    object_path = os.path.join("C:/Users/BEGUM/PycharmProjects/pythonProject2/dataset", object_name, "rgbd-dataset", base_name, object_name)
    images = []
    for file_name in os.listdir(object_path):
        if file_name.endswith('_crop.png'):
            img_path = os.path.join(object_path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images

def extract_features(detector, images):
    all_descriptors = []
    for img in images:
        keypoints, descriptors = detector.detectAndCompute(img, None)
        if descriptors is not None:
            all_descriptors.extend(descriptors)
    return all_descriptors

def augment_image(img):
    flipped_img = cv2.flip(img, 1)
    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return [img, flipped_img, rotated_img]

object_names = ["lime_1", "bowl_4", "lemon_1", "instant_noodles_1", "ball_1", "orange_2", "toothpaste_2", "peach_3", "water_bottle_1", "scissors_3"]
detectors = {
    'SIFT': cv2.SIFT_create(),
    'ORB': cv2.ORB_create(),
    'AKAZE': cv2.AKAZE_create(),
    'BRISK': cv2.BRISK_create()
}

results = defaultdict(lambda: defaultdict(int))  # Store results for each object and detector

for name, detector in detectors.items():
    X = []
    y = []
    print(f"Processing with {name} detector...")
    start_time = time.time()
    for obj in object_names:
        print(f"Processing object: {obj}")
        images = load_images(obj)
        if images:
            augmented_images = []
            for img in images:
                augmented_images.extend(augment_image(img))
            print(f"Extracting features with {name}...")
            features = extract_features(detector, augmented_images)
            if features:
                X.extend(features)
                y.extend([obj] * len(features))
                results[obj][name] = len(features)
        else:
            print(f"No images found for object: {obj}")

    if not X or not y:
        raise ValueError(f"No samples or labels found for detector {name}. Please check the dataset and feature extraction process.")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split the data
    print("Splitting the data...")
    split_start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    print(f"Data splitting completed in {time.time() - split_start_time:.2f} seconds.")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Define classifier
    print("Defining the classifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

    # Train classifier
    print("Training the classifier...")
    train_start_time = time.time()
    rf_clf.fit(X_train, y_train)
    print(f"Training completed in {time.time() - train_start_time:.2f} seconds for {name}.")

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = rf_clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {accuracy}")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=object_names)

    # Print confusion matrix
    print(f"Confusion Matrix for {name}:")
    print(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=object_names, yticklabels=object_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for {name}')
    plt.show()

    end_time = time.time()
    print(f"Total processing time for {name}: {end_time - start_time:.2f} seconds.")

# Print the number of features extracted per object by each detector
print("\nNumber of features extracted per object by each detector:")
for obj in object_names:
    print(f"\nObject: {obj}")
    for detector_name, num_features in results[obj].items():
        print(f"{detector_name}: {num_features}")

# Visualization of the number of features extracted per object by each detector
objects = list(results.keys())
detector_names = list(detectors.keys())
num_features = [[results[obj][det] for det in detector_names] for obj in objects]

# Plotting
fig, ax = plt.subplots(figsize=(15, 10))

# Position of bars on x-axis
bar_width = 0.2
bar_positions = np.arange(len(objects))

# Colors for each detector
colors = ['blue', 'orange', 'green', 'red']

# Plot each detector's features as a separate bar chart
for i, detector_name in enumerate(detector_names):
    ax.bar(bar_positions + i * bar_width, [results[obj][detector_name] for obj in objects],
           color=colors[i], width=bar_width, label=detector_name)

# Labeling the x-axis
ax.set_xlabel('Objects')
ax.set_ylabel('Number of Features')
ax.set_title('Number of Features Extracted per Object by Each Detector')
ax.set_xticks(bar_positions + bar_width * (len(detector_names) - 1) / 2)
ax.set_xticklabels(objects, rotation=45, ha='right')
ax.legend()

plt.show()
