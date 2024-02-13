#Weaving the Tapestry of Artists’ Similarity with a CNN Model
#From: https://medium.com/@zheng.qingq/weaving-the-tapestry-of-artists-similarity-with-a-cnn-model-3388ef75a2c5

#Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import sklearn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from keras.models import Model
import tensorflow as tf
#from keras.applications import NASNetLarge
#from tensorflow.keras.applications import NASNetLarge


from os import listdir
#Dataset: WikiArt
#Size of large dataset: 34GB(!)
#Source: https://www.kaggle.com/datasets/steubk/wikiart
path_img = "wikiart/"
list_img = [file for file in listdir(path_img)]
artist = [name.split('_')[0].replace('-','_') for name in list_img]

print(f"Number of images provided: {len(list_img)}")
print(f"Number of artists in dataset: {len(np.unique(artist))}")
print(f"Artists in dataset: \n{np.unique(artist).tolist()}")


# Display 25 randomly selected images from folder in a plot
import os
import random
import matplotlib.image as mpimg

# Resize images for easy display
image_size = (120, 120)

# Get a list of all image filenames in the folder
image_files = [f for f in os.listdir(path_img) if f.endswith('.jpg') or f.endswith('.png')]

num_images_to_display = 4
#num_images_to_display = 9
#num_images_to_display = 36 
random_images = random.sample(image_files, num_images_to_display)

num_rows = 2
num_cols = 2
#num_rows = 3
#num_cols = 3
#num_rows = 6 
#num_cols = 6
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
for i, ax in enumerate(axes.flatten()):
    img_path = os.path.join(path_img, random_images[i])
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.suptitle("Random Selection of Images in Sample", fontsize=12)
plt.show()


# Preprocessing images, resize to 224x224 and load into RBG format
images = []
ignored_indices = []
for i in range(len(list_img)):
    try:
        image = cv2.imread(path_img+list_img[i], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
    except:
        print(f"Ignoring file at index {i}: {list_img[i]}")
        ignored_indices.append(i)

print(f"Ignored indices: {ignored_indices}")


# Print an example image array
print(images[0])


#Exploratory data analysis
# A function to calculate the average colour of an image
def get_avg_colour(img):
   
    image = Image.open(path_img+img)
    pixels = np.array(image)
    flattened_pixels = pixels.reshape(-1, 3)

    # Count the frequency of each colour 
    from collections import Counter
    colour_counts = Counter(map(tuple, flattened_pixels))

    # Remove the top 5 most frequent colours
    top_colours = colour_counts.most_common(5)
    top_colour_set = set([colour[0] for colour in top_colours])
    new_pixels = [pixel for pixel in flattened_pixels if tuple(pixel) not in top_colour_set]
    num_pixels = len(new_pixels)  

    # Calculate the average RGB values
    total_rgb = np.sum(new_pixels, axis=0)
    avg_rgb = total_rgb // num_pixels

    # Return the average RGB values to a hex colour code
    avg_colour = '#{:02x}{:02x}{:02x}'.format(*avg_rgb)
    return avg_colour


# Create a dataframe with image labels
df_label=pd.DataFrame(list_img)
df_label.columns=['img']
df_label['artist']=artist
df_label.head()

# Get the average colour per paiting
avg_colours=[]
for img in list_img:
    colour=get_avg_colour(img)
    avg_colours.append(colour)
df_label['avg_colours']=avg_colours

df_label.head()


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

average_colour_per_artist = df_label.groupby('artist')['avg_colours'].apply(
    lambda x: rgb_to_hex(np.round(np.mean([hex_to_rgb(hex_code) for hex_code in x], axis=0)).astype(int))
)
average_colour_per_artist = average_colour_per_artist.to_frame().reset_index()
average_colour_per_artist.head()



# A plot with 4 rows a 5 columns
fig, ax = plt.subplots(figsize=(10, 6))
num_rows = 4
num_columns = len(average_colour_per_artist) // num_rows

# Calculate the width and height of each rectangle
rect_width = 1 / num_columns
rect_height = 1 / num_rows

# Iterate over the rows of the dataframe
for index, row in average_colour_per_artist.iterrows():
    artist = row['artist']
    colour = row['avg_colours']
    
    row_index = index // num_columns
    col_index = index % num_columns
    
    # Create rectangular shapes filled with ghe calculated colour
    rect_x = col_index * rect_width
    rect_y = row_index * rect_height
    rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height, facecolor=colour)
    ax.add_patch(rect)
    
    # Add seperating lines
    if col_index > 0:
        ax.plot([rect_x, rect_x], [0, 1], color='white', linewidth=1)
    
    if row_index > 0:
        ax.plot([0, 1], [rect_y, rect_y], color='white', linewidth=1)
    
    # Add the artist name as centered text in the center of the rectangle
    ax.text(rect_x + rect_width / 2, rect_y + rect_height / 2, artist,
            ha='center', va='center', color='white')

# Set the axis limits and labels
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_xlabel(None)

ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_ylabel(None)

# Remove the frame and spines
ax.set_frame_on(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.suptitle("Average Colour by Artist", fontsize=12)
plt.tight_layout()
plt.show()



# Extract features using NASNetLarge pretrained model
# The last layer is removed
wikiart_model = tf.keras.applications.NASNetLarge(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
model = Model(inputs=wikiart_model.inputs, outputs=wikiart_model.layers[-1].output)

features = []
for i in range(len(list_img)-len(ignored_indices)):
    if i%100 == 0 : print(i)
    image = np.array(images[i])
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = tf.keras.applications.nasnet.preprocess_input(image)
    feat = model.predict(image)
    features.append(feat)

features_cnn = np.concatenate(features, axis=0)
features_cnn_flat = features_cnn.reshape(features_cnn.shape[0], -1)
print(features_cnn_flat.shape)


df_features=pd.DataFrame(features_cnn_flat)
df=pd.concat([df_label,df_features],axis=1)
df.head()


df_grp = df.groupby('artist').mean()


pca=PCA(n_components=0.99)
pca.fit(df_grp)
df_pca = pca.transform(df_grp)
#df_pca = pd.DataFrame(df_pca, index=df_grp.index)
df_pca = pd.DataFrame(df_pca, index=df_grp.index)

print(f"Size of feature before PCA : {features_cnn_flat.shape[0]}")
print(f"Size of feature after PCA : {df_pca.shape[1]}")


print(f"Cumulative explained variance ration of first two PCs: {pca.explained_variance_ratio_.cumsum()[0:2]}")

df_pca_transformed = df_pca.T
df_pca_transformed.columns = range(df_pca_transformed.shape[1])
df_pca_transformed_back = df_pca_transformed.T
df_pca = df_pca_transformed_back

artist_names = df_grp.index.tolist()
#artist_names = df_pca['artist'].tolist()
pcs = df_pca.values
#pcs = df_pca.drop('artist', axis=1).values

def clustering_task(pcs):
    db_scan = DBSCAN(eps=1, min_samples=2)
    db_scan.fit(pcs)
    labels = db_scan.labels_
    core_samples_mask = np.zeros_like(db_scan.labels_, dtype=bool)
    core_samples_mask[db_scan.core_sample_indices_] = True

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            col = 'k'
        else:
            col = plt.cm.jet(label / n_clusters_)
        class_member_mask = (labels == label)
        
        ax.scatter(pcs[:0], pcs[:1], pcs[:2], c=col, s=50, edgecolor='k', label=label)
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title('DBSCAN Clustering of Artists')
    plt.show()
    
clustering_task(pcs.tolist())

# Plotting the first two principal components
plt.figure(figsize=(12, 10))
plt.scatter(pcs[:, 0], pcs[:, 1], s=30,  alpha=0.8, color='royalblue',edgecolor='black', linewidth=0.5)

# Add artist names near the data points
for i, name in enumerate(artist_names):
    plt.annotate(name, (pcs[i, 0]+0.8, pcs[i, 1]-0.7), fontsize=12, alpha=0.7)
    
total_variance = pca.explained_variance_ratio_.cumsum()[1]

plt.title(f"Artists Projected on First Two PCs, Explianed Variance: {round(total_variance * 100)}%")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.xticks([])
plt.yticks([])
plt.box(None)
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 6))
plt.scatter(pcs[:, 0], pcs[:, 1], s=30,  alpha=0.8, color='royalblue',edgecolor='black', linewidth=0.5)

# Add artist names near the data points
for i, name in enumerate(artist_names):
    plt.annotate(name, (pcs[i, 0]+0.8, pcs[i, 1]-0.7), fontsize=12, alpha=0.7)

# Zoom in
x_min, x_max = -30, -10  
y_min, y_max = -20, 20 
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.title(f"Zoom in Artists Projected on First Two PCs")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.xticks([])
plt.yticks([])
plt.box(None)
plt.tight_layout()
plt.show()

"""
Aufgabe 1a)

In this file, there are three plots generated. The first plot shows a random selection of images from the dataset. 
The second plot shows the average colour of each artist's work. The third plot shows the artists projected on the first two principal components.

The first task is to describe the method that is used to calculate the similarity between the artists.

To calculate the similarity between the artists, every pixel of each image is used. 
Those pixels have RGB formats, but are reshaped into values betweem -1 and 1. 
The image informations is thrown into the models predict method, which returns the prediction of features of the image.
The Model which is used is a Convolutional Neural Network (CNN) called NASNetLarge. NASNetLarge has been trained on more than a million images.
(This will sound very informal, but I can't find a better way to describe it. I'm sorry. I hope it's okay.) -> it is extremely cool, that we can just import a CNN,
and make use of it, without having to train it ourselves.

After saving those predictions/features, in a dataframe with fitting labels, the dataframe is grouped and then the average numbers are calculated. 

After the the magic happens. A PCA instance is created and the fit method is called with the grouped dataframe as parameter, which trains the model to the dataframe data.
PCA (Principal Component Analysis) is a dimension reduction technique, which is used to extract important features from a high dimensional dataset
and transforming them into a set of vairables, which do not correcalte (Principal Components).

In this case, 7 PCAs are reduced to 5. And the first two PCAs are used to plot the artists in a 2D space.
The first PC captures the direction in the feature space, where the data has the highest variance. 
The second PC has the second highest variance, ... and so on.

"""