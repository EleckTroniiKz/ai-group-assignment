#Weaving the Tapestry of Artists’ Similarity with a CNN Model
#From: https://medium.com/@zheng.qingq/weaving-the-tapestry-of-artists-similarity-with-a-cnn-model-3388ef75a2c5

#Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from keras.models import Model
import tensorflow as tf
from os import listdir
#Dataset: WikiArt
#Size of large dataset: 34GB(!)
#Source: https://www.kaggle.com/datasets/steubk/wikiart

# Load the data
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
random_images = random.sample(image_files, num_images_to_display)

num_rows = 2
num_cols = 2
# Plot 4 images from the datasamples randomly
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

# Get the average colour per painting
avg_colours=[]
for img in list_img:
    colour=get_avg_colour(img)
    avg_colours.append(colour)
df_label['avg_colours']=avg_colours


# Method to turn hex colour codes to RGB
def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

# Method to turn RGB to hex colour codes
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

# Calculate the average colour per artist
average_colour_per_artist = df_label.groupby('artist')['avg_colours'].apply(
    lambda x: rgb_to_hex(np.round(np.mean([hex_to_rgb(hex_code) for hex_code in x], axis=0)).astype(int))
)
average_colour_per_artist = average_colour_per_artist.to_frame().reset_index()

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
    
    # Create rectangular shapes filled with the average colour per artist
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

# extract features of the images
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

df_features=pd.DataFrame(features_cnn_flat)
df=pd.concat([df_label,df_features],axis=1)

df_grp = df.groupby('artist').mean()

# Initialize Primary Component Analysis
pca=PCA(n_components=0.99)
pca.fit(df_grp)
df_pca = pca.transform(df_grp)
df_pca = pd.DataFrame(df_pca, index=df_grp.index)

print(f"Size of feature before PCA : {features_cnn_flat.shape[0]}")
print(f"Size of feature after PCA : {df_pca.shape[1]}")


print(f"Cumulative explained variance ration of first two PCs: {pca.explained_variance_ratio_.cumsum()[0:2]}")

df_pca_transformed = df_pca.T
df_pca_transformed.columns = range(df_pca_transformed.shape[1])
df_pca_transformed_back = df_pca_transformed.T
df_pca = df_pca_transformed_back

artist_names = df_grp.index.tolist()
pcs = df_pca.values

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

At first the code is not even comparing the similarities. First it will plot 4 random images and then calculate the average color of each artist. 
That is not of importance for the similarity calculation, but it is a nice visualization of the data.

To measure the similarity between the artists, the code uses a pretrained model called NASNetLarge. NASNetLarge is a convolutional neural network, which is trained on the ImageNet dataset.
The code uses the NASNetLarge model to extract features from the images. The features are then used to calculate the similarity between the artists.
After extracting the features, the images are represented by a matrix of numbers with the dimensions of 197568, which provide how much of a certain feature is present in the image.

The artists are then grouped and then the average value of each feature is calculated for each artist.
After that, the dimension of the matrices is lowered. To lower the dimension from 197568, to a more manageable size, the PCA algorithm is used.
The PCA algorithm is used to reduce the dimension of the feature matrix, while still keeping the most important features.

With the first two Primary Components the artist are then plotted in a 2D space. The distance between the artists in the 2D space is then used to calculate the similarity between the artists.

"""

"""
Aufgabe 1b)
According to the task, we need to "Cluster the artists in senseful groups".

We are not fully sure, if this means, that the professor expects us to program a clustering algorithm, or if we should just use the PCA to cluster the artists.

But first, we will only cluster the currently given artists work by hand. 
We have provided Pablo Picasso, Claude Monet, Henri Rousseau and Paul Gauguin. 
The plot is very easily sererable into four clusters, which are the artists themselves.

If we look into the plot which is generated by the code, it is noticeable that the points are equidistant (such a cool word). So we would have 4 Clusters, which are the artists themselves, and the names of the cluster depend on the artist.
There are different ways to cluster the images. We could cluster based on the average colour of the images, or we could just cluster about the historical and cultural relations. 
But In this task, were we cluster by hand, we will use the art epoches to cluster the images.

First Pablo Picasso. The Clust of Pablo Picasso would have the name "Cubism". Cubbism ist a art epoch, which was strongly influenced by Pablo Picasso and Georges Braque.
Cubistic art often portraits objects from different perspectives, which are combined into one picture. The objects are often shown in a geometric form.

Second Claude Monet. The Cluster of Claude Monet would have the name "Impressionism". Impressionism is a art epoch, which was strongly influenced by Claude Monet and Pierre-Auguste Renoir.
Impressionistic art has the goal to capture the momentary impression of a scene. The paintings are often very colorful and the brushstrokes are very visible.

Third Henri Rousseau. The Cluster of Henri Rousseau would have the name "Primitive". Primitive Art is a art epoch, in which the art makes use of simplistic forms, strong colors and symbolic motives from non western or primitive cultures.
It is important to note, that primitive is not a label to "insult" the artists.

Fourth Paul Gauguin. The Cluster of Paul Gauguin would have the name "Post Impressionism". Post Impressionism is a art epoch, which was strongly influenced by Paul Gauguin and Vincent van Gogh.
Post Impressionistic art is a reaction to the Impressionism. The art is often more symbolic and the colors are often more expressive.


If we would need to cluster the artists via code, we would use the DBSCAN algorithm, which is a density based clustering algorithm.
The DBSCAN algorithm is used to find high density areas in the dataset, and can be used here, because it is not previously known, how many clusters there are.
We would use the DBSCAN class by SKLearn. Passed to it would be two parameters. The first parameter would be the maximum distance between two samples for one to be considered as in the neighborhood of the other. The second parameter would be the minimum number of samples in a neighborhood for a point to be considered as a core point.
Unfortunately, with the provided images, we are not able to test the clustering efficiently and meaningfully, due to a lack of data. 
But the for the images the PCs are extracted, which are the Primary Components. If we pass those into the dbscan, it can cluster the artists based on the distance between the artists in the 2D space (or depending on the amount of PCs even more).
"""

def dbscan_cluster(primary_components):
    dbscan = DBSCAN(eps=2, min_samples=1)
    clusters = dbscan.fit_predict(primary_components)
    return clusters


"""
    Aufgabe 1c)

    Now following is the task, in which we need to evaluate the significance of the similarity measurement of artists that is described in the code.

    The similarity measurement of the artists is significant, because it is based on the features of the images. The features are extracted from the images by a pretrained model, which is trained on the ImageNet dataset.
    So the model was trained on millions of images, and is able to extract the most important features from the images. The features are then used to calculate the similarity between the artists.

    It is fair to assume, that the similarity that is measured between the artists is fairly accurate. 

"""