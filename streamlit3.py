import os
import numpy as np
import pandas as pd
import random
import warnings
from tabulate import tabulate
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
ds = tfp.distributions

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'pdf.fonttype': 'truetype'})
warnings.simplefilter("ignore")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

TRAIN_BUF = 10000
BATCH_SIZE = 512
DIMS = (28, 28, 1)
N_TRAIN_BATCHES = int(TRAIN_BUF/BATCH_SIZE)

# split dataset
train_images = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255.0

train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(TRAIN_BUF)
    .batch(BATCH_SIZE)
)


# Define your VAE model and load any necessary data
class VAE(tf.keras.Model):

    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.__dict__.update(kwargs)

        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

    def encode(self, x):
        mu, sigma = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def reconstruct(self, x):
        mu, _ = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return self.decode(mu)

    def decode(self, z):
        return self.dec(z)

    def compute_loss(self, x):
        q_z = self.encode(x)
        z = q_z.sample()
        x_recon = self.decode(z)
        p_z = ds.MultivariateNormalDiag(
          loc=[0.] * z.shape[-1], scale_diag=[1.] * z.shape[-1]
          )
        kl_div = ds.kl_divergence(q_z, p_z)
        latent_loss = tf.reduce_mean(tf.maximum(kl_div, 0))
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(x - x_recon), axis=0))

        return recon_loss, latent_loss

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)

    @tf.function
    def train(self, train_x):
        gradients = self.compute_gradients(train_x)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        
        
N_Z = 2
encoder = [
    tf.keras.layers.InputLayer(input_shape=DIMS),
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=N_Z*2),
]

decoder = [
    tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
    tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
    tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
    ),
]

#
# Model training
#
optimizer = tf.keras.optimizers.Adam(1e-3)
model = VAE(
    enc = encoder,
    dec = decoder,
    optimizer = optimizer,
)

n_epochs = 1
for epoch in range(n_epochs):
    print(f'Epoch {epoch}...')
    for batch, train_x in zip(range(N_TRAIN_BATCHES), train_dataset):
        model.train(train_x)

#Query Nearest Neighbors
#The embeddings obtained using VAE can be used for nearest neighbor search.


# Function to perform nearest neighbor search
def perform_nearest_neighbor_search(query_image_id, k):
    # Get embeddings for all training images
    embeddings, _ = tf.split(model.enc(train_images), num_or_size_splits=2, axis=1)
    
    # Get the embedding for the query image
    query_embedding = embeddings[query_image_id]
    
    # Calculate distances between the query embedding and all other embeddings
    distances = np.linalg.norm(query_embedding - embeddings, axis=1)
    
    # Find the indices of the k nearest neighbors
    nearest_neighbor_indices = np.argpartition(distances, k)[:k]
    
    return nearest_neighbor_indices


# Streamlit application
st.title("Nearest Neighbor Search with VAE")

# Input for query_image_id
query_image_id = st.slider("Select Query Image ID", 0, len(train_images) - 1, 0)

# Input for k (number of nearest neighbors)
k = st.slider("Number of Nearest Neighbors (k)", 1, 20, 6)

# Button to perform the search
if st.button("Perform Nearest Neighbor Search"):
    nearest_neighbor_indices = perform_nearest_neighbor_search(query_image_id, k)
    
    # Display the query image
    st.image(1 - train_images[query_image_id], caption="Query Image", use_column_width=True)
    
    # Display the nearest neighbor images
    st.subheader("Nearest Neighbor Images:")
    for index in nearest_neighbor_indices:
        st.image(1 - train_images[index], caption=f"Image ID: {index}", use_column_width=True)
