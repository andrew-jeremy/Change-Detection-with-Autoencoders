There are 3 repositories:
1) Deep Autoencoder: 2x 4 Conv layer deep autoencoders, for encoding pair of images for change detection.

2) Variational Autoencoder: Uses two variational autoencoders to encode the image pair separately and to generate the change map

3) Single Batch Autoencoder: Encodes the image pair to compare using the same encoder and generates the change map from the same encoder. Faster since it is only using a single autoencoder and more accurately encodes both images in the same latent space for comparison.
