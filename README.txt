There are 3 repositories:
1) Deep Autoencoder: 2x 4 Conv layer deep autoencoders, for encoding pair of images for change detection.

2) Variational Autoencoder: Uses two variational autoencoders to encode the image pair separately and to generate the change map

3) Single Batch Autoencoder: Encodes the image pair to compare using the same encoder and generates the change map from the same encoder. Faster since it is only using a single autoencoder and more accurately encodes both images in the same latent space for comparison.



To run any of the code, I recommend that you use a visual development environment such as spyder or pycharm to display the results graphically at the end of the run. I have included unit test data in both repositories for first run validation of your development environment. In all cases, look for a section of code given below to modify for runs on custom data:def parse_cmd_line():    p = argparse.ArgumentParser(description='change detector')    p.set_defaults(verbose=0, quiet=False)    p.add_argument('-d', '--dir', help='image directory', default='../small_pairs_tiles/')    p.add_argument('-f', '--filename', help='filename string of image pair', default='c')    p.add_argument('-e', '--epochs', help='number of epochs', default=400)    args = p.parse_args()    return args