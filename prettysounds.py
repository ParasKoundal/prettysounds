
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from skimage.exposure import rescale_intensity
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray

from midiutil.MidiFile import MIDIFile


def as_gray(image_filter, image, *args, **kwargs):
    """
    Converts a color image into grayscale.
    """
    gray_image = rgb2gray(image)
    return image_filter(gray_image, *args, **kwargs)

@adapt_rgb(as_gray)
def sobel_gray(image):
    return filters.sobel(image)

def preprocess_image(image_path, mask_thresh=None):
    """

    Finds images in an image file and converts them into a numpy matrix.

    Parameters
    ==========

    image_path : path to image file. Images can be of filetype TODO
    mask_thresh : float between 0 and 1 that sets a treshold for intensity values to include.
    High values result in only stronger edges in the output matrix

    Returns:
    ========

    sobel_image : numpy matrix of floats where value represents the intensity of an edge
    in the same X*Y dimension of the original image file.

    """

    image = imread(image_path)
    sobel_image = sobel_gray(image)

    if mask_thresh:
        sobel_image = np.array([[i if i >=mask_thresh else 0 for i in img_row ] for img_row in sobel_image])

    return sobel_image

def plot_grayscale_img(image_mat):

    """
    tool to visualize the image matrix.
    """

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(1-image_mat,cmap=cm.gray)

    return fig

def matrix_to_midi(input_mat, first_note=0, tempo=120, duration=1, output_file=None):

    """
    Converts a numpy matrix into midi format and writes it to a file.

    Parameters
    ==========

    input_mat : numpy matrix where y is notes and x is time
    first_note : first note of y-axis in midi note number
    tempo : tempo of track to be written
    duration : Duration of each beat
    output_file : String of path and filename of midi file to write to. Should end in .mid

    """

    num_times = np.shape(input_mat)[1]
    track = 0
    channel = 0
    time = 0 # Time it starts

    MyMIDI = MIDIFile(1) # One track
    MyMIDI.addTempo(track, time, tempo)
    for times in range(num_times):
        freq=np.nonzero(input_mat[:,times])[0]+first_note
        volume=np.squeeze(np.matrix((input_mat[np.nonzero(input_mat[:,times]),times])))*100

        if len(freq)>0:
            for jj in range(len(freq)):
                MyMIDI.addNote(track, channel, freq[jj], times, duration, volume[0,jj].astype(np.int64))

    if not output_file:
        binfile = open("output.mid", 'wb')
    else:
        binfile = open(output_file, 'wb')
    MyMIDI.writeFile(binfile)
    binfile.close()

    return True

