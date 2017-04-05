
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from skimage.exposure import rescale_intensity
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters
from scipy.misc import imread
import scipy.signal
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


def reshape_image(image_mat, x_samples=300, y_samples=150, verbose=True):
    """
    Reshapes the image matrix.

    """

    if verbose:
        print 'reshaping image to ' + str(x_samples) + 'x' + str(y_samples)

    if x_samples:
        image_mat = np.array([scipy.signal.resample(image_mat[i,:],x_samples) for i in xrange(image_mat.shape[0])])

    if y_samples:
        image_mat = np.array([scipy.signal.resample(image_mat[:,i],y_samples) for i in xrange(image_mat.shape[1])]).T

    image_mat[image_mat<0] = 0

    return image_mat


def preprocess_image(image_path, reshape_params=[500,125], mask_thresh=0.25):
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

    myimage = imread(image_path)

    sobel_image = sobel_gray(myimage)

    if reshape_params:
        sobel_image = reshape_image(sobel_image, x_samples=reshape_params[0], y_samples=reshape_params[1])

    # images > 150 don't fit in midi files
    if sobel_image.shape[1] > 150:
        sobel_image = reshape_image(sobel_image, y_samples=150, x_samples=None)

    # increase contrast to make intensity values larger
    sobel_image = sobel_image/np.max(sobel_image)

    # remove small values
    if mask_thresh:
        sobel_image[sobel_image<mask_thresh] = 0

    return sobel_image


def padmat(input_mat,left_pad=0,right_pad=0,up_pad=0,down_pad=0):
    """
    Helper function to pad zeros to image matrices to edges don't get cut off.
    """

    new_mat = np.zeros((input_mat.shape[0]+up_pad+down_pad,input_mat.shape[1]+left_pad+right_pad))
    new_mat[up_pad:new_mat.shape[0]-down_pad,left_pad:new_mat.shape[1]-right_pad]=input_mat

    return new_mat


def plot_grayscale_img(image_mat,title=None):
    """
    tool to visualize the image matrix.
    """

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(image_mat,cmap=cm.gray)
    if title:
        ax.set_title(title)
    fig.colorbar(im)

    return fig


def add_music(input_mat, scale_template,method='slice'):
    """
    Removes tones not falling on the given scale

    scale_and_firstnote should be a 1X7 array with the first element as the
    key. e.g. scale_template = np.asarray([0,2,4,5,7,9,11])
    """
    freq_size = np.shape(input_mat)[0]
    num_rep = np.floor(freq_size/12).astype(np.int64)
    scale = np.tile(scale_template, (1,num_rep))
    add_offset = np.tile(np.asarray(range(num_rep)),(len(scale_template),1))
    scale = np.squeeze(scale + 12 * add_offset.flatten('F'))

    if method == 'slice':
        remove_scale = np.setdiff1d(range(freq_size),scale)
        input_mat[remove_scale,:]=0

    elif method == 'round':
        for col in xrange(input_mat.shape[1]):
            for ind,val in enumerate(input_mat[:,col]):
                if ind not in scale and val !=0:
                    # find closest place in scale
                    nearest_neighbor = min(scale, key=lambda x:abs(x-ind))
                    # move this value there and replace bad index with 0
                    input_mat[nearest_neighbor,col] = val
                    input_mat[ind,col] = 0
    else:
        print 'method not recognized'
        return None

    return input_mat


def apply_chord_changes(input_mat,chord_array,scale_templates=None,chord_timing=None,
                        starting_ind=None,method='round'):
    """
    Applies chord changes to a matrix by calling add_music

    Parameters
    ==========

    input_mat : numpy matrix where y is notes and x is time
    chord_array : list of chords to iterate through by half step
    scale_templates : keys of music for chord changes. defaults to major scales
    chord_timing : time to iterate through each chord in the progression. defaults to 4 beats
    starting_ind: point in time to start iterating through chords. defaults to first note
    method : 'round'|'slice' method to remove notes not in chord

    Returns:
    ========

    chord_change_mat : image matrix with chord changes applied.

    """
    if scale_templates is None:
        maj_scale = np.asarray([0,2,4,5,7,9,11]);
        scale_templates = [maj_scale for i in chord_array]
    if chord_timing is None:
        chord_timing = [16 for i in chord_array]
    if starting_ind is None:
        starting_ind = np.where(sum(input_mat,0)>0)[0][0]

    # add a little more space so it sounds nice
    input_mat = padmat(input_mat,right_pad=max(chord_timing))

    chord_change_mat = np.zeros(input_mat.shape);
    x_ind = starting_ind;
    chord_ind=0
    inrange=True
    while inrange:
        this_chunk = x_ind+chord_timing[chord_ind]
        if this_chunk < input_mat.shape[1]:
            unprocessed_chunk = input_mat[:,x_ind:this_chunk] # damn flip for some reason
            chord_chunk = add_music(unprocessed_chunk,chord_array[chord_ind]+scale_templates[chord_ind],method=method)
            chord_change_mat[:,x_ind:this_chunk] = chord_chunk
            chord_ind +=1
            if chord_ind>len(chord_array)-1:
                chord_ind=0
            x_ind = x_ind+chord_timing[chord_ind]-1
        else:
            inrange=False
    return chord_change_mat


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

    # shit is upside down for no reason
    input_mat = np.flipud(input_mat)

    num_times = np.shape(input_mat)[1]
    track = 0
    channel = 0
    time = 0 # Time it starts

    MyMIDI = MIDIFile(1) # One track
    MyMIDI.addTempo(track, time, tempo)

    print 'writing midi file'
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



