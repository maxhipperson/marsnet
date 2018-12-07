import os
import hyperspy.api as hs
from skimage import io
import numpy as np
from functools import wraps
from time import time

def timeit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        print('\nRunning {}...\n'.format(f.__name__))
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('\nElapsed time for {}: {:f}s\n'.format(f.__name__, end - start))
        return result
    return wrapper

def read_hdr_file(filename):
    """
    Read information from ENVI header file to a dictionary.
    By default all keys are converted to lowercase.
    To stop this behaviour and keep the original case set 'keep_case = True'

    Adapted from:

    Module to for working with ENVI header files
    Author: Dan Clewley
    Creation Date: 07/08/2015
    find_hdr_file and read_hdr_file written by Ben Taylor
    """
    dict = {}
    # comments = ''
    inblock = False

    try:
        file = open(filename, 'r')
    except:
        raise IOError("Could not open hdr file {}. Reason: {} {}".format(filename, sys.exc_info()[1], sys.exc_info()[2]))

    # Read line, split it on equals, strip whitespace from resulting strings
    # and add key/value pair to output

    for line in file:

        # { indicates a block - check for these
        if not inblock:

            # Check for a comment
            # if ";" in line:
            #     comments += line

            # Split line at first equals sign
            if "=" in line:

                linesplit = line.split("=", 1)
                key = linesplit[0].strip()
                value = linesplit[1].strip()

                # key = key.replace(' ', '_')

                # Convert key to lower case
                key = key.lower()

                # If value starts with an open brace, it's the start of a block
                # - strip the brace off and read the rest of the block

                if value.startswith('{'):

                    inblock = True
                    value = value.replace('{', '')

                    # If value ends with a close brace it's the end
                    # of the block as well - strip the brace off
                    if value.endswith('}'):

                        inblock = False
                        value = value.replace('}', '')

                value = value.strip()
                dict[key] = value
                # print('Added {} to dict'.format(key))

        # If in block, read, strip and add to current dict[key]
        else:
            value = line.strip()

            if value.endswith('}'):
                inblock = False
                value = value.replace('}', '')
                value = value.strip()

            dict[key] = dict[key] + value

    file.close()

    # Convert numeric strings to float
    for key, value in dict.items():
        valuesplit = value.split(',')

        if len(valuesplit) == 1:
            try:
                value = float(value)
                # print('Converted {} to float'.format(key))
            except ValueError:
                pass

        elif len(valuesplit) > 1:
            value = [entry.strip() for entry in valuesplit]

            for i, entry in enumerate(value):
                try:
                    entry = float(entry)
                    value[i] = entry
                except ValueError:
                    pass

        dict[key] = value

    # dict['_comments'] = comments

    return dict


def read_data_cube(hdr_path, img_path):
    """
    Reads a .tif data cube and a header file in from the paths specified.

    :param hdr_path: Path of the envi header file.
    :param img_path: Path of the .tif data cube
    :return: The header as a dictionary and the image as a hyperspy 1D signal object
    """

    header = read_hdr_file(hdr_path)
    img = io.imread(img_path)

    print('\nLoaded {}:'.format(hdr_path))
    print('\nLoaded {}'.format(img_path))

    c, w, h = img.shape
    img = img.transpose(1, 2, 0)

    # remove ignored values
    img[img == header['data ignore value']] = 0

    (y, x, ch) = img.shape

    axes_x = {'name': 'x', 'size': x, 'units': 'px'}
    axes_y = {'name': 'y', 'size': y, 'units': 'px'}
    axes_ch = {'name': 'wavelength band', 'size': ch, 'units': 'index'}

    # convert image to signal object
    im = hs.signals.Signal1D(img, axes=[axes_y, axes_x, axes_ch])
    print('\n', im.axes_manager)

    return header, im

def crop_signal(im, header, lower, upper):
    """
    Crops the image in the signal dimension by referencing the header file.

    :param im:
    :param lower:
    :param upper:
    :return:
    """



    # return im