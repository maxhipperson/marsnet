"""
Adapted from:

Module to for working with ENVI header files
Author: Dan Clewley
Creation Date: 07/08/2015
find_hdr_file and read_hdr_file written by Ben Taylor
"""

###########################################################
# This file has been created by ARSF Data Analysis Node and
# is licensed under the GPL v3 Licence. A copy of this
# licence is available to download with this file.
###########################################################

from __future__ import print_function
import sys
import numpy

ENVI_TO_NUMPY_DTYPE = {'1':  numpy.uint8,
                       '2':  numpy.int16,
                       '3':  numpy.int32,
                       '4':  numpy.float32,
                       '5':  numpy.float64,
                       '6':  numpy.complex64,
                       '9':  numpy.complex128,
                       '12': numpy.uint16,
                       '13': numpy.uint32,
                       '14': numpy.int64,
                       '15': numpy.uint64}

# def find_hdr_file(rawfilename):
#     """
#     Find ENVI header file associated with data file
#     """
#     if not os.path.isfile(rawfilename):
#         raise IOError("Could not find file " + rawfilename)
#
#     # Get the filename without path or extension
#     filename = os.path.basename(rawfilename)
#     filesplit = os.path.splitext(filename)
#     filebase = filesplit[0]
#     dirname = os.path.dirname(rawfilename)
#
#     # See if we can find the header file to use
#     if os.path.isfile(os.path.join(dirname, filebase + ".hdr")):
#         hdrfile = os.path.join(dirname, filebase + ".hdr")
#     elif os.path.isfile(os.path.join(dirname, filename + ".hdr")):
#         hdrfile = os.path.join(dirname, filename + ".hdr")
#     else:
#         hdrfile = None
#
#     return hdrfile

def read_hdr_file(filename):
    """
    Read information from ENVI header file to a dictionary.
    By default all keys are converted to lowercase.
    To stop this behaviour and keep the original case set 'keep_case = True'
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

# def write_envi_header(filename, header_dict):
#     """
#     Writes a dictionary to an ENVI header file
#     Comments can be added to the end of the file using the '_comments' key.
#     """
#
#     # Open header file for writing
#     try:
#         hdrfile = open(filename, "w")
#     except:
#         raise IOError("Could not open hdr file {}. ".format(filename))
#
#     hdrfile.write("ENVI\n")
#     for key in header_dict.keys():
#         # Check not comments key (will write separately)
#         if key != "_comments":
#             # If it contains commas likely a list so put in curly braces
#             if str(header_dict[key]).count(',') > 0:
#                 hdrfile.write("{} = {{{}}}\n".format(key, header_dict[key]))
#             else:
#                 # Write key at start of line
#                 hdrfile.write("{} = {}\n".format(key, header_dict[key]))
#
#     # Write out comments at the end
#     # Check they start with ';' and add one if they don't
#     for comment_line in header_dict['_comments'].split('\n'):
#         if re.search("^;", comment_line) is None:
#             comment_line = ";{}\n".format(comment_line)
#         else:
#             comment_line = "{}\n".format(comment_line)
#         # Check line contains a comment before writing out.
#         if comment_line.strip() != ";":
#             hdrfile.write(comment_line)
#     hdrfile.close()