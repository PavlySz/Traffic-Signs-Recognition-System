import numpy as np

# in order to display the full 1D array without truncations
np.set_printoptions(threshold=np.inf)

'''
# read file content
def read_file(path_file):
    with open(path_file, 'r') as my_file:
        content = my_file.read()
    
    return content
'''

# convert the array into another array, but concatinating every two characters with each other (1 byte = 2 hex)
def concat_two_elements(big_fucking_string):
    si = iter(big_fucking_string)
    si_conc = map(str.__add__, si, si)
    si_list = list(si_conc)         # 1-D array of characters of length 960*240
    return si_list

# convert each element in the numpy array from hex to decimal
def convert_hex_arr_to_dec(hex_array):
    list_decimal = []
    for i in hex_array:
        list_decimal.append(int(i, 16))
    return list_decimal

# create three arrays from the 1D array -- red, green and blue
def create_3_1d_channels(dec_array):
    # red
    red = []
    for i in range(0, len(dec_array), 3):
        red.append(dec_array[i])

    # green
    green = []
    for i in range(1, len(dec_array), 3):
        green.append(dec_array[i])

    # blue
    blue = []
    for i in range(2, len(dec_array), 3):
        blue.append(dec_array[i])
    
    return red, green, blue

# reshaping an array
def reshape_array(array, width, height):
    array_np = np.array(array, dtype=np.uint8)
    array_reshaped = array_np.reshape(width, height)
    return array_reshaped


# converting the channels into one RGB image
def create_rgb_img(red_channel, green_channel, blue_channel):
    img = np.dstack([red_channel, green_channel, blue_channel])
    return img

# main function
def process_frame(frame):
    si_list = concat_two_elements(str(frame))

    si_list_decimal = convert_hex_arr_to_dec(si_list)

    red, green, blue = create_3_1d_channels(si_list_decimal)

    # reshape each 1d array of a channel to a 2d numpy array of width 240 and height 320
    red_reshaped = reshape_array(red, 240, 320)
    green_reshaped = reshape_array(green, 240, 320)
    blue_reshaped = reshape_array(blue, 240, 320)

    image = create_rgb_img(red_reshaped, green_reshaped, blue_reshaped)

    return image