##############################################################################
# FILE: cartoonify.py
# WRITER: Amir Rosengarten, amir.rosen15, 207942285
# EXERCISE: Intro2cs2 ex6 2021-2022
# DESCRIPTION: Practice multi-dimensional loops and lists, and exposure
# to basic image processing tools.
##############################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
import copy
import sys
from ex6_helper import *
from typing import Optional
import math


def separate_channels(image: ColoredImage) -> List[List[List[int]]]:
    """
    The function receives an image (three-dimensional list) whose
     dimensions rows * columns * channels.
    :param image: Three-dimensional list
    :return: The function returns a three-dimensional list, whose
     dimensions channels * rows * columns. A list of two-dimensional images
     that each represent a single color channel.
    """

    seperate_lst = []
    for channels in range(len(image[0][0])):
        seperate_lst.append([])
        for rows in range(len(image)):
            seperate_lst[channels].append([])
            for columns in range(len(image[0])):
                seperate_lst[channels][rows].append(image[rows][columns]
                                                    [channels])
    return seperate_lst


def combine_channels(channels: List[List[List[int]]]) -> ColoredImage:
    """
    The function receives a channels-length list of two-dimensional images
     consisting of individual color channels.
    :param channels: List of 2D images.
    :return: The function returns one color image whose dimensions
    rows * columns * channels.
    """

    combine_lst = []
    for rows in range(len(channels[0])):
        combine_lst.append([])
        for columns in range(len(channels[0][0])):
            combine_lst[rows].append([])
            for channels_1 in range(len(channels)):
                combine_lst[rows][columns].append(
                    channels[channels_1][rows][columns])
    return combine_lst


def sum_lst_and_round(d1_lst):
    """
    The function receives a list, and calculates according to a formula,
     the extraction of the values of the colored pixels into one value.
    :param d1_lst: A list.
    :return: The function returns the requested value according
     to a formula.
    """

    count = d1_lst[0] * 0.299 + d1_lst[1] * 0.587 + d1_lst[2] * 0.114
    round_count = round(count)
    return round_count


def RGB2grayscale(colored_image: ColoredImage) -> SingleChannelImage:
    """
    The function gets a color image.
    :param colored_image: Color image.
    :return: The function returns an image in black and white.
    """

    gray_lst = []
    for i in range(len(colored_image)):
        gray_lst.append([])
        for j in range(len(colored_image[i])):
            gray_lst[i].append(sum_lst_and_round(colored_image[i][j]))
    return gray_lst


def blur_kernel(size: int) -> Kernel:
    """
    The function gets a whole value.
    :param size: A whole value.
    :return: The function returns a list the size of size * size, that
     each cell contains the value: 1 / size ** 2.
    """

    main_lst = [[1 / size ** 2] * size] * size
    return main_lst


def out_of_boundaries(i, j, image):
    """
    The function checks whether the index of the cell is within the image
     limits or not.
    :param i: Index of rows.
    :param j: Index of columns.
    :param image: The image.
    :return: The function returns True or False about the test.
    """

    rows = len(image)
    cols = len(image[0])
    return not (i >= 0 and j >= 0 and i < rows and j < cols)


def cal_cur_value1(a, b, k, kernel_value, image):
    """
    The function calculates the sum of the neighbors' values
     (including the pixel itself), multiplied by the corresponding
      entry in the kernel.
    :param a: Index of rows.
    :param b: Index of columns.
    :param k: The whole value is divided by 2 by the length of the kernel.
    :param kernel_value: The value inside the kernel.
    :param image: The image.
    :return: The function returns the sum of a pixel in the new image when
     it is calculated by running the kernel on it.
    """

    cur_sum = 0
    for i in range(a - k, a + k + 1):
        for j in range(b - k, b + k + 1):
            if out_of_boundaries(i, j, image):
                val = image[a][b]
            else:
                val = image[i][j]
            cur_sum += val * kernel_value
    cur_sum = round(cur_sum)
    if cur_sum > 255:
        cur_sum = 255
    if cur_sum < 0:
        cur_sum = 0
    return cur_sum


def apply_kernel(image: SingleChannelImage,
                 kernel: Kernel) -> SingleChannelImage:
    """
    The function receives an image with a single color channel,
     and a kernel.
    :param image: The image.
    :param kernel: The kernel.
    :return: The function returns an image of the same size as the
     original image, with each pixel in the new image calculated by
      running the kernel on it.
    """

    k = len(kernel[0]) // 2
    kernel_value = kernel[0][0]
    new_image = copy.deepcopy(image)
    for i in range(len(image)):
        for j in range(len(image[0])):
            new_image[i][j] = cal_cur_value1(i, j, k, kernel_value, image)
    return new_image


def bilinear_interpolation(image: SingleChannelImage, y: float,
                           x: float) -> int:
    """
    The function receives an image with a single color channel, and the
     coordinates of a pixel from the target image as they are in the
      original image.
    :param image: The image.
    :param y: Index of rows.
    :param x: Index of columns.
    :return: The function returns the value of the same pixel according to
     the requested calculation.
    """

    value_a = image[math.floor(y)][math.floor(x)]
    value_b = image[math.ceil(y)][math.floor(x)]
    value_c = image[math.floor(y)][math.ceil(x)]
    value_d = image[math.ceil(y)][math.ceil(x)]
    d_y = y - math.floor(y)
    d_x = x - math.floor(x)
    pixel_value = value_a * (1 - d_x) * (1 - d_y) + \
                  value_b * d_y * (1 - d_x) + value_c * d_x * (1 - d_y) + \
                  value_d * d_x * d_y
    return round(pixel_value)


def cal_resize(image, i, j, new_height, new_width):
    """
    The function checks the edge cases of the image
     (the edges of the image). In addition, the function calculates the
      values of the new rows and columns, and calculates the pixel values
       relative to them.
    :param image: The image.
    :param i: Index of rows.
    :param j: Index of columns.
    :param new_height: The new height of the new image.
    :param new_width: The new width of the new image.
    :return: The new pixel value.
    """

    if i == 0 and j == 0:
        val = image[i][j]
        return val
    if i == 0 and j == new_width - 1:
        val = image[i][len(image[0]) - 1]
        return val
    if i == new_height - 1 and j == 0:
        val = image[len(image) - 1][j]
        return val
    if i == new_height - 1 and j == new_width - 1:
        val = image[len(image) - 1][len(image[0]) - 1]
        return val
    y = (i / new_height) * (len(image) - 1)
    x = (j / new_width) * (len(image[0]) - 1)
    # Call the function bilinear_interpolation.
    val = bilinear_interpolation(image, y, x)
    return val


def resize(image: SingleChannelImage, new_height: int,
           new_width: int) -> SingleChannelImage:
    """
    The function receives an image with a single color channel,
     and two integers.
    :param image: The image.
    :param new_height: The new height of the new image.
    :param new_width: The new width of the new image.
    :return: The function returns a new image in size
     new height * new width so the value of each pixel in the new image is
      calculated based on its relative position in the original image.
    """

    new_img = list()
    for i in range(new_height):
        new_img.append([])
        for j in range(new_width):
            val = cal_resize(image, i, j, new_height, new_width)
            new_img[i].append(val)
    return new_img


def scale_down_colored_image(image: ColoredImage, max_size: int) -> \
        Optional[ColoredImage]:
    """
    The function gets a color image, and a positive integer that
     represents the maximum number of pixels we want to allow for the
      image in each direction.
    :param image: The image.
    :param max_size: Maximum number of pixels
    :return: The function will check if the image meets this constraint.
     If so, the function will return None. If not, the function returns
      a new color image that is the smallest of the input image to the
       maximum size that is constrained while maintaining the original
        proportions of the image.
    """

    if len(image) <= max_size and len(image[0]) <= max_size:
        return None
    new_row = len(image)
    new_column = len(image[0])
    if new_row > new_column:
        y = max_size
        x = round((new_column * max_size) / new_row)
    else:  # new_column >= new_row:
        y = round((new_row * max_size) / new_column)
        x = max_size
    # Call the function separate_channels.
    channels = separate_channels(image)
    for channel in range(len(channels)):
        # Call the function resize.
        channels[channel] = resize(channels[channel], y, x)
    # Call the function combine_channels.
    return combine_channels(channels)


def reverse_matrix(matrix_lst):
    """
    The function gets a two-dimensional list of the matrix integers.
    :param matrix_lst: List of the matrix integers.
    :return: The function returns a two-dimensional list of the matrix
     integers, so that each internal list returns in reverse order.
    """

    reversed_mat = []
    for row in matrix_lst:
        reversed_mat.append(row[:])
    for i in range(len(reversed_mat)):
        reversed_mat[i].reverse()
    return reversed_mat


def rotate_90(image: Image, direction: str) -> Image:
    """
    The function gets an image and direction.
    :param image: The image.
    :param direction: The direction.
    :return: The function returns a similar image, rotated 90 degrees
    to the desired direction.
    """

    new_image = []
    if direction == 'L':
        # Change the image to the left.
        new_image = [[image[j][i] for j in range(len(image))] for i in
                     range(len(image[0]) - 1, -1, -1)]
    elif direction == 'R':
        # Change the image to the right.
        new_image = [[image[j][i] for j in range(len(image))]
                     for i in range(len(image[0]))]
        # Call the function reverse_matrix.
        new_image = reverse_matrix(new_image)
    return new_image


def cal_threshold(blur_image, r, i, j, c):
    """
    The function checks if the indexes of the rows and columns are outside
     the image, and it calculates the formula to threshold accordingly.
    :param blur_image: The blur image.
    :param r: block_size // 2
    :param i: Index of rows.
    :param j: Index of columns.
    :param c: A certain constant parameter.
    """

    cur_sum = 0
    count = 0
    for a in range(i - r, i + r + 1):
        for b in range(j - r, j + r + 1):
            # Call the function out_of_boundaries.
            if not out_of_boundaries(a, b, blur_image):
                cur_sum = cur_sum + blur_image[a][b]
                count += 1
            else:
                cur_sum += blur_image[i][j]
                count += 1
    return (cur_sum / count) - c


def get_edges(image: SingleChannelImage, blur_size: int, block_size: int,
              c: int) -> SingleChannelImage:
    """
    The function receives a black and white image and three numbers.
    :param image: The image.
    :param blur_size: The parameter of image blurring.
    :param block_size: Blocking parameter.
    :param c: A certain constant parameter.
    :return: The function returns a new image, with the same dimensions,
     consisting of only two values (black and white), with black pixels
      marking boundaries in the image.
    """

    new_image = list()
    r = block_size // 2
    # Call the function blur_kernel.
    kernel_lst = blur_kernel(blur_size)
    # Call the function apply_kernel.
    blur_image = apply_kernel(image, kernel_lst)
    for i in range(len(image)):
        new_image.append([])
        for j in range(len(image[0])):
            # Call the function cal_threshold.
            threshold = cal_threshold(blur_image, r, i, j, c)
            if threshold > blur_image[i][j]:
                new_image[i].append(0)
            else:
                new_image[i].append(255)
    return new_image


def quantize(image: SingleChannelImage, N: int) -> SingleChannelImage:
    """
    The function receives an image as a two-dimensional list,
     and a natural number.
    :param image: The image.
    :param N: The natural number.
    :return: The function returns an image with the same dimensions,
     in which the values of the pixels are calculated according to the
      requested formula.
    """

    new_image = list()
    for i in range(len(image)):
        new_image.append([])
        for j in range(len(image[0])):
            new_image[i].append(round
                                (math.floor(
                                    image[i][j] * N / 256) * 255 / (
                                         N - 1)))
    return new_image


def quantize_colored_image(image: ColoredImage, N: int) -> ColoredImage:
    """
    The function receives a color image (3D list), and a natural number.
    :param image: The image.
    :param N: The natural number.
    :return: The function returns a similar image after quantization to N
     to the power of channels shades.
    """
    new_img = list()
    # Call the function separate_channels.
    channels = separate_channels(image)
    for channel in channels:
        # Call the function quantize.
        new_img.append(quantize(channel, N))
    # Call the function combine_channels.
    return combine_channels(new_img)


def add_mask_2D(img1, img2, mask):
    """
    The function gets the 2 images and the two-dimensional list.
    :param img1: An image.
    :param img2: An image.
    :param mask: Two-dimensional list.
    :return: The function returns a new image in which each pixel is
    calculated according to the requested formula.
    """
    new_img = list()
    for i in range(len(img1)):
        new_img.append([])
        for j in range(len(img1[0])):
            new_img[i].append(round(
                img1[i][j] * mask[i][j] + img2[i][j] * (1 - mask[i][j])))
    return new_img


def check3D(image1):
    """
    The function checks if the image is a three-dimensional list
     (color image), or a two-dimensional list (in black and white).
    :param image1: An image.
    """

    if isinstance(image1[0][0], list):
        return True
    else:
        return False


def add_mask(image1: Image, image2: Image,
             mask: List[List[float]]) -> Image:
    """
    The function receives 2 images with identical dimensions,
    and a two-dimensional list whose dimensions correspond to the
    dimensions of a single channel in the images, and whose values move
    in the field [1, 0].
    :param image1: An image.
    :param image2: An image.
    :param mask: Two-dimensional list.
    :return: The function returns a new image in which each pixel is
    calculated according to the requested formula.
    """

    # Call the function check3D.
    if not check3D(image1):
        # Call the function add_mask_2D.
        return add_mask_2D(image1, image2, mask)
    new_img = list()
    # Call the function separate_channels.
    channels_img1 = separate_channels(image1)
    channels_img2 = separate_channels(image2)
    for i in range(len(channels_img1)):
        cur_channel1 = channels_img1[i]
        cur_channel2 = channels_img2[i]
        # Call the function add_mask_2D.
        val = add_mask_2D(cur_channel1, cur_channel2, mask)
        new_img.append(val)
    # Call the function combine_channels.
    new_img = combine_channels(new_img)
    return new_img


def norm_img(img):
    """
    The function receives an image, and returns an image in black
    and white.
    """
    new_img = []
    for i in range(len(img)):
        row = []
        for j in range(len(img[0])):
            row.append(img[i][j] / 255)
        new_img.append(row)
    return new_img


def cartoonify(image: ColoredImage, blur_size: int, th_block_size: int,
               th_c: int, quant_num_shades: int) -> ColoredImage:
    """
    The function that receives a color image and all the relevant
     parameters.
    :param image: The image.
    :param blur_size: Blur kernel size.
    :param th_block_size: The size of the environment used to determine
     the threshold of each pixel.
    :param th_c: The constant is deducted from the average value we
     calculated for the final threshold.
    :param quant_num_shades: The number of shades we will use in the
     quantization step.
    :return: The function returns a color image which is the illustrated
     version of the original image.
    """

    new_image_s = quantize_colored_image(image, quant_num_shades)
    new_image_bw = RGB2grayscale(image)
    new_image_bw = get_edges(new_image_bw, blur_size, th_block_size, th_c)
    # Call the function norm_img.
    mask = norm_img(new_image_bw)
    channels = separate_channels(new_image_s)
    new_image =[add_mask(c, new_image_bw, mask) for c in channels]
    return combine_channels(new_image)


def main():
    """
    The main function that makes calls to all the other functions.
    """

    try:
        # Get the arguments.
        image_source = sys.argv[1]
        cartoon_dest = sys.argv[2]
        max_im_size = sys.argv[3]
        blur_size = sys.argv[4]
        th_block_size = sys.argv[5]
        th_c = sys.argv[6]
        quant_num_shades = sys.argv[7]
    except:
        # If the input is incorrect, the function returns a message due
        # to an error.
        print('The number of parameters is incorrect')
        return
    # A snippet of code that runs the program and saves the result
    # to a file.
    image = load_image(image_source)
    scaled = scale_down_colored_image(image, int(max_im_size))
    if scaled is not None:
        image = scaled
    new_image = cartoonify(image, int(blur_size), int(th_block_size),
                           int(th_c), int(quant_num_shades))
    save_image(new_image, filename=cartoon_dest)


if __name__ == '__main__':
    main()
