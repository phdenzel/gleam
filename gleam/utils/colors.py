#!/usr/bin/env python
"""
@author: phdenzel

Better color module for more beautiful plots
"""
###############################################################################
# Colors
###############################################################################

yellow = '#edde45'
gold = '#c18d1d'
orange = '#da9605'
brown = '#d88c4e'
winered = '#7c3658'
pink = '#fd4d83'
red = '#fe4365'
purple = '#9950cb'
purpleblue = '#603dd0'
lightblue = '#39b3e6'
blue = '#4182c4'
neongreen = '#0bf759'
green = '#91c442'
turquoise = '#00d1a4'
darkgreen = '#0d8e66'
greyishgreen = '#57776f'
textcolor = '#555555'
slategrey = '#708090'
gray = '#aab1b7'
gray2 = '#a9b0b6'
darkish = '#25232d'
darkish2 = '#31343f'


def color_variant(hex_color, shift=10):
    """
    Takes a color in hex code and produces a lighter or darker variant depending on the shift

    Args:
        hex_color <str> - formatted as '#' + rgb hex string of length 6

    Kwargs:
        shift <int> - decimal shift of the rgb hex string

    Return:
        variant <str> - formatted as '#' + rgb hex string of length 6
    """
    if len(hex_color) != 7:
        message = "Passed {} to color_variant(), needs to be in hex format."
        raise Exception(message.format(hex_color))
    rgb_hex = [hex_color[x:x+2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + shift for hex_value in rgb_hex]
    # limit to interval 0 and 255
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int]
    # hex() produces "0x88", we want the last two digits
    return "#" + "".join([hex(i)[2:] for i in new_rgb_int])
