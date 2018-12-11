#!/usr/bin/env python
"""
@author: phdenzel

Better color module for more beautiful plots
"""
###############################################################################
# Colors
###############################################################################

yellow     = '#ffd60a'  # rba(155, 214,  10)
golden     = '#feb125'  # rba(256, 177,  37)
orange     = '#ff9f0a'  # rba(255, 159,  10)
brown      = '#d88c4e'  # rba(172, 142, 104)
pink       = '#ff375f'  # rba(255,  55,  95)
red        = '#ff453a'  # rba(255,  69,  58)
purple     = '#603dd0'  # rba( 96,  61, 208)
purpleblue = '#7d7de1'  # rba(125, 125, 225)
blue       = '#6b89ff'  # rba(107, 137, 255)
turquoise  = '#00d1a4'  # rba( 10, 210, 165)
green      = '#32d74b'  # rba( 50, 215,  75)

gray       = '#98989d'  # rba(152, 152, 157)
darkish    = '#666769'  # rba(102, 103, 105)
dark       = '#3d3e41'  # rba( 61,  62,  65)
darker     = '#333437'  # rba( 51,  52,  55)
darkest    = '#212225'  # rba( 33,  34,  37)
black      = '#090F0F'  # rba(  9,  15,  15)
textcolor  = '#dddee1'  # rba(221, 222, 225)

colors = [red, purpleblue, green, orange, yellow, gray, turquoise,
          pink, purple, golden, brown, blue, darker, textcolor]


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
