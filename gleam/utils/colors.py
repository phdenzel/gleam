#!/usr/bin/env python
"""
@author: phdenzel

Better color module for more beautiful plots
"""
import numpy as np
import random
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt


# #############################################################################
# Functions
# #############################################################################
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



# #############################################################################
# Colors
# #############################################################################

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


# #############################################################################
# Palettes
# #############################################################################
aquaria_palette     = ['#00207F', '#A992FA', '#EA55B1', '#FEC763']
geometric_palette   = ['#08F7FE', '#09FBD3', '#FE53BB', '#F5D300']
neon_palette        = ['#560A86', '#7122FA', '#F148FB', '#FFACFC']
psychedelic_palette = ['#011FFD', '#FF2281', '#B76CFD', '#75D5FD']
vivid_palette       = ['#FDC7D7', '#FF9DE6', '#A5D8F3', '#E8E500']
abstract_palette    = ['#7B61F8', '#FF85EA', '#FDF200', '#00FECA']
phoenix_palette     = ['#33135C', '#652EC7', '#DE38C8', '#FFC300']
cosmicnova_palette  = ['#3B27BA', '#E847AE', '#13CA91', '#FF9472']
pinkpop_palette     = ['#35212A', '#3B55CE', '#FF61BE', '#FFDEF3']
agaveglitch_palette = ['#01535F', '#02B8A2', '#FDB232', '#FDD400']
coralglow_palette   = ['#F85125', '#FF8B8B', '#FEA0FE', '#79FFFE']
luxuria_palette     = ['#037A90', '#00C2BA', '#FF8FCF', '#CE96FB']
luminos_palette     = ['#9D72FF', '#FFB3FD', '#01FFFF', '#01FFC3']
stationary_palette  = ['#FE6B25', '#28CF75', '#EBF875', '#A0EDFF']
prism_palette       = ['#C24CF6', '#FF1493', '#FC6E22', '#FFFF66']
retro_palette       = ['#CE0000', '#FF5F01', '#FE1C80', '#FFE3F1']
acryliq_palette     = ['#F21A1D', '#FF822E', '#03DDDC', '#FEF900']
hibokeh_palette     = ['#0310EA', '#FB33DB', '#7FFF00', '#FCF340']
flashy_palette      = ['#04005E', '#440BD4', '#FF2079', '#E92EFB']
cyber_palette       = ['#00218A', '#535EEB', '#BC75F9', '#BDBDFD']
zoas_palette        = ['#001437', '#7898FB', '#5CE5D5', '#B8FB3C']
graphiq_palette     = ['#48ADF1', '#C6BDEA', '#FDCBFC', '#8AF7E4']
vibes_palette       = ['#027A9F', '#12B296', '#FFAA01', '#E1EF7E']
purplerain_palette  = ['#120052', '#8A2BE2', '#B537F2', '#3CB9FC']
# _palette     = ['#', '#', '#', '#']


# #############################################################################
# Color maps
# #############################################################################
aquaria = LinearSegmentedColormap.from_list('aquaria', aquaria_palette)
geometric = LinearSegmentedColormap.from_list('geometric', geometric_palette)
neon = LinearSegmentedColormap.from_list('neon', neon_palette)
psychedelic = LinearSegmentedColormap.from_list('psychedelic', psychedelic_palette)
vivid = LinearSegmentedColormap.from_list('vivid', vivid_palette)
abstract = LinearSegmentedColormap.from_list('abstract', abstract_palette)
phoenix = LinearSegmentedColormap.from_list('phoenix', phoenix_palette)
cosmicnova = LinearSegmentedColormap.from_list('cosmicnova', cosmicnova_palette)
pinkpop = LinearSegmentedColormap.from_list('pinkpop', pinkpop_palette)
agaveglitch = LinearSegmentedColormap.from_list('agaveglitch', agaveglitch_palette)
coralglow = LinearSegmentedColormap.from_list('coralglow', coralglow_palette)
luxuria = LinearSegmentedColormap.from_list('luxuria', luxuria_palette)
luminos = LinearSegmentedColormap.from_list('luminos', luminos_palette)
stationary = LinearSegmentedColormap.from_list('stationary', stationary_palette)
prism = LinearSegmentedColormap.from_list('prism', prism_palette)
retro = LinearSegmentedColormap.from_list('retro', retro_palette)
acryliq = LinearSegmentedColormap.from_list('acryliq', acryliq_palette)
hibokeh = LinearSegmentedColormap.from_list('hibokeh', hibokeh_palette)
flashy = LinearSegmentedColormap.from_list('flashy', flashy_palette)
cyber = LinearSegmentedColormap.from_list('cyber', cyber_palette)
zoas = LinearSegmentedColormap.from_list('zoas', zoas_palette)
graphiq = LinearSegmentedColormap.from_list('graphiq', graphiq_palette)
vibes = LinearSegmentedColormap.from_list('vibes', vibes_palette)
purplerain = LinearSegmentedColormap.from_list('purplerain', purplerain_palette)
# = LinearSegmentedColormap.from_list('', _palette)


class GLEAMcmaps:
    """
    """
    aquaria = aquaria
    geometric = geometric
    neon = neon
    psychedelic = psychedelic
    vivid = vivid
    abstract = abstract
    phoenix = phoenix
    cosmicnova = cosmicnova
    pinkpop = pinkpop
    agaveglitch = agaveglitch
    coralglow = coralglow
    luxuria = luxuria
    luminos = luminos
    stationary = stationary
    prism = prism
    retro = retro
    acryliq = acryliq
    hibokeh = hibokeh
    flashy = flashy
    cyber = cyber
    zoas = zoas
    graphiq = graphiq
    vibes = vibes
    purplerain = purplerain

    aslist = [aquaria, geometric, neon, psychedelic, vivid, abstract, phoenix, cosmicnova,
              pinkpop, agaveglitch, coralglow, luxuria, luminos, stationary, prism, retro,
              acryliq, hibokeh, flashy, cyber, zoas, graphiq, vibes, purplerain]
    asarray = np.asarray(aslist)
    N = len(aslist)

    @classmethod
    def random(cls):
        """
        Choose a random color map

        Args/Kwargs:
            None

        Return:
            cmap <mpl.colors.LinearSegmentedColormap object> - random colormap from custom list
        """
        return random.choice(cls.aslist)

    @classmethod
    def gen(cls):
        """
        Generate colormaps

        Args/Kwargs:
            None

        Return:
            cmap <mpl.colors.LinearSegmentedColormap object> - colormap generated from custom list
        """
        for cmap in cls.aslist:
            yield cmap

    @classmethod
    def plot_gradients(cls, savefig=False):
        """
        Plot all color-map gradients

        Args:
            None

        Kwargs:
            savefig <bool> - save figure as palettes.png

        Return:
            None
        """
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        fig, axes = plt.subplots(nrows=GLEAMcmaps.N)
        fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
        for ax, cmap in zip(axes, cls.aslist):
            ax.imshow(gradient, aspect='auto', cmap=cmap)
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3]/2.
            fig.text(x_text, y_text, cmap.name, va='center', ha='right', fontsize=10)
        for ax in axes:
            ax.set_axis_off()
        if savefig:
            plt.savefig('palette.png')
