#!/usr/bin/env python
"""
@author: phdenzel

Better color module for more beautiful plots
"""
import numpy as np
import random
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt

__all__ = ['color_variant', 'GLEAMcolors', 'GLEAMcmaps']


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
# Remove once replaced into different color groups
golden = '#feb125'  # rba(256, 177,  37)
purpleblue = '#7d7de1'  # rba(125, 125, 225)
turquoise = '#00d1a4'  # rba( 10, 210, 165)
gray = '#98989d'  # rba(152, 152, 157)
darkish = '#666769'  # rba(102, 103, 105)
dark = '#3d3e41'  # rba( 61,  62,  65)
darker = '#333437'  # rba( 51,  52,  55)
darkest = '#212225'  # rba( 33,  34,  37)
black = '#090F0F'  # rba(  9,  15,  15)
textcolor = '#dddee1'  # rba(221, 222, 225)
# colors = [red, purpleblue, green, orange, yellow, gray, turquoise,
#           pink, purple, golden, brown, blue, darker, textcolor]

# primary colors
red = '#FF6767'
pink = '#FF375F'  # rba(255,  55,  95)
orange = '#FF9F0A'  # rba(255, 159,  10)
yellow = '#FFD60A'  # rba(155, 214,  10)
purple = '#603DD0'  # rba( 96,  61, 208)
green = '#32D74B'  # rba( 50, 215,  75)
cyan = '#5BC1AE'
blue = '#6767FF'
brown = '#D88C4E'  # rba(172, 142, 104)
white = '#DDDEE1'  # rba(221, 222, 225)
grey = '#98989D'  # rba(152, 152, 157)

# light colors
cyan_light = '#A0DED2'

# dark colors
cyan_dark = '#24A38B'
blue_marguerite = '#756BB1'

# scheme colors
aquaria1, aquaria2, aquaria3, aquaria4 = '#00207F', '#A992FA', '#EA55B1', '#FEC763'
geometric1, geometric2, geometric3, geometric4 = '#08F7FE', '#09FBD3', '#FE53BB', '#F5D300'
neon1, neon2, neon3, neon4 = '#560A86', '#7122FA', '#F148FB', '#FFACFC'
psychedelic1, psychedelic2, psychedelic3, psychedelic4 = '#011FFD', '#FF2281', '#B76CFD', '#75D5FD'
vivid1, vivid2, vivid3, vivid4 = '#FDC7D7', '#FF9DE6', '#A5D8F3', '#E8E500'
abstract1, abstract2, abstract3, abstract4 = '#7B61F8', '#FF85EA', '#FDF200', '#00FECA'
phoenix1, phoenix2, phoenix3, phoenix4 = '#33135C', '#652EC7', '#DE38C8', '#FFC300'
cosmicnova1, cosmicnova2, cosmicnova3, cosmicnova4 = '#3B27BA', '#E847AE', '#13CA91', '#FF9472'
pinkpop1, pinkpop2, pinkpop3, pinkpop4 = '#35212A', '#3B55CE', '#FF61BE', '#FFDEF3'
agaveglitch1, agaveglitch2, agaveglitch3, agaveglitch4 = '#01535F', '#02B8A2', '#FDB232', '#FDD400'
coralglow1, coralglow2, coralglow3, coralglow4 = '#F85125', '#FF8B8B', '#FEA0FE', '#79FFFE'
luxuria1, luxuria2, luxuria3, luxuria4 = '#037A90', '#00C2BA', '#FF8FCF', '#CE96FB'
luminos1, luminos2, luminos3, luminos4 = '#9D72FF', '#FFB3FD', '#01FFFF', '#01FFC3'
stationary1, stationary2, stationary3, stationary4 = '#FE6B25', '#28CF75', '#EBF875', '#A0EDFF'
prism1, prism2, prism3, prism4 = '#C24CF6', '#FF1493', '#FC6E22', '#FFFF66'
retro1, retro2, retro3, retro4 = '#CE0000', '#FF5F01', '#FE1C80', '#FFE3F1'
acryliq1, acryliq2, acryliq3, acryliq4 = '#F21A1D', '#FF822E', '#03DDDC', '#FEF900'
hibokeh1, hibokeh2, hibokeh3, hibokeh4 = '#0310EA', '#FB33DB', '#7FFF00', '#FCF340'
flashy1, flashy2, flashy3, flashy4 = '#04005E', '#440BD4', '#FF2079', '#E92EFB'
cyber1, cyber2, cyber3, cyber4 = '#00218A', '#535EEB', '#BC75F9', '#BDBDFD'
zoas1, zoas2, zoas3, zoas4 = '#001437', '#7898FB', '#5CE5D5', '#B8FB3C'
vilux1, vilux2, vilux3, vilux4, vilux5, vilux6, vilux7 = (
    '#001437', '#85B2FF', '#17E7B6', '#D4FD87', '#FDEB87', '#FDD74C', '#FCAA43')
graphiq1, graphiq2, graphiq3, graphiq4 = '#48ADF1', '#C6BDEA', '#FDCBFC', '#8AF7E4'
vibes1, vibes2, vibes3, vibes4 = '#027A9F', '#12B296', '#FFAA01', '#E1EF7E'
purplerain1, purplerain2, purplerain3, purplerain4 = '#120052', '#8A2BE2', '#B537F2', '#3CB9FC'


# #############################################################################
# Palettes
# #############################################################################
# color lists
colors = [
    red, pink, orange, yellow, purple, green, cyan, blue, brown, white, grey,
    cyan_light,
    cyan_dark
]
light_colors = [
    cyan_light,
]
dark_colors = [
    cyan_dark,
]
misc_colors = [
    blue_marguerite,
]
aquaria_palette = [aquaria1, aquaria2, aquaria3, aquaria4]
geometric_palette = [geometric1, geometric2, geometric3, geometric4]
neon_palette = [neon1, neon2, neon3, neon4]
psychedelic_palette = [psychedelic1, psychedelic2, psychedelic3, psychedelic4]
vivid_palette = [vivid1, vivid2, vivid3, vivid4]
abstract_palette = [abstract1, abstract2, abstract3, abstract4]
phoenix_palette = [phoenix1, phoenix2, phoenix3, phoenix4]
cosmicnova_palette = [cosmicnova1, cosmicnova2, cosmicnova3, cosmicnova4]
pinkpop_palette = [pinkpop1, pinkpop2, pinkpop3, pinkpop4]
agaveglitch_palette = [agaveglitch1, agaveglitch2, agaveglitch3, agaveglitch4]
coralglow_palette = [coralglow1, coralglow2, coralglow3, coralglow4]
luxuria_palette = [luxuria1, luxuria2, luxuria3, luxuria4]
luminos_palette = [luminos1, luminos2, luminos3, luminos4]
stationary_palette = [stationary1, stationary2, stationary3, stationary4]
prism_palette = [prism1, prism2, prism3, prism4]
retro_palette = [retro1, retro2, retro3, retro4]
acryliq_palette = [acryliq1, acryliq2, acryliq3, acryliq4]
hibokeh_palette = [hibokeh1, hibokeh2, hibokeh3, hibokeh4]
flashy_palette = [flashy1, flashy2, flashy3, flashy4]
cyber_palette = [cyber1, cyber2, cyber3, cyber4]
zoas_palette = [zoas1, zoas2, zoas3, zoas4]
vilux_palette = [vilux1, vilux2, vilux3, vilux4, vilux5, vilux6, vilux7]
graphiq_palette = [graphiq1, graphiq2, graphiq3, graphiq4]
vibes_palette = [vibes1, vibes2, vibes3, vibes4]
purplerain_palette = [purplerain1, purplerain2, purplerain3, purplerain4]
# _palette = ['#', '#', '#', '#']


class GLEAMcolors:
    """
    An assortment of colors and palettes
    """
    red, pink, orange, yellow, purple, green, blue, brown, white, grey = (
        red, pink, orange, yellow, purple, green, blue, brown, white, grey)
    cyan_light = (cyan_light, )
    cyan_dark, = (cyan_dark, )
    blue_marguerite, = (blue_marguerite, )
    primary_colors = colors
    light_colors = light_colors
    dark_colors = dark_colors
    misc_colors = misc_colors
    colors = primary_colors + light_colors + dark_colors + misc_colors

    aquaria_palette = aquaria_palette
    aquaria1, aquaria2, aquaria3, aquaria4 = (aquaria1, aquaria2,
                                              aquaria3, aquaria4)
    geometric_palette = geometric_palette
    geometric1, geometric2, geometric3, geometric4 = (geometric1, geometric2,
                                                      geometric3, geometric4)
    neon_palette = neon_palette
    neon1, neon2, neon3, neon4 = (neon1, neon2,
                                  neon3, neon4)
    psychedelic_palette = psychedelic_palette
    psychedelic1, psychedelic2, psychedelic3, psychedelic4 = (psychedelic1, psychedelic2,
                                                              psychedelic3, psychedelic4)
    vivid_palette = vivid_palette
    vivid1, vivid2, vivid3, vivid4 = (vivid1, vivid2,
                                      vivid3, vivid4)
    abstract_palette = abstract_palette
    abstract1, abstract2, abstract3, abstract4 = (abstract1, abstract2,
                                                  abstract3, abstract4)
    phoenix_palette = phoenix_palette
    phoenix1, phoenix2, phoenix3, phoenix4 = (phoenix1, phoenix2,
                                              phoenix3, phoenix4)
    cosmicnova_palette = cosmicnova_palette
    cosmicnova1, cosmicnova2, cosmicnova3, cosmicnova4 = (cosmicnova1, cosmicnova2,
                                                          cosmicnova3, cosmicnova4)
    pinkpop_palette = pinkpop_palette
    pinkpop1, pinkpop2, pinkpop3, pinkpop4 = (pinkpop1, pinkpop2,
                                              pinkpop3, pinkpop4)
    agaveglitch_palette = agaveglitch_palette
    agaveglitch1, agaveglitch2, agaveglitch3, agaveglitch4 = (agaveglitch1, agaveglitch2,
                                                              agaveglitch3, agaveglitch4)
    coralglow_palette = coralglow_palette
    coralglow1, coralglow2, coralglow3, coralglow4 = (coralglow1, coralglow2,
                                                      coralglow3, coralglow4)
    luxuria_palette = luxuria_palette
    luxuria1, luxuria2, luxuria3, luxuria4 = (luxuria1, luxuria2,
                                              luxuria3, luxuria4)
    luminos_palette = luminos_palette
    luminos1, luminos2, luminos3, luminos4 = (luminos1, luminos2,
                                              luminos3, luminos4)
    stationary_palette = stationary_palette
    stationary1, stationary2, stationary3, stationary4 = (stationary1, stationary2,
                                                          stationary3, stationary4)
    prism_palette = prism_palette
    prism1, prism2, prism3, prism4 = (prism1, prism2,
                                      prism3, prism4)
    retro_palette = retro_palette
    retro1, retro2, retro3, retro4 = (retro1, retro2,
                                      retro3, retro4)
    acryliq_palette = acryliq_palette
    acryliq1, acryliq2, acryliq3, acryliq4 = (acryliq1, acryliq2,
                                              acryliq3, acryliq4)
    hibokeh_palette = hibokeh_palette
    hibokeh1, hibokeh2, hibokeh3, hibokeh4 = (hibokeh1, hibokeh2,
                                              hibokeh3, hibokeh4)
    flashy_palette = flashy_palette
    flashy1, flashy2, flashy3, flashy4 = (flashy1, flashy2,
                                          flashy3, flashy4)
    cyber_palette = cyber_palette
    cyber1, cyber2, cyber3, cyber4 = (cyber1, cyber2,
                                      cyber3, cyber4)
    zoas_palette = zoas_palette
    zoas1, zoas2, zoas3, zoas4 = (zoas1, zoas2,
                                  zoas3, zoas4)
    vilux_palette = vilux_palette
    vilux1, vilux2, vilux3, vilux4, vilux5, vilux6, vilux7 = (vilux1, vilux2, vilux3, vilux4,
                                                              vilux5, vilux6, vilux7)
    graphiq_palette = graphiq_palette
    graphiq1, graphiq2, graphiq3, graphiq4 = (graphiq1, graphiq2,
                                              graphiq3, graphiq4)
    vibes_palette = vibes_palette
    vibes1, vibes2, vibes3, vibes4 = (vibes1, vibes2,
                                      vibes3, vibes4)
    purplerain_palette = purplerain_palette
    purplerain1, purplerain2, purplerain3, purplerain4 = (purplerain1, purplerain2,
                                                          purplerain3, purplerain4)
    palettes = [aquaria_palette, geometric_palette, neon_palette, psychedelic_palette,
                vivid_palette, abstract_palette, phoenix_palette, cosmicnova_palette,
                pinkpop_palette, agaveglitch_palette, coralglow_palette, luxuria_palette,
                luminos_palette, stationary_palette, prism_palette, retro_palette,
                acryliq_palette, hibokeh_palette, flashy_palette, cyber_palette, zoas_palette,
                vilux_palette, graphiq_palette, vibes_palette, purplerain_palette]

    @classmethod
    def cmap_from_color(cls, color_str, secondary_color=None):
        """
        Create a colormap from a single color

        Args:
            color_str <str> - color string of the class color

        Kwargs:
            secondary_color <str> - color into which the color changes in the colormap

        Return:
            cmap <mpl.colors.LinearSegmentedColormap object> - reversed colormap
        """
        if secondary_color is None:
            secondary_color = color_variant(color_str, shift=125)
        if color_str in cls.__dict__:
            color = cls.__dict__[color_str]
        else:
            color = color_str
        cmap = LinearSegmentedColormap.from_list('GLEAM'+color_str, [secondary_color, color])
        return cmap


# #############################################################################
# Color maps
# #############################################################################
arcus = LinearSegmentedColormap.from_list('arcus', colors)
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
vilux = LinearSegmentedColormap.from_list('vilux', vilux_palette)
graphiq = LinearSegmentedColormap.from_list('graphiq', graphiq_palette)
vibes = LinearSegmentedColormap.from_list('vibes', vibes_palette)
purplerain = LinearSegmentedColormap.from_list('purplerain', purplerain_palette)
# = LinearSegmentedColormap.from_list('', _palette)


class GLEAMcmaps:
    """
    An assortment of linearly interpolated colormaps based on 4 colors each
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
    vilux = vilux
    graphiq = graphiq
    vibes = vibes
    purplerain = purplerain

    aslist = [aquaria, geometric, neon, psychedelic, vivid, abstract, phoenix, cosmicnova,
              pinkpop, agaveglitch, coralglow, luxuria, luminos, stationary, prism, retro,
              acryliq, hibokeh, flashy, cyber, zoas, graphiq, vibes, purplerain, vilux]
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
    def reverse(cls, cmap, set_bad=None, set_under=None, set_over=None):
        """
        Reverse the specified colormap

        Args:
           cmap <mpl.colors.LinearSegmentedColormap object> - colormap to be reversed

        Kwargs:
            set_bad <str> - set colormaps bad values to a different color
            set_under <str> - set colormaps under values to a different color
            set_over <str> - set colormaps over values to a different color

        Return:
            rcmap <mpl.colors.LinearSegmentedColormap object> - reversed colormap
        """
        reverse = []
        k = []

        for key in cmap._segmentdata:
            k.append(key)

            channel = cmap._segmentdata[key]
            data = []

            for t in channel:
                data.append((1-t[0], t[2], t[1]))
            reverse.append(sorted(data))

        LinearL = dict(zip(k, reverse))
        rcmap = LinearSegmentedColormap('{}_r'.format(cmap.name), LinearL)
        if set_bad is not None:
            rcmap.set_bad(set_bad)
        if set_under is not None:
            rcmap.set_over(set_under)
        if set_over is not None:
            rcmap.set_over(set_over)
        return rcmap

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
