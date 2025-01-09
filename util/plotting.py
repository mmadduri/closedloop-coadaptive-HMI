import numpy as np
import matplotlib.pyplot as plt
import warnings
import unittest
from scipy.stats import wilcoxon
import matplotlib.ticker as ticker




# convolving by a bunch of 1's -- a moving average box or low-pass filter
# see: https://danielmuellerkomorowska.com/2020/06/02/smoothing-data-by-rolling-average-with-numpy/
def smooth(data, kernal_size = 10, mode='valid'):
    '''
    desc: smooths out a function by convolving the function with ones (a moving average)
    
    inputs:
    data = samples (Y,)
    kernal_size = number of points to convolve over (M, )
    mode = 'valid', 'full', 'same' 
    
    see: https://numpy.org/doc/stable/reference/generated/numpy.convolve.html for modes and sizes
    '''
    kernal = np.ones(kernal_size)/kernal_size
    data_smooth = np.convolve(data, kernal, mode=mode)
    return data_smooth


def calcAndPlotWilcoxonBoxplot(data1, data2, data_size, xlabel1, xlabel2, color1='blue', color2='green', fig=plt, ax=plt):
    bplot = ax.boxplot([np.ndarray.flatten(data1), np.ndarray.flatten(data2)], 
                    showfliers=False,patch_artist=True,medianprops=dict(color='k'))

    for patch, color in zip(bplot['boxes'], [color1, color2]):
        patch.set_facecolor(color)
        
    assert(len(data1) == data_size)
    assert(len(data2) == data_size)
    
    ax.set_xticks([1, 2],[xlabel1, xlabel2])

    w = wilcoxon(np.ndarray.flatten(data1), np.ndarray.flatten(data2))
    
    return w

def remove_and_set_axes(axs, tick_size = 10, top=False, right=False, bottom=False, left=False):
    
    '''
    desc: removes the box around the axs and the ticks 
    
    inputs:
    axs = the axs of the figure
    '''

    axs.spines["top"].set_visible(top)
    axs.spines["right"].set_visible(right)
    axs.spines["bottom"].set_visible(bottom)
    axs.spines["left"].set_visible(left)

    axs.tick_params(axis='x', labelsize=tick_size)
    axs.tick_params(axis='y', labelsize=tick_size)
    
def figure_set_up(axs, tick_size = 10, 
                  top=False, right=False, bottom=False, left=False, 
                  x_major = 1, x_minor = 1, y_major = 1, y_minor = 1):
    '''
    desc: removes the box around the axs and the ticks 
    
    inputs:
    axs = the axs of the figure
    '''

    # set spines to be visible 
    axs.spines["top"].set_visible(top)
    axs.spines["right"].set_visible(right)
    axs.spines["bottom"].set_visible(bottom)
    axs.spines["left"].set_visible(left)

    # tick parameters
    axs.tick_params(axis='both', which='major', labelsize = tick_size)
    axs.tick_params(axis='both', which='minor', labelsize = tick_size)

    # set major and minor axes
    axs.xaxis.set_major_locator(ticker.MultipleLocator(base=x_major))
    axs.xaxis.set_minor_locator(ticker.MultipleLocator(base=x_minor))
    axs.yaxis.set_major_locator(ticker.MultipleLocator(base=y_major))
    axs.yaxis.set_minor_locator(ticker.MultipleLocator(base=y_minor))
    

def plot_time_domain(t, data, axis = (0, 1, 2), color = 'k', lw = 3, alpha = 0.2, ax = None, ls='-',
                      ms = 1, label='', edgecolor = 'None', remove_axes = True):
    
    '''
    create time series plot showing median and interquartile 
    smooth time-domain data if chosen
    
    data -- (N, M....) x s -- N trials with s samples
    '''
    
    if ax is None:
        fig,ax = plt.subplots(1,1,sharex=True, figsize=(12, 6))
    
    if remove_axes:
        remove_and_set_axes(ax)
    
    d25, d50, d75 = np.percentile(data, [25, 50, 75], axis = axis)

    ax.fill_between(t, d25, d75, alpha=alpha, color=color, edgecolor = edgecolor)
    ax.plot(t, d50, ls, color=color, linewidth=lw, label=label, ms = ms)


def plot_smooth_time_domain(time_x, data, data_len, axis = (0, 1, 2), kernal_size = 10,
                     color = 'k', lw = 3, alpha = 0.2, ax = None, ls='-', label='', 
                     edgecolor = 'None', remove_axes = True):
    
    '''
    create time series plot showing median and interquartile 
    smooth time-domain data if chosen
    
    data -- (N, M....) x s -- N trials with s samples
    '''
    
    if ax is None:
        fig,ax = plt.subplots(1,1,sharex=True, figsize=(fw, fh))
    
    if remove_axes:
        remove_and_set_axes(ax)
    
    d25, d50, d75 = np.percentile(data, [25, 50, 75], axis = axis)
    assert(data.shape[axis] == data_len)
    
    # smooth time-series data    
    d25 = smooth(d25, kernal_size, mode='valid')
    d50 = smooth(d50, kernal_size, mode='valid')
    d75 = smooth(d75, kernal_size, mode='valid') # shape = data.shape[-1] - ks + 1

    ax.fill_between(time_x[kernal_size-1:], d25, d75, alpha=alpha, color=color, edgecolor = edgecolor)
    ax.plot(time_x[kernal_size-1:], d50, ls, color=color, linewidth=lw, label=label)


def plot_time_domain_ref_curs(time, r, p, 
                              cr, cp, alpha = 1,
                              r_mkr = '--', r_lw = 3, rlabel = r'$\tau$', 
                              c_mkr = '-', c_lw = 3, clabel = '$y$', FrameBool = False, fig = None, ax = None):
    if ax is None:
        fig,ax = plt.subplots(1,1,sharex=True, figsize=(7,3))
    
    ax.plot(time, r, r_mkr, color = cr, alpha = alpha, linewidth = r_lw, label = rlabel);
    ax.plot(time, p, c_mkr, color = cp, alpha = alpha, linewidth = c_lw, label = clabel);
    
    ax.set_frame_on(FrameBool)
    ax.set_yticks([])

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


'''
use this function with caution -- it's a bit delicate
example usage:


fig, axs = plt.subplots(1, 1)
data1 = np.ndarray.flatten(td_error_first_med)
data2 = np.ndarray.flatten(td_error_last_med)
data_groups = [data1, data2]
data_labels = ['Early', 'Late']
data_pos = [0, 0.4]
plot_significance(pvalue = w.pvalue, data1=data1, data2 = data2, data_pos = data_pos, fig=fig, ax=axs, fontsize=40, lw=4)
'''
def plot_significance(pvalue, data1, data2, data_pos,
                      y_bar = 1, y_asterix = 1.5, 
                      fontsize = 40, lw = 1, color = 'black', 
                      fig = None, ax = None):
    if ax is None:
        fig,ax = plt.subplots(1,1, sharex=True, figsize=(7,3))
    
    # Check if the p-value is below the significance level
    if pvalue <= 0.001: #10^-3
        asterix =   "**"
    elif pvalue <= 0.05:
        asterix = "*"
    else:
        asterix = "ns"

    # Add significance asterisk and line between boxplots
    ax.text((max(data_pos) + min(data_pos))/2, max(max(data1), max(data2)) + y_asterix, asterix, ha='center', va='center', fontsize=fontsize, color = color)
    
    # significance line
    ax.plot(data_pos, [max(max(data1), max(data2)) + y_bar, max(max(data1), max(data2)) + y_bar], lw=lw, c=color)


'''
use this function with caution -- it's a bit delicate
example usage:


fig, axs = plt.subplots(1, 1)
data1 = np.ndarray.flatten(td_error_first_med)
data2 = np.ndarray.flatten(td_error_last_med)
data_groups = [data1, data2]
data_labels = ['Early', 'Late']
data_pos = [0, 0.4]
plot_significance(pvalue = w.pvalue, data1=data1, data2 = data2, data_pos = data_pos, fig=fig, ax=axs, fontsize=40, lw=4)
'''
def plot_significance_only_asterix(pvalue, data1, data2, data_pos,
                      y_bar = 1, y_asterix = 1.5, 
                      fontsize = 40, lw = 1, color = 'black', 
                      fig = None, ax = None):
    if ax is None:
        fig,ax = plt.subplots(1,1, sharex=True, figsize=(7,3))
    
    # Check if the p-value is below the significance level
    if pvalue <= 0.05:
        asterix =   "*"
        # Add significance asterisk and line between boxplots
        ax.text((max(data_pos) + min(data_pos))/2, max(max(data1), max(data2)) + y_asterix, asterix, ha='center', va='center', fontsize=fontsize, color = color)
    
        # significance line
        ax.plot(data_pos, [max(max(data1), max(data2)) + y_bar, max(max(data1), max(data2)) + y_bar], lw=lw, c=color)
    else:
        asterix = "ns"
        # don't print significance

### set up subplots
        
def identify_axes_center(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    # kw = dict(ha="right", va="top", fontsize=fontsize, color="darkgrey")

    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)


def subplots_mosaic(fig, mosaic, id_axes = True, id_axes_fontsize = 20):
    """
    Function to set up the subplots according to a specified pattern (called "mosaic")

    fig: the figure to add the plots to
    mosaic: the pattern. Needs to be a string 
    
    returns a figure and a way to access the axes in a dictionary 
    fig: figure with subplots
    ax_dict: axes to plot figures on

    example usage:
        
    fig_emg = plt.figure(figsize = (15, 8)) # set the total figure size
    fig_emg, ax_dict = subplots_mosaic(fig_emg, mosaic)
    plt.tight_layout()
    """
    ax_dict = fig.subplot_mosaic(mosaic)

    if id_axes:
        identify_axes_center(ax_dict, id_axes_fontsize)

    return fig, ax_dict

def plot_agent_nullcline(fig, ax, dpdx, dpdy, meshX, meshY, alpha_x, alpha_y, penalty_x, penalty_y, level_x, level_y, 
                         agent_x = 'User', agent_y = 'Machine', color_x = 'blue', color_y = 'black',
                         stream_color = 'lightgray', stream_lw = 0.5, stream_density = 0.5, stream_arrowsize = 0.4):

    # Calculate nullclines
    nullclines_x = alpha_x*-dpdx(meshX, meshY, penalty_x, penalty_y)
    nullclines_y = alpha_y*-dpdy(meshX, meshY, penalty_x, penalty_y)

    # Create a stream plot
    if fig == None:
        fig, ax = plt.subplots(figsize=(1.3, 1.3))

    ax.streamplot(meshX, meshY, 
              -alpha_x*dpdx(meshX, meshY, penalty_x, penalty_y), 
              -alpha_y*dpdy(meshX, meshY, penalty_x, penalty_y), 
              color=stream_color, linewidth=stream_lw,  density=stream_density, arrowsize=stream_arrowsize)

    # Plot nullclines
    ax.contour(meshX, meshY, nullclines_x, levels=[level_x], colors=color_x)
    ax.contour(meshX, meshY, nullclines_y, levels=[level_y], colors=color_y)


    # Set labels and title
    ax.set_xlabel(agent_x)
    ax.xaxis.label.set_color(color_x)
    ax.set_ylabel(agent_y)
    ax.yaxis.label.set_color(color_y)