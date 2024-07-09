"""
Author: Yiqiao Zheng
Email: yiqiao.zheng@wustl.edu
"""

import os
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import mpra_CRE_utils


# make predicted occupancy histogram
def make_predictedOccp_histo(occp_df, motif_name, histocolor="grey", alpha=.6, density=True, bins=None, 
                                            xticks=None, yticks=None, figax=None):
    """
    Wrapper script for making a predicted occupancy histogram


    Parameters
    ----------
    xticks : list-like
        If specified, set the x axis ticks to these values.
    yticks: list-like
        If specified, set the y axis ticks to these values.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.

    Returns
    -------
    fig : Figure handle
    ax : Axes handle
    """

    # prepare data to plot
    data=occp_df.copy()
    if motif_name not in data.columns:
        raise ValueError(f"{motif_name} not in input occupancy dataframe.")
    else:
        data = data[motif_name]

    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    # density plot
    if density:
        ax = sns.kdeplot(data, color=histocolor, shade=True, ax=ax)
        ax.set_ylabel("Density")
    # count histogram
    else:
        if bins is None:
            bins = 100
        ax = sns.histplot(data, color=histocolor, alpha=alpha, ax=ax, bins=bins)
        ax.set_ylabel("CRE count")

    ax.set_xlabel("Predicted occupancy (a.u.)")

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    # add plot title
    ax.set_title(f"Predicted Occupancy for {motif_name}")
                
    # force equal aspect
    ax.set_aspect("auto")

    return fig, ax


# motif occurrence barchart - overlapped (make another grouped barchart function)
def make_motifOccurence_bar(motifCount_df, motif_name, histocolor="grey", alpha=.6, density=True, bins=None,
                                            xticks=None, yticks=None, figax=None):
    """
    Wrapper script for making a predicted occupancy histogram


    Parameters
    ----------
    xticks : list-like
        If specified, set the x axis ticks to these values.
    yticks: list-like
        If specified, set the y axis ticks to these values.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.

    Returns
    -------
    fig : Figure handle
    ax : Axes handle
    """

    # prepare data to plot
    data=motifCount_df.copy()

    if motif_name not in data.columns:
        raise ValueError(f"{motif_name} not in input occurrence dataframe.")
    else:
        # theorectically we are not instersted in CREs that contain bizarre nucleotide compositions
        data_count = data[motif_name].value_counts().to_frame().sort_index()
        data_count = data_count[data_count.index<=5].reset_index(drop=False)

    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    # density plot
    if density:
        ax = sns.kdeplot(data[motif_name], color=histocolor, shade=True, ax=ax)
        ax.set_ylabel("Density")
    # count histogram
    else:
        ax = sns.barplot(data=data_count, x="index", y=motif_name, color=histocolor, alpha=alpha, ax=ax)
        ax.set_ylabel("CRE count")

    ax.set_xlabel("Motif count")

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    # add plot title
    ax.set_title(f"Motif occurrence for {motif_name}")
                
    # force equal aspect
    ax.set_aspect("auto")

    return fig, ax
    
    
# generate GC content distribution histogram for a series of fasta sequences
def make_GC_histo(fasta_ser, histocolor="grey", alpha=.6, bins=10, density=True, 
                                            xticks=None, yticks=None, figax=None):
    """
    Wrapper script for making a scatter plot using GC percentages


    Parameters
    ----------
    xticks : list-like
        If specified, set the x axis ticks to these values.
    yticks: list-like
        If specified, set the y axis ticks to these values.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.

    Returns
    -------
    fig : Figure handle
    ax : Axes handle
    """

    # calculate GC content for each fasta sequence
    fasta_ser=fasta_ser.copy()
    fasta_GC_ser=mpra_CRE_utils.calculate_fasta_GC(fasta_ser)

    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    # density plot
    if density:
        ax = sns.kdeplot(fasta_GC_ser.GCperc, color=histocolor, shade=True, ax=ax)
        ax.set_ylabel("Density")
    # count histogram
    else:
        ax = sns.histplot(fasta_GC_ser.GCperc, color=histocolor, alpha=alpha, ax=ax, bins=bins)
        ax.set_ylabel("CRE count")

    ax.set_xlabel("GC percentage (%)")

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    # set y axis scales
    ax.set_xlim(0, 1)

    # add plot title
    ax.set_title("GC Content Distribution")
                
    # force equal aspect
    ax.set_aspect("auto")

    return fasta_GC_ser, fig, ax


# generate position GC content scatter plot for a series of fasta sequences
def make_GC_scatter(fasta_ser, markercolor="black", marker='.', mrsize=8, intprt_line=True, linecolor="grey", 
                                            xticks=None, yticks=None, figax=None):
    """
    Wrapper script for making a scatter plot using GC percentages


    Parameters
    ----------
    xticks : list-like
        If specified, set the x axis ticks to these values.
    yticks: list-like
        If specified, set the y axis ticks to these values.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.

    Returns
    -------
    fig : Figure handle
    ax : Axes handle
    """

    # calculate position GC content
    fasta_ser = fasta_ser.copy()
    position_GC_ser = mpra_CRE_utils.calculate_position_GC(fasta_ser)

    x=position_GC_ser.index.copy()
    y=position_GC_ser.copy()

    # convert fraction to percentage
    if y[0] < 1:
        y = y*100

    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    if intprt_line:
        # 1-D interpolation 
        #interpolated_f = scipy.interpolate.interp1d(x, y, kind='cubic')
        #ax.plot(interpolated_f(x), '-', color="orange", linewidth=1)

        #tck = scipy.interpolate.splrep(x, y, s=4)
        #xnew = np.arange(0, len(x), len(x)/100)
        #ynew = scipy.interpolate.splev(xnew, tck, der=0)
        #ax.plot(xnew, ynew, '-', color="green", linewidth=1)

        xnew1 = np.linspace(x.min(), x.max(), 300)
        gfg = scipy.interpolate.make_interp_spline(x, y, k=1)
        ynew1 = gfg(xnew1)
        ax.plot(xnew1, ynew1, color=linecolor, linestyle="-", linewidth=1)


    # plot raw datapoints
    ax.scatter(x, y, marker = marker, s=mrsize, facecolors='none', edgecolors = markercolor)
    ax.set_xlabel("postition")
    ax.set_ylabel("GC percentage (%)")

    # format xaxis ticks and labels
    CRE_len = len(y)
    if CRE_len%2==1: # odd length
        half_len = round(CRE_len/2)
        center = half_len+1
    else: # even length
        half_len = CRE_len/2
        center = half_len-0.5
    
    ax.set_xticks([0,center,CRE_len])
    ax.set_xticklabels([f"-{half_len}bp","summit",f"+{half_len}bp"])

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    # set y axis scales
    ax.set_ylim(0, 100)

    # add plot title
    ax.set_title("GC Content Distribution")
                
    # force equal aspect
    ax.set_aspect("auto")

    return position_GC_ser, fig, ax