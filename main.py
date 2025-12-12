from __future__ import division
from __future__ import print_function

import warnings
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import pytweening
import glob
import os
import re
from tqdm import *
from pathlib import Path
from PIL import Image
from os import path
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment as hung

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# setting up the style for the charts
sns.set_style("darkgrid")
mpl.rcParams['font.size'] = 12.0
mpl.rcParams['text.color'] = '#222222'
mpl.rcParams['pdf.fonttype'] = 42
current_path = Path(__file__).resolve().parent


# from https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
# This function calculates the summary statistics for the given set of points
def get_values(df):

    xm = df.x.mean()
    ym = df.y.mean()
    xsd = df.x.std()
    ysd = df.y.std()
    pc = df.corr().x.y

    return [xm, ym, xsd, ysd, pc]


# from https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
# checks to see if the statistics are still within the acceptable bounds
# with df1 as the original dataset, and df2 as the one we are testing
def is_error_still_ok(df1, df2, decimals=2):
    r1 = get_values(df1)
    r2 = get_values(df2)

    # check each of the error values to check if they are the same to the correct number of decimals
    r1 = [math.floor(r * 10 ** decimals) for r in r1]
    r2 = [math.floor(r * 10 ** decimals) for r in r2]

    # we are good if r1 and r2 have the same numbers
    er = np.subtract(r1, r2)
    er = [abs(n) for n in er]

    return np.max(er) == 0


# from https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
def save_scatter_and_results(df, iteration, directory: str, dp=72, labels=["X Mean", "Y Mean", "X SD", "Y SD", "Corr."]):

    show_scatter_and_results(df, labels=labels)
    plt.savefig(f"{directory}/{str(iteration)}.png", dpi=dp)
    plt.clf()
    plt.cla()
    plt.close()


# from https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
def show_scatter_and_results(df, labels=["X Mean", "Y Mean", "X SD", "Y SD", "Corr."]):

    res = get_values(df)
    fs = 30
    max_label_length = max(len(l) for l in labels)

    # Create figure with tight layout using GridSpec
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.8])

    # === LEFT AXIS: SCATTERPLOT ===
    ax = fig.add_subplot(gs[0, 0])
    sns.regplot(
        x="x", y="y", data=df, ci=None, fit_reg=False,
        scatter_kws={"s": 4, "alpha": 0.9, "color": "black"},
        ax=ax
    )
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect("equal", adjustable="box")  # perfect square axes

    # === RIGHT AXIS: TEXT BLOCK ===
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis("off")

    y_positions = [0.9, 0.75, 0.60, 0.45, 0.30]

    # shadow / lighter text
    for i, (label, value) in enumerate(zip(labels, res)):
        ax_text.text(
            0.0, y_positions[i],
            label.ljust(max_label_length) + ": " + format(value, "0.9f")[:-2],
            fontsize=fs, alpha=0.3, transform=ax_text.transAxes
        )

    # main bold text
    for i, (label, value) in enumerate(zip(labels, res)):
        ax_text.text(
            0.0, y_positions[i],
            label.ljust(max_label_length) + ": " + format(value, "0.9f")[:-7],
            fontsize=fs, alpha=1, transform=ax_text.transAxes
        )


# less precise chamfer algorithm, allows for multiple points to overlap, good for first initialization
def chamfer(pos_1, pos_2):

    pdist = distance_matrix(pos_1, pos_2)
    term1 = pdist.min(axis=1).mean()
    term2 = pdist.min(axis=0).mean()

    return term1 + term2


# precise hungarian algorithm, measures most optimal distances between every pair of points (O(n^3)), good for refining results
def hungarian(pos_1, pos_2):

    # square the distances so larger distances weigh more heavily
    pdist = distance_matrix(pos_1, pos_2) ** 2
    row_idcs, col_idcs = hung(pdist)

    # sqrt the distances for the final sum
    return np.sqrt(pdist)[row_idcs, col_idcs].sum()


# helper function to scale all data between 0 and 1, manually scale between 20, 80
def scale(x0):

    mtx3 = (x0 - np.min(x0)) / (np.max(x0) - np.min(x0))
    mtx3 *= 60
    mtx3 += 20

    return mtx3


# # # helper function to normalize data
# def normalize_shape(X):
#
#     X = X - X.mean(axis=0)
#     X = X / X.std(axis=0, ddof=0)
#
#     return X


# inspired by and adapted from https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
# This is the function which does one round of perturbation
# df: is the current dataset
# tar_df: is the target dataset
# sample_size: how many points we move in one perturbation
# shake: the maximum amount of movement in each iteration
# temp: the temperature, how often are we accepting bad results
# x_bounds and y_bounds: boundaries of the scatterplot, set to 0 and 100
# dis_func: which distance function to use
# min_move: the minimum distance each move should be
def perturb(df, tar_df,
            shake=0.1,
            sample_size=20,
            temp=0,
            x_bounds=[0, 100], y_bounds=[0, 100],
            dis_func=chamfer, min_move = 5):

    # this is the simulated annealing step, if "do_bad", then we are willing to
    # accept a new state which is worse than the current one
    do_bad = np.random.random_sample() < temp

    scaled_tar_df = scale(tar_df)
    old_dist = dis_func(scaled_tar_df, scale(df.to_numpy()))

    while True:

        # take multiple rows at random and shift them
        row = np.random.randint(0, len(df), sample_size)

        # save old vals
        old_vals = [df['x'][row], df['y'][row]]

        # perturb the new rows
        i_xm = df['x'][row]
        i_ym = df['y'][row]
        xm = i_xm + np.random.randn() * shake
        ym = i_ym + np.random.randn() * shake

        # if our new dataset is out of bounds then we can skip the rest, redo the above
        if not ((xm >= x_bounds[0]).all() & (xm <= x_bounds[1]).all() &
                (ym >= y_bounds[0]).all() & (ym <= y_bounds[1]).all()):
            continue

        # set new vals and compute the distance between current dataset and target dataset
        df['x'][row] = xm
        df['y'][row] = ym
        new_dist = dis_func(scaled_tar_df, scale(df.to_numpy()))

        # we accept new vals if we are closer (with a minimum amount) or if we are allowed to accept bad solution
        if (new_dist < old_dist and (abs(new_dist - old_dist) >= min_move)) or do_bad:
            break
        else:
            # set back to old vals if our solution is unacceptable
            df['x'][row] = old_vals[0]
            df['y'][row] = old_vals[1]

    return df, new_dist


# from: https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
# helper function for setting the shake, temperature and sample size
def s_curve(v):
    return pytweening.easeInOutQuad(v)


# inspired by and adapted from: https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
# this is the main function, for taking one dataset and perturbing it into a target shape
# df: the initial dataset
# target: the shape we are aiming for
# directory: where to save results
# iters: how many iterations to run the algorithm for
# num_frames: how many frames to save to disk (for animations)
# decimals: how many decimal points to keep fixed
# max_shake: the step size at the start
# min_shake: the step size near the end of the process, the step size changes fros max to min over time
# max_sample: the sample size at the start
# min_sample: the sample size near the end of the process, the sample size changes fros max to min over time
# max_temp: the temperature at the start
# min_temp: the temperature near the end of the process, the temperature changes fros max to min over time
# function_calls: list of what function to call at each iteration
#
def run_pattern(df, target, directory, iters=100000, num_frames=100, decimals=2, max_shake=0.6, min_shake=0.1,
                max_sample=20, min_sample=1,
                max_temp=0.4, min_temp=0,
                function_calls=None,
                ramp_in=False, ramp_out=False, freeze_for=0,
                labels=["X Mean", "Y Mean", "X SD", "Y SD", "Corr."],
                reset_counts=False):

    global frame_count
    global it_count

    if reset_counts:
        it_count = 0
        frame_count = 0

    # load target dataframe and scale
    r_good = df.copy()
    tar_df = pd.read_csv("{}/target_datasets/{}.csv".format(current_path, target), header=None,
                         names=['x', 'y'])
    tar_df = tar_df.to_numpy()

    # this is a list of frames that we will end up writing to file
    write_frames = [int(round(pytweening.linear(x) * iters)) for x in np.arange(0, 1, 1 / (num_frames - freeze_for))]

    if ramp_in and not ramp_out:
        write_frames = [int(round(pytweening.easeInSine(x) * iters)) for x in
                        np.arange(0, 1, 1 / (num_frames - freeze_for))]
    elif ramp_out and not ramp_in:
        write_frames = [int(round(pytweening.easeOutSine(x) * iters)) for x in
                        np.arange(0, 1, 1 / (num_frames - freeze_for))]
    elif ramp_out and ramp_in:
        write_frames = [int(round(pytweening.easeInOutSine(x) * iters)) for x in
                        np.arange(0, 1, 1 / (num_frames - freeze_for))]

    extras = [iters] * freeze_for
    write_frames.extend(extras)

    looper = trange(iters + 1, leave=True, ascii=True, desc=target + " pattern")
    best_dis = 1e9

    func_list = function_calls
    prev_func = func_list[0]

    # get the distance between point clouds at the start and create the minimum movement needed based off that and the number of iterations
    start_tot_dis = prev_func(df, scale(tar_df))
    min_move = start_tot_dis / (iters * 1.5)
    dis_progression = [0] * (iters + 1)

    # this is the main loop, were we run for many iterations to come up with the pattern
    for i in looper:

        # set the current temperature, shake, sample size and distance function depending on which iteration we are in
        t = (max_temp - min_temp) * s_curve(((iters - i) / iters)) + min_temp
        curr_shake = (max_shake - min_shake) * s_curve(((iters - i) / iters)) + min_shake
        curr_sample_size = int((max_sample - min_sample) * s_curve(((iters - i) / iters)) + min_sample)
        curr_func = func_list[i]

        # when we switch to a different distance function we need to reset the loss so we set it to an arbitrary large value
        if prev_func != curr_func:
            # curr_sample_size = 1
            best_dis = 1e9
            start_tot_dis = curr_func(df, scale(tar_df))
            min_move = start_tot_dis / (iters * 1.5)

        # main jittered result and new distance
        test_good, new_dis = perturb(r_good.copy(), temp=t, tar_df=tar_df,
                                     shake=curr_shake, sample_size=curr_sample_size, dis_func=curr_func, min_move=min_move)

        if i == 0:
            dis_progression[i] = new_dis

        # here we are checking that after the purturbation, that the statistics are still within the allowable bounds
        if is_error_still_ok(df, test_good, decimals):
            r_good = test_good

            # tracking of distance (loss) and adding it to the tqdm thing
            if new_dis < best_dis:
                best_dis = new_dis

            looper.set_description("Current loss: {} | Best loss: {}".format(str(round(new_dis, 4)), str(round(best_dis, 4))))

        dis_progression[i] = new_dis

        # save this chart to the file
        for x in range(write_frames.count(i)):
            save_scatter_and_results(r_good, target + "-image-" + format(int(frame_count), '05'), dp=150, labels=labels,
                                     directory=directory)
            r_good.to_csv(f'{directory}/{target}' + "-data-" + format(int(frame_count), '05') + ".csv", index=False, header=False)

            frame_count = frame_count + 1

        prev_func = func_list[i]

    # save the final result to the seed dataset
    # r_good.to_csv('{}/seed_datasets/{}.csv'.format(current_path, target), index=False, header=False)

    # save the loss progression
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False, facecolor='w')
    ax1.plot(dis_progression)
    ax2.plot(dis_progression)
    ax1.set_ylim(100, np.max(dis_progression))
    ax2.set_ylim(0, 5)
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig.suptitle('Progression of the distances')
    fig.supylabel('Distance')
    fig.supxlabel('Iterations')
    plt.savefig(f'{directory}/{target}' + '_progression.jpeg')
    plt.close('all')

    return r_good


# inspired by and adapted from: https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
# function to load a dataset, and then perturb it
# start_dataset: name of the starting dataset
# target: the name of the target dataset
# iterations: how many iterations to run the algorithm for
# decimals: how many decimal points to keep fixed
# num_frames: how many frames to save to disk (for animations)
# max_temp: the temperature at the start
# min_temp: the temperature near the end of the process, the temperature changes fros max to min over time
# max_shake: the step size at the start
# min_shake: the step size near the end of the process, the step size changes fros max to min over time
# max_sample_divis: integer that determines how large our sample will be, divides the number of points in df by this integer
# min_sample: the sample size near the end of the process, the sample size changes fros max to min over time
# function_calls: list of what function to call at each iteration
def do_single_run(start_dataset, target, iterations=100000, decimals=2, num_frames=100, max_temp=0.4, min_temp=0,
                  max_shake=0.6, min_shake=0.1, max_sample_divis=35, min_sample=1, function_calls=None):

    global it_count
    global frame_count

    it_count = 0
    frame_count = 0

    # load dataset
    df = pd.read_csv("seed_datasets/{}.csv".format(start_dataset), header=None, names=['x', 'y'])

    # set the maximum sample size based on the sample division arg
    max_sample = int(len(df) / max_sample_divis)

    # if we don't have specified distance functions then set the chamfer distance to be the default
    if function_calls is None:
        function_calls = [chamfer] * (iterations + 1000)

    temp = run_pattern(df, target, iters=iterations, num_frames=num_frames, directory=f'results/{start_dataset}_{target}',
                       decimals=decimals, max_temp=max_temp, min_temp=min_temp, max_shake=max_shake,
                       min_shake=min_shake, max_sample=max_sample, min_sample=min_sample, function_calls=function_calls)
    return temp


# function to extract the numeric part from the filename for sorting
def extract_number(filename):
    match = re.findall(r'\d+', filename)  # Find the first number in the filename
    return int(match[-1]) if match else 0


# inspired by and adapted from: https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
def create_gifs(shape_start, shape_end):

    # create the frames from all png files
    imgs = glob.glob(f"results/{shape_start}_{shape_end}/*.png")

    # get all the PNG files in the directory (sorted by the numeric part of the filename)
    frames = [Image.open(os.path.join(filename)) for filename in
              sorted(imgs, key=extract_number)]

    # Save into a GIF file that loops forever
    if not path.exists('progression_gifs'):
        os.mkdir('progression_gifs')

    frames[0].save(f"progression_gifs/{shape_start}_{shape_end}.gif", format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=700 // 6, loop=0)

    #######################################
    # # for 2nd round of datadance we add the previous ones for one long gif
    # # create the frames from all png files
    # imgs1 = glob.glob(f"results/circle_250_{shape_end}/*.png")
    #
    # # get all the PNG files in the directory (sorted by the numeric part of the filename)
    # frames1 = [Image.open(os.path.join(filename)) for filename in
    #            sorted(imgs1, key=extract_number)]
    #
    # imgs2 = glob.glob(f"results/{shape_start}_{shape_end}/*.png")
    #
    # # get all the PNG files in the directory (sorted by the numeric part of the filename)
    # frames2 = [Image.open(os.path.join(filename)) for filename in
    #            sorted(imgs2, key=extract_number)]
    #
    # frames3 = frames1 + frames2
    #
    # frames[0].save(f"progression_gifs/circle_{shape_start}_{shape_end}.gif", format='GIF',
    #                append_images=frames3[1:],
    #                save_all=True,
    #                duration=700 // 6, loop=0)


if __name__ == '__main__':

    # SET ALL ARGUMENTS HERE
    it = 150000  # 100000 for 2nd round of datadance, 150000 normal
    de = 2
    frames = 100
    max_shake = 0.5  # 0.25 for 2nd round of datadance, 0.5 normal
    min_shake = 0.1
    max_temp = 0.4  # 0.15 for 2nd round of datadance, 0.4 normal, 0.15 test
    min_temp = 0
    max_sample_divis = 35  # 40 for 2nd round of datadance, 35 normal
    min_sample = 1

    # distance functions
    # func_list = [chamfer] * int(it * 0.9) + [hungarian] * (int(it * 0.1) + 1000)  # for normal Datasaurus Dozen
    func_list = [hungarian] * int(it * 0.85) + [chamfer] * (int(it * 0.15) + 1000)  # for datadance
    # func_list = [hungarian] * int(it * 0.9) + [chamfer] * (int(it * 0.1) + 1000)

    shape_start = 'circle_250'
    n_points = 250
    shape_ends = ['datadance_250_{}'.format(cnt) for cnt in range(1, 25)]

    # uncomment for shape replication for Datasaurus Dozen
    # shape_ends = ['x', 'h_lines', 'v_lines', 'wide_lines', 'high_lines', 'slant_up', 'slant_down', 'circle', 'star', 'down_parab', 'bullseye', 'dots']
    # for i in range(len(shape_ends)):
    #     shape_ends[i] = shape_ends[i] + '_' + str(n_points)

    for shape_end in shape_ends:

        # shape_start = shape_end  # for 2nd round of datadance, comment out for others (this is an easy way of using e.g. 'datadance_250_1' as seed dataset
        # and the same as target dataset for extra refinements
        print('Doing shape: ' + shape_end)
        save_directory = f'results/{shape_start}_{shape_end}'

        # check if we have the seed dataset and target dataset
        if (shape_start + '.csv' in os.listdir('seed_datasets')) and (shape_end + '.csv' in os.listdir('target_datasets')):

            # make the directory if we have to
            if path.exists(save_directory):
                print(f"File {save_directory} exists")
            Path(save_directory).mkdir(exist_ok=True)

            do_single_run(shape_start, shape_end, iterations=it, decimals=de, num_frames=frames, max_shake=max_shake,
                          min_shake=min_shake, max_temp=max_temp, min_temp=min_temp, max_sample_divis=max_sample_divis,
                          min_sample=min_sample, function_calls=func_list)
        else:
            if shape_start + '.csv' not in os.listdir('seed_datasets'):
                print('Starting shape is incorrect')
            elif shape_end + '.csv' not in os.listdir('target_datasets'):
                print("End shape is incorrect")

        create_gifs(shape_start, shape_end)
