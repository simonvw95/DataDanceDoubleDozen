import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from main import is_error_still_ok, get_values

"""
lines = list of segments, where each segment is [[x1,y1], [x2,y2]]
Returns Nx2 array of evenly spaced points along all segments.
"""


def sample_points_from_lines(lines, n_points):

    # convert to numpy
    lines = [np.array(seg, dtype=float) for seg in lines]

    # compute segment lengths
    lengths = []
    for seg in lines:
        p1, p2 = seg
        lengths.append(np.linalg.norm(p2 - p1))
    lengths = np.array(lengths)

    total_length = lengths.sum()
    if total_length == 0:
        return np.tile(lines[0][0], (n_points, 1))  # degenerate single point

    # number of points per segment (proportional to length)
    pts_per_segment = np.round(n_points * lengths / total_length).astype(int)

    # fix rounding errors: ensure total = n_points
    diff = n_points - pts_per_segment.sum()
    if diff != 0:
        # add/subtract remaining points to the longest segment(s)
        order = np.argsort(-lengths)
        for idx in order[:abs(diff)]:
            pts_per_segment[idx] += np.sign(diff)

    # sample points
    result = []
    for (seg, k) in zip(lines, pts_per_segment):
        if k <= 0:
            continue
        p1, p2 = seg
        ts = np.linspace(0, 1, k, endpoint=False)
        pts = p1 + (p2 - p1)[None, :] * ts[:, None]
        result.append(pts)

    result = np.vstack(result)

    # if rounding caused too many points, trim.
    if len(result) > n_points:
        result = result[:n_points]

    return result


def generate_dataset(line_shape, n_points):

    points = None

    if line_shape == 'x':
        l1 = [[20, 0], [100, 100]]
        l2 = [[20, 100], [100, 0]]
        lines = [l1, l2]

    elif line_shape == "h_lines":
        lines = [[[0, y], [100, y]] for y in [10, 30, 50, 70, 90]]

    elif line_shape == 'v_lines':
        lines = [[[x, 0], [x, 100]] for x in [10, 30, 50, 70, 90]]

    elif line_shape == 'wide_lines':
        lines = [[[10, 0], [10, 100]], [[90, 0], [90, 100]]]

    elif line_shape == 'high_lines':
        lines = [[[0, 10], [100, 10]], [[0, 90], [100, 90]]]

    elif line_shape == 'slant_up':
        lines = [
            [[0, 0], [100, 100]],
            [[0, 30], [70, 100]],
            [[30, 0], [100, 70]],
            [[50, 0], [100, 50]],
            [[0, 50], [50, 100]],
        ]

    elif line_shape == 'slant_down':
        lines = [
            [[0, 100], [100, 0]],
            [[0, 70], [70, 0]],
            [[30, 100], [100, 30]],
            [[0, 50], [50, 0]],
            [[50, 100], [100, 50]],
        ]

    elif line_shape == 'circle':

        cx, cy = 54.26, 47.83
        r = 30

        # approximate circle using many tiny line segments
        theta = np.linspace(0, 2 * np.pi, 200)
        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta)
        pts = np.column_stack([xs, ys]).tolist()

        # convert consecutive points into line segments
        lines = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
        lines.append([pts[-1], pts[0]])

    elif line_shape == 'bullseye':
        cx, cy = 54.26, 47.83
        radii = [18, 37]  # two rings

        lines = []
        for r in radii:
            theta = np.linspace(0, 2 * np.pi, 100)
            xs = cx + r * np.cos(theta)
            ys = cy + r * np.sin(theta)
            pts = np.column_stack([xs, ys]).tolist()

            # convert to line segments
            segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
            segs.append([pts[-1], pts[0]])  # close circle
            lines.extend(segs)

    elif line_shape == 'dots':

        xs = [25, 50, 75]

        ys = [20, 50, 80]

        pts = [(x, y) for x in xs for y in ys]

        # distribute n_points evenly across the 9 dots

        n_per_dot = n_points // len(pts)

        remainder = n_points % len(pts)

        result = []

        for i, (px, py) in enumerate(pts):
            k = n_per_dot + (1 if i < remainder else 0)

            dot_points = np.tile([px, py], (k, 1))

            result.append(dot_points)

        points = np.vstack(result)

    elif line_shape == 'star':
        star_pts = [10, 40, 40, 40, 50, 10, 60, 40, 90, 40,
                    65, 60, 75, 90, 50, 70, 25, 90, 35, 60]
        pts = [star_pts[i:i + 2] for i in range(0, len(star_pts), 2)]
        pts = [[p[0] * 0.8 + 20, 100 - p[1]] for p in pts]
        pts.append(pts[0])  # close the shape
        lines = [pts[i:i + 2] for i in range(len(pts) - 1)]

    elif line_shape == 'down_parab':
        curve = [[x, -((x - 50) / 4) ** 2 + 90] for x in np.arange(0, 100, 3)]
        lines = [curve[i:i + 2] for i in range(len(curve) - 1)]
        lines = np.array(lines)
        lines = (lines - np.min(lines)) / (np.max(lines) - np.min(lines)) * 100
        lines = lines.tolist()

    elif line_shape == 'random_cloud':
        # load original dataframe
        df = pd.read_csv('seed_datasets/random_cloud_142.csv', header=None, names=['x', 'y'])
        print("Initial mean/std/corr:\n")
        print(get_values(df))
        df_full = generate_refined_2d_cloud(df, n_points)
        print("Final mean/std/corr:\n")
        print(get_values(df_full))
        points = df_full.to_numpy()
    else:
        raise ValueError(f"Unknown shape: {line_shape}")

    # generate points
    if points is None:
        lines = np.array(lines)
        lines = (lines - np.min(lines)) / (np.max(lines) - np.min(lines)) * 100
        lines *= 0.6
        lines += 24
        lines = lines.tolist()
        points = sample_points_from_lines(lines, n_points)

    outdir = "target_datasets"
    os.makedirs(outdir, exist_ok=True)

    # save file
    outfile = f"{outdir}/{line_shape}_{n_points}.csv"
    df = pd.DataFrame(points, columns=['x', 'y'])
    df.to_csv(outfile, index=False, header=False)

    plt.scatter(x = df['x'], y = df['y'])
    plt.savefig(f"{outdir}/{line_shape}_{n_points}.png")
    plt.close('all')

    return outfile


"""
Generate a 2D point cloud with specified mean, standard deviations, correlation,
and number of points, clipped to [0,100] and refined to match statistics.
"""


def generate_refined_2d_cloud(init_df, n_points, seed=None, noise_scale=0.5):

    if seed is not None:
        np.random.seed(seed)

    # step 1: Generate initial cloud
    Z = np.random.randn(n_points, 2)
    xm, ym, xsd, ysd, rho = get_values(init_df)
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=0)
    cov = np.array([[xsd**2, rho*xsd*ysd],
                    [rho*xsd*ysd, ysd**2]])

    L = np.linalg.cholesky(cov)
    X = Z @ L.T
    X[:, 0] += xm
    X[:, 1] += ym
    X = np.clip(X, 0, 100)
    df_full = pd.DataFrame(X, columns=['x', 'y'])

    # step 2: Iteratively refine to match target statistics
    while True:
        idx = np.random.randint(0, n_points)
        new_point = df_full.iloc[idx].values + np.random.randn(2) * noise_scale
        new_point = np.clip(new_point, 0, 100)

        df_temp = df_full.copy()
        df_temp.iloc[idx] = new_point

        # accept if statistics closer to target
        old_err = np.max(np.abs(np.array(get_values(df_full)) - np.array([xm, ym, xsd, ysd, rho])))
        new_err = np.max(np.abs(np.array(get_values(df_temp)) - np.array([xm, ym, xsd, ysd, rho])))
        if new_err < old_err:
            df_full.iloc[idx] = new_point

        if is_error_still_ok(init_df, df_full, decimals=2):

            return df_full


if __name__ == '__main__':

    ############################################################################################################
    # very important variable, sets the number of points we want to use for each data set
    N = 250
    ############################################################################################################

    line_shapes = ['x', 'h_lines', 'v_lines', 'wide_lines', 'high_lines', 'slant_up', 'slant_down', 'circle', 'star', 'down_parab', 'bullseye', 'dots', 'random_cloud']

    # N = 855
    # line_shapes = ['random_cloud', 'circle']

    for line_shape in line_shapes:

        generate_dataset(line_shape, N)

    print('Done generating all shapes for specified number of points')
