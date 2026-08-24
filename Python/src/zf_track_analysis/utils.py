"""
    Auxiliary functions used in core.py to compute kinematic parameters.
"""
from copy import copy
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import h5py

def compute_derivative(values : np.ndarray,
                       pad=False,
                       delta_x : float = 1.0,
                       x = None) -> np.ndarray:
    """compute derivative using the central difference method with a homogeneous
        delta_x. If delta_x is not given, simple computes the central difference.

    Args:
        values (np.ndarray): array with values whose derivatives will be computed.
            pad (bool, optional): pad start and end of vector with first and last values,
            respectively. Defaults to False.
        x (float or np.ndarray, optional): if Float, x is considered the denominator 
            to calculate the derivative (delta_x) if x is an np.array, it is considered
            the points in x coordinate, from wich differences will be computed and these
            differences will be used then as denominator for the derivative calculation.
             Defaults to 1.0.
    """
    if x is None:
        if pad:
            right = np.diff(values, append=values[-1])
            left = np.diff(values, prepend=values[0])
        if not pad:
            right= np.diff(values)
            left = np.diff(values)

            right = np.insert(right, 0, np.nan)
            left = np.insert(left, -1, np.nan)

        divider = np.hstack([delta_x, np.repeat(2*delta_x, len(values)-2), delta_x])
        derivative = np.nansum([left,right], axis=0)/ divider

    if x is not None:
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be np.ndarray of shape (n,)")
        if len(x.shape) != 1:
            raise ValueError("x must be 1-dimensional")
        if x.shape != values.shape:
            raise ValueError("Input 'values' and 'x' must have the same shape.")
        if pad:
            right = np.diff(values,append=values[-1])
            left = np.diff(values,prepend=values[0])

            right_dx = np.diff(x, append= x[-1])
            left_dx = np.diff(x, prepend =x[0])

            # right = right / right_dx
            # left = left / left_dx
            right = np.divide(right, right_dx, out=np.full_like(right, np.nan, dtype=float), where=right_dx!=0)
            left = np.divide(left, left_dx, out=np.full_like(left, np.nan, dtype=float), where=left_dx!=0)

        if not pad:
            # diffs = np.diff(values) / np.diff(x)
            dx = np.diff(x)
            diffs = np.divide(np.diff(values), dx, out=np.full_like(dx, np.nan, dtype=float), where=dx!=0)
            right = np.insert(diffs, -1, np.nan)
            left = np.insert(diffs, 0, np.nan)

        derivative = np.nansum([left,right], axis=0) / 2

    return derivative

def get_body_center(dataframe : pd.DataFrame,
                    bodyparts : list,
                    new_name = 'mean_body_keypoint',
                    pcutoff = 0.8,
                    return_fraction_lost = False):

    """ Compute a new body keypoint (default named 'mean_body_keypoint')
        from a DeepLabCut results dataframe and a list of bodyparts,
        by calculating the centroid from multiple keypoints of the original pose model.

    Args:
        dataframe (pd.DataFrame): DeepLabCut tracking result of a video from .h5 file
        bodyparts (list): list of strings with bodyparts matching existing
            bodyparts in dataframe. eg.: ['swim_bladder', 'R_eye_top', 'L_eye_top']
        mask_likelihood (bool, optional): Choose whether or not to filter
            datapoints with likelihood lower than pcutoff. Defaults to False.
        pcutoff (float, optional): likelihood cutoff to calculate centroid.
            This means, if the likelihood of that keypoint's prediction is lower than the pcutoff,
            that point will not be used to calculate the centroid. Defaults to 0.8.
        return_fraction_lost (bool, optional): if True, also return the fraction of
            datapoints (across all given bodyparts) that fell below pcutoff. Defaults to False.

    Returns:
        df (pandas.DataFrame): dataframe with the new keypoint data named
        'mean_body_keypoint' calculated from the mean of bodyparts coordinates.
        fraction_lost (float): only returned if return_fraction_lost is True. np.nan if
        pcutoff is None (no filtering was performed).
    """

    # merge columns from all bodyparts
    scorer = dataframe.columns.get_level_values(0)[0]
    merged_x = np.array([dataframe[scorer][i]['x'] for i in bodyparts])
    merged_y = np.array([dataframe[scorer][i]['y'] for i in bodyparts])
    merged_likelihood = np.array([dataframe[scorer][i]['likelihood'] for i in bodyparts])

    # exclude datapoints with likelihood less than pcutoff for computations
    # then interpolate the missing points
    fraction_lost = np.nan
    if pcutoff is not None:
        mask = merged_likelihood < pcutoff

        fraction_lost = np.sum(mask)/(len(bodyparts)*dataframe.shape[0])
        if fraction_lost > 0.1:
            print(f"    (get_body_center) fraction of points that do not pass pcutoff: {fraction_lost:.2f}")
        # Set low-confidence points to NaN to mark them as missing
        merged_x[mask] = np.nan
        merged_y[mask] = np.nan
        merged_likelihood[mask] = np.nan

        num_bodyparts, num_frames = merged_x.shape
        frames = np.arange(num_frames)

        # Interpolate each bodypart's time series independently
        for i in range(num_bodyparts):
            # interpolate x coordinates
            valid_mask_x = ~np.isnan(merged_x[i, :])
            merged_x[i, :] = np.interp(frames, frames[valid_mask_x], merged_x[i, valid_mask_x])

            # interpolate y coordinates
            valid_mask_y = ~np.isnan(merged_y[i, :])
            merged_y[i, :] = np.interp(frames, frames[valid_mask_y], merged_y[i, valid_mask_y])

    # compute mean x and y coordinates and likeligood of the centroid
    mean_x = np.reshape(np.mean(merged_x, axis=0), (-1, 1))
    mean_y = np.reshape(np.mean(merged_y, axis=0), (-1, 1))
    mean_likelihood = np.reshape(np.mean(merged_likelihood, axis=0), (-1, 1))

    # feed the means to a pandas dataframe and return with with
    # new name 'mean_body_keypoint'
    tuples = list(zip([scorer]*3, [new_name]*3, ['x', 'y', 'likelihood']))
    index = pd.MultiIndex.from_tuples(tuples,
                                      names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame(columns=index,
                      data=np.hstack((mean_x, mean_y, mean_likelihood)))

    if return_fraction_lost:
        return df, fraction_lost
    return df

def compute_base_and_heading(dataframe : pd.DataFrame,
                             time_array : np.ndarray,
                             origin : list,
                             point : list,
                             **kwargs):

    """ Computes the base points and with them also obtains the heading vector.

        This function treats the 'origin' as the vector's starting
        coordinates and the 'point' as its ending coordinates (the tip).

        The calculation is performed by subtracting the origin from the point,
        which can be visualized as follows:

                          P (point)
                         /|
                        / |
                       /  |
                      /   |
                     /    |
        (origin)  o ------+

        OR
            o --------------------> P
            (x,y) - origin       (x,y) - point

        heading_vector = P - o
    
    Args:
        dataframe (pd.DataFrame): dlc results dataframe
        time_array (np.ndarray): array of shape (n_obs, 2), where column 0 has frame number
            values and column 1, the time points in seconds.
        origin (list): list of str with the name of the keypoints to be used to calculate 
            the origin point of the heading vector. 
        point (list): list of str with the name of the keypoints to be used to calculate
            the 'tip' point of the heading vector. 
        truncate (int):  calculate vector metrics using fewer tail points by setting 
            truncate value to a number lower than 10, (10 is the total number of tail
            points in our pose model), this is useful when the tail points 
            further from the head are two jittery/noisy.
    kwargs:
        'smooth' (int): window size, in frames (smooth > 3), with which to smooth the
            heading angle data. Uses the savitzky-golay filter (polyorder=3). 
            This will help smooth the mean tail curvature metrics (mtc, mtc_velocity, mtc_accel)

    Returns:
        pd.DataFrame : base dataframe with all vector-based metrics columns 
            (example with only 2 tails segments for brevity)

        [(         'origin',               'x'),
         (         'origin',               'y'),
         (          'point',               'x'),
         (          'point',               'y'),
         ( 'heading_vector',               'x'),
         ( 'heading_vector',               'y'),
         ('heading_degrees', 'heading_degrees')]
    """

    if 'smooth' in kwargs:
        SMOOTH = kwargs['smooth']
    else:
        SMOOTH = None

    scorer = dataframe.columns.get_level_values(0)[0]

    if len(origin) > 1:
        origin_df = get_body_center(dataframe,
                                    origin,
                                    new_name='origin')
        origin_matrix = origin_df[scorer]['origin'][['x', 'y']].values[:]
    else:
        origin_matrix = dataframe[scorer][origin[0]][['x', 'y']].values[:]

    if len(point) > 1:
        point_df = get_body_center(dataframe,
                                point,
                                new_name='point')
        point_matrix = point_df[scorer]['point'][['x', 'y']].values[:]
    else:
        point_matrix = dataframe[scorer][point[0]][['x', 'y']].values[:]

    # compute the heading vector and normalize it to make it
    # a unit vector
    heading_vector = point_matrix - origin_matrix
    heading_vector = heading_vector / np.reshape(np.linalg.norm(heading_vector, axis=1), (-1, 1))
    ref_vector = np.array([1, 0])

    dot = np.dot(heading_vector, ref_vector)
    det = [np.linalg.det([i, ref_vector]) for i in heading_vector]
    heading = np.degrees(np.arctan2(det, dot) + np.pi)
    if SMOOTH is not None:
        heading = savgol_filter(heading, SMOOTH, polyorder=3, axis=0)

    results_structure = [['origin',
                          'origin',
                          'point',
                          'point',
                          'heading_vector', 
                          'heading_vector',
                          'heading_degrees'],
                        ['x',
                         'y',
                         'x',
                         'y',
                         'x',
                         'y',
                         'heading_degrees']]

    tuples = list(zip(*results_structure))
    results_structure = pd.MultiIndex.from_tuples(tuples)
    df = pd.DataFrame(columns=results_structure)

    df[df.columns[0:2]] = origin_matrix
    df[df.columns[2:4]] = point_matrix
    df[df.columns[4:6]] = heading_vector
    df[df.columns[6]] = heading

    return df

def compute_turn_angles(heading_df : pd.DataFrame, time_array: np.ndarray) -> pd.DataFrame:
    """ Computes turning angles between headings from consecutive frames.
        Adds a new column to heading_df dataframe 
            (    'turn_angles',     'turn_angles')

    Args:
        heading_df (pd.DataFrame): vector and heading dataframe output 
            of compute_base_vectors_and_heading()

    Returns:
        pd.DataFrame: returns the same dataframe expanded with turning angles as a new column. 
    """

    t_diff = np.diff(time_array[:,0])
    t_diff = np.insert(t_diff, [0], [0], axis=0)

    turn_angles = np.empty((len(t_diff), 1))
    turn_angles[:] = np.nan
    turn_angles = turn_angles.astype(np.longdouble)

    # generate a mask for the pair v0, v1
    idx_v1 = t_diff == 1.0
    idx_v0 = copy(idx_v1)
    idx_v0[0:-1] = idx_v1[1:]
    idx_v0[-1] = False

    v1 = heading_df['heading_vector'][['x', 'y']].values[idx_v1]
    v0 = heading_df['heading_vector'][['x', 'y']].values[idx_v0]
    v1 = v1 / np.reshape(np.linalg.norm(v1, axis=1), (-1, 1))
    v0 = v0 / np.reshape(np.linalg.norm(v0, axis=1), (-1, 1))

    v1 = np.hstack((v1, np.zeros((v1.shape[0], 1))))
    v0 = np.hstack((v0, np.zeros((v0.shape[0], 1))))
    dot_product = np.einsum('ij,ij->i', v1, v0)
    cross_product = np.cross(v1, v0)
    dot_with_cross = np.dot(cross_product, [0, 0, 1])
    turn_angles[idx_v1] = np.reshape(np.arctan2(dot_with_cross, dot_product), (-1, 1))

    # temp_df = pd.DataFrame(turn_angles)
    # results_structure = [['turn_angles'], ['turn_angles']] 
    # tuples = list(zip(*results_structure))
    # temp_df.columns = pd.MultiIndex.from_tuples(tuples)

    # # place the dataframe into the existing heading_df
    new_columns = [('turn_angles', 'turn_angles')]
    heading_df[new_columns] = turn_angles

    return heading_df

def compute_tail_curvature_metrics(heading_vector : np.ndarray,
                                   time_array : np.ndarray,
                                   tail_points : np.ndarray,
                                   truncate=None,
                                   **kwargs):
    """ Computes the vector-based tail curvature metrics. Adds the following new columns to the
        heading_df pd.DataFrame 

         (  'tail_vector_1',               'x'),
         (  'tail_vector_1',               'y'),
         (  'tail_vector_2',               'x'),
         (  'tail_vector_2',               'y'),
         (   'tail_angle_1',           'angle'),
         (   'tail_angle_2',           'angle'),
         (            'mtc',             'mtc'),
         (   'mtc_velocity',    'mtc_velocity'),
         (      'mtc_accel',       'mtc_accel')
    Args:
        heading_vector (np.ndarray): normalized heading vector matrix (n-by-2)
        tail_points (np.ndarray): tail points coordinate matrix, if using our dlc model
        it is commonly shape n-by-22 (swim bladder + 10 tail point coordinates)
        truncate (int, optional): tail point number where tail angles and mean tail point 
            calculation will be truncated (example, truncate of 8 will only use the 
            first 8 tail points for angles and mean tail curvature calculation). 
            Defaults to None. If none uses all tail points to 
            calculate tail angles and mean tail curvature.
    kwargs:
        'smooth' (int): window size, in frames (smooth > 3), with which to smooth the tail 
            segment angle data. Uses the savitzky-golay filter (polyorder=3). 
            This will help smooth the mean tail curvature metrics (mtc, mtc_velocity, mtc_accel)
    
    Raises:
        ValueError: If the num. of observations in heading vector (rows) does not match num. obs.
            (rows) in tail_points, stop computations.

    Returns:
        dictionary: dictionary with two keys: 'tail_angles' and 'mtc', containing with tail angles 
            (n-by-m, m=truncate) and mean tail curvature results (n-by-1), respectively.
    """

    n_obs, n_cols = tail_points.shape

    if 'smooth' in kwargs:
        SMOOTH = kwargs['smooth']
    else:
        SMOOTH = None

    if truncate is None:
        truncate = int((n_cols/2 - 1))

    # print(f"truncate value is now {truncate} and number of tail points is {n_cols/2-1}")

    diff = int((n_cols/2 -1) - truncate)

    # print(f"""difference in number of tail points to truncate and actual
            #   number of tail points: {diff}""")

    if diff != 0:
        stop_point = - diff*2
        tail_vectors = tail_points[:, 2:stop_point] - tail_points[:, :stop_point-2]
    else:
        tail_vectors = tail_points[:, 2:] - tail_points[:, :-2]

    n_obs, n_cols = tail_vectors.shape

    # reshape data in a way that we can compute the angles efficiently
    flat_tail_vectors = tail_vectors.reshape((-1,2))

    repeated_hvs = np.tile(-heading_vector, (1,truncate))
    repeated_hvs = repeated_hvs.reshape((-1,2))

    #check if shapes match, otherwise stop computation
    if flat_tail_vectors.shape != repeated_hvs.shape:
        raise ValueError("Tail vectors not the same size as repeated heading vectors")

    #add the third dimension populated with zeros for the cross product
    repeated_hvs = np.hstack((repeated_hvs, np.zeros((repeated_hvs.shape[0],1))))

    #normalize tail vectors and add third dimension populated with zeros for cross product
    flat_tail_vectors = flat_tail_vectors / np.reshape(np.linalg.norm(flat_tail_vectors, axis=1), \
                                                       (-1, 1))
    flat_tail_vectors = np.hstack((flat_tail_vectors, np.zeros((flat_tail_vectors.shape[0],1))))

    # compute tail angles with respect to inverted heading vector
    dot_product = np.einsum('ij,ij->i', flat_tail_vectors, repeated_hvs)
    cross_product = np.cross(flat_tail_vectors, repeated_hvs)
    dot_with_cross = np.dot(cross_product, [0, 0, 1])
    tail_angles = np.reshape(np.arctan2(dot_with_cross, dot_product), (-1, truncate))

    tail_angles = np.degrees(tail_angles)

    # compute mean tail curvature
    if SMOOTH is not None:
        # print("smoothing tail angles")
        tail_angles = savgol_filter(tail_angles, SMOOTH, polyorder=3, axis=0)
    mean_tail_curvature = np.mean(tail_angles, axis=1)

    full_output = {'tail_vectors': flat_tail_vectors[:,:2].reshape((n_obs, truncate*2)),
                   'tail_angles': tail_angles,
                   'mtc': mean_tail_curvature}

    results_structure = [[],[]]

    for i in range(1,truncate+1):
        results_structure[0].extend(['tail_vector_' + str(i), 'tail_vector_' + str(i)])
    for i in range(1,truncate+1):
        results_structure[0].extend(['tail_angle_' + str(i)])

    results_structure[1].extend(['x', 'y'] * truncate)
    results_structure[1].extend(['angle'] * truncate)
    results_structure[0].extend(['mtc', 'mtc_velocity', 'mtc_accel'])
    results_structure[1].extend(['mtc', 'mtc_velocity', 'mtc_accel'])

    mtc_velocity = compute_derivative(mean_tail_curvature, x=time_array[:,1], pad=True)
    mtc_accel = compute_derivative(mtc_velocity, x=time_array[:,1], pad=True)

    tuples = list(zip(*results_structure))
    results_structure = pd.MultiIndex.from_tuples(tuples)
    df = pd.DataFrame(data=np.hstack([full_output['tail_vectors'],
                                      full_output['tail_angles'],
                                      full_output['mtc'].reshape((-1,1)),
                                      mtc_velocity.reshape((-1,1)),
                                      mtc_accel.reshape((-1,1))]), columns=results_structure)

    return df

def find_repeats(input_array) -> np.ndarray:
    """
    Finds consecutive repeated values in a numeric array.

    This is a Python translation of the MATLAB `find_repeats` function.
    It calculates the number of consecutive repetitions for each element.

    Args:
        input_array (np.ndarray): A 1D numeric array.

    Returns:
        np.ndarray: An array where each element contains the number of
                    consecutive repetitions for the corresponding input value.
    """
    if input_array.size == 0:
        return np.array([])

    # True where values change.
    d = np.concatenate(([True], np.diff(input_array) != 0, [True]))

    # Indices of the start of each block.
    idx = np.where(d)[0]

    # Length of each block of identical values.
    n = np.diff(idx)

    # Replicate block lengths to match original array size.
    return np.repeat(n, n)

def quick_smooth(input_array, smooth_val) -> np.ndarray:
    """
    Performs a fast moving average smoothing on an array.

    Args:
        input_array (np.ndarray): A 1D numeric array to be smoothed.
        smooth_val (int): An odd integer specifying the window size.

    Returns:
        np.ndarray: The smoothed array. The length will be
                    `len(input) - smooth_val + 1`.
    """
    if smooth_val % 2 != 1:
        raise ValueError("smooth_val must be an odd integer.")

    return np.convolve(input_array.reshape((-1,)), np.ones(smooth_val), 'valid') / smooth_val

def process_bout_overlaps(bout_metric : dict, merge_threshold=0.95) -> dict:
    """
    Removes or merges overlapping bouts from a bout metrics dictionary.

    This function iterates through bouts and resolves overlaps based on the
    Intersection over Union (IoU).

    - If IoU > merge_threshold, the shorter bout is removed.
    - If 0 < IoU <= merge_threshold, the bouts are merged.

    Args:
        bout_metric (dict): A dictionary containing 'onset' and 'offset' keys
                            with numpy arrays of bout start and end frames.
        merge_threshold (float, optional): The IoU threshold to decide between
                                           merging and removing. Defaults to 0.95.

    Returns:
        dict: A new dictionary with overlapping bouts resolved.
    """
    if 'onset' not in bout_metric or 'offset' not in bout_metric:
        raise ValueError("bout_metric must contain 'onset' and 'offset' keys.")

    onsets = bout_metric['onset'].copy()
    offsets = bout_metric['offset'].copy()

    # Sort bouts by onset time to simplify comparisons
    sort_indices = np.argsort(onsets)
    onsets = onsets[sort_indices]
    offsets = offsets[sort_indices]

    i = 0
    while i < len(onsets) - 1:
        start_a, end_a = onsets[i], offsets[i]
        start_b, end_b = onsets[i+1], offsets[i+1]

        # Calculate Intersection and Union
        intersection_start = max(start_a, start_b)
        intersection_end = min(end_a, end_b)
        intersection_len = max(0, intersection_end - intersection_start)

        if intersection_len > 0:
            union_start = min(start_a, start_b)
            union_end = max(end_a, end_b)
            union_len = union_end - union_start
            iou = intersection_len / union_len

            if iou > merge_threshold:
                # High overlap: remove the shorter bout
                duration_a = end_a - start_a
                duration_b = end_b - start_b
                if duration_a >= duration_b:
                    onsets = np.delete(onsets, i + 1)
                    offsets = np.delete(offsets, i + 1)
                else:
                    onsets = np.delete(onsets, i)
                    offsets = np.delete(offsets, i)
            else:
                # Partial overlap: merge the bouts
                onsets[i] = union_start
                offsets[i] = union_end
                onsets = np.delete(onsets, i + 1)
                offsets = np.delete(offsets, i + 1)

            # After a merge/delete, restart check from the current index
            continue
        i += 1

    processed_bout_metric = {'onset': onsets, 'offset': offsets}
    return processed_bout_metric

def sleap_to_dlc_format(sleap_hdf5_path, indices=None):
    """Converts a sleap hdf5 file to a pandas DataFrame
    Args:
        sleap_hdf5_path (string): path to sleap predictions hdf5 file
        indices (list, optional): list of indices of the frames to extract. Default is to use all frames. 

    Returns:
        _type_: pd.DataFrame
    """    
    with h5py.File(sleap_hdf5_path, "r") as f:
        locations = f["tracks"][:].T
        likelihoods = f['point_scores'][:].T
        
        node_names = [n.decode() for n in f['node_names'][:]]

        if indices is None:
            locations = locations[:,:,:,0].reshape(-1,len(node_names)*2)
            likelihoods = likelihoods[:,:,0].reshape(-1, len(node_names))

            n_data_pts, _ = f['instance_scores'][:].T.shape
            results = np.empty((n_data_pts, 45))

            results[:, 0::3] = locations[:,0::2]
            results[:, 1::3] = locations[:,1::2]
            results[:, 2::3] = likelihoods
        else:
            locations = locations[indices,:,:,0].reshape(-1,len(node_names)*2)
            likelihoods = likelihoods[indices,:,0].reshape(-1, len(node_names))

            results = np.empty((len(indices), 45))
            results[:, 0::3] = locations[indices,0::2]
            results[:, 1::3] = locations[indices,1::2]
            results[:, 2::3] = likelihoods[indices]
   
        columns = create_df_header(node_names)
        results_structure = pd.MultiIndex.from_tuples(columns)
        if indices is not None:
            df = pd.DataFrame(data=results, columns=results_structure, index=indices)
        else:
            df = pd.DataFrame(data=results, columns=results_structure)

    return df

def create_df_header(keypoint_names: list):
    """From a list of keypoint names, create the DeepLabCut output style header. 

    Args:
        keypoint_names (list): list of string with the names of the animal pose keypoints.

    Returns:
        list: list of tuples that form the header of a pd.MultiIndex to be fed to 
        pd.MultiIndex_from_tuples().
    """

    top_level = []
    bottom_level = []
    for keypt in keypoint_names:
        top_level.extend([keypt]*3)
        bottom_level.extend(['x', 'y', 'likelihood'])

    scorer_level = ['scorer']*len(top_level)
    columns_structure = [scorer_level, top_level, bottom_level]
    columns_structure = list(zip(*columns_structure))

    return columns_structure
