"""
    Core functions to compute kinematic parameters from free-swimming behavioural
    data results obtained from pose-estimation models.
"""
import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks
from scipy.fft import fft, fftfreq
from .utils import compute_derivative, get_body_center
from .utils import compute_base_and_heading, compute_turn_angles
from .utils import compute_tail_curvature_metrics
from .utils import quick_smooth, find_repeats, process_bout_overlaps

def get_point_metrics(dataframe : pd.DataFrame,
                      bodypart : str,
                      mask=None,
                      px_tolerances=(0.0, 1000),
                      movavg=None,
                      **kwargs) -> pd.DataFrame:
    """ Computes point-based metrics from a pd.DataFrame with a dlc result structure.  
        Calculates normalized x and y positions, distance from image center (for square videos),
         distance travelled, speeds and acceleration.
        Variable names: 
            'x', 'y', 'x_norm', 'y_norm', 'likelihood', 'dist_travelled',
            'velocity', 'acceleration', 'dist_to_center'
        additional variables (if 'movavg' is in kwargs)
            'x_smooth', 'y_smooth', 'x_norm_smooth', 'y_norm_smooth',
            'dist_travelled_smooth', 'velocity_smooth', 'acceleration_smooth',
            'dist_to_center_smooth'
    Args:
        dataframe (pd.DataFrame): dataframe in the same structure of a dlc output from h5/csv file.
        bodypart (str): str with the name of the body part in the dataframe to be used to compute 
            the point metrics. after getting body center, we tend to use 'mean_body_keypoint' but 
            you can use an existing keypoint if you prefer (e.g. 'swim_bladder')
        mask (np.ndarray): a boolean mask that will filter out unwanted datapoints. 
        px_tol (tuple): two-element tuple with a low and high distance travelled (in pixels) 
            tolerance. Values lower than the first element and values higher than the second
            element, will be set to zero. 
    kwargs:
        'resolution' (float) : resolution in mm per pixel of the video that originated the dlc
            result file. 
        'frame_size' (int): width/height of the video frames (currently this calculation is only
            meaningful for square videos width=height)
        'fps' (int): framerate of the video that originated the dlc result file
        'movavg' (int): window size (in frames) to compue smoothed coordinates 
    Returns:
        pd.DataFrame : result dataframe with all the point metrics
    """

    ### I think it may make sense to change the output of this function to be a dictionary
    scorer = dataframe.columns.get_level_values(0)[0]

    # Process kwargs
    if 'resolution' in kwargs:
        RESOLUTION = kwargs['resolution']
    else:
        Warning('Image resolution not given, distance and velocity results will be in px')
        RESOLUTION = 1.0

    if 'frame_size' in kwargs:
        if kwargs['frame_size'] is not None:
            FRAME_SIZE = kwargs['frame_size']
        else:
            Warning("""Frame size (px) is None, point coordinate will not be 
                     and distance to center calculations will be incorrect""")
            FRAME_SIZE = 1.0
    else:
        Warning("""Frame size (px) not given, point coordinate will not be normalized
                and distance to center calculations will be incorrect""")
        FRAME_SIZE = 1.0

    if 'fps' in kwargs:
        if kwargs['fps'] is not None:
            FPS = kwargs['fps']
        else:
            Warning('FPS is none. Velocity calculations will be computed using dt = 1 s')
            FPS = 1.0
    else:
        Warning('FPS was not given. Velocity calculations will be computed using dt = 1 s')
        FPS = 1.0

    # but are also incorrectly detected to be in (0,0) point in the video
    # thus, it is necessary to remove those
    if mask is None:
        # just find points that are not (0,0) 
        mask = np.all(dataframe[scorer][bodypart][['x', 'y']].values != 0, axis=1)
    else:
        non_zero_rows = np.all(dataframe[scorer][bodypart][['x', 'y']].values != 0, axis=1)
        mask = np.logical_and(mask, non_zero_rows)

    # note for the future: As of now, moving average does not take into account whether the
    # points are from consecutive frames or not, if many datapoints are lost due to incorrect 
    # dlc detection (e.g. point (0,0) but high likelihood). this may be an issue
    # as the moving averages will show considerable movement but the fish really isnt moving it 
    # may be good to implement a moving average for chunks of consecutive frames

    # now, the result will have the same size (number of data points)
    # as the dlc output, but the filtered points will be populated with nans
    n_time_pts = len(dataframe)
    frames = np.arange(n_time_pts)
    times = frames*(1/FPS)
    time_array = np.full((n_time_pts,2), np.nan)
    time_array[mask, 0] = frames[mask]
    time_array[mask, 1] = times[mask]

    coords = np.full(dataframe[scorer][bodypart][['x', 'y']].values.shape, np.nan)
    coords[mask, :] = dataframe[scorer][bodypart][['x', 'y']].values[mask]

    likelihood = np.full(dataframe[scorer][bodypart]['likelihood'].values.shape, np.nan)
    likelihood[mask] = dataframe[scorer][bodypart]['likelihood'].values[mask]

    # CREATE THE SAME STUFF WITH NANS FOR THE RESULTS
    dist = np.full((time_array.shape[0], ), np.nan)
    velocities = np.full((time_array.shape[0], ), np.nan)
    acceleration = np.full((time_array.shape[0], ), np.nan)
    coords_norm = np.full((time_array.shape[0],2), np.nan)
    dist_to_center = np.full((time_array.shape[0], ), np.nan)

    # Clean dist and velocities using the px_tol which are to minimize flickering in in the low end
    # and incorrect detection of keypoints due to artifacts in the frame in the high end
    x_diff = compute_derivative(coords[mask, 0])
    y_diff = compute_derivative(coords[mask, 1])
    dist[mask] = np.sqrt(x_diff**2 + y_diff**2)
    dist[dist < px_tolerances[0]] = 0.0
    dist[dist > px_tolerances[1]] = 0.0
    dist = dist * RESOLUTION

    # calculate velocities and acceleration
    v_x = compute_derivative(coords[mask, 0], x=time_array[mask,1])
    v_y = compute_derivative(coords[mask, 1], x=time_array[mask,1])
    velocities[mask] = np.sqrt(v_x**2 + v_y**2)
    velocities = velocities * RESOLUTION
    acceleration[mask] = compute_derivative(velocities[mask], x=time_array[mask,1])

    # calulate the normalized coordinates with 0,0 being the center
    # of the frame, -1, 1 being edges of the frame
    # which are always approximately the edge of the circular well (+- 1 mm)
    #coords_norm = (coords - (FRAME_SIZE/2))/ (FRAME_SIZE/2)
    coords_norm[mask, :] = (coords[mask, :] - (FRAME_SIZE/2))/ (FRAME_SIZE/2)

    diff_to_center = coords[mask, :] - np.array([(FRAME_SIZE/2), (FRAME_SIZE/2)])
    dist_to_center[mask] = np.sqrt(np.sum(diff_to_center**2, axis=1)) * RESOLUTION

    # define column names
    row_1 = ['Frame',
             'Time',
             bodypart,
             bodypart,
             bodypart,
             bodypart,
             bodypart,
             bodypart,
             bodypart,
             bodypart,
             bodypart]
    row_2 = ['Frame',
             'Time',
             'x',
             'y',
             'x_norm',
             'y_norm',
             'likelihood',
             'dist_travelled',
             'velocity',
             'acceleration',
             'dist_to_center']

    if movavg is not None:

        coords_smooth = np.full(dataframe[scorer][bodypart][['x', 'y']].values.shape, np.nan)
        coords_smooth[mask, :] = dataframe[scorer][bodypart][['x', 'y']].values[mask]

        if (movavg % 2) == 0:
            temp_coords = np.vstack((np.repeat(coords_smooth[mask,:][0].reshape(1,2),
                                               int((movavg-1)/2), axis=0),
                                               coords_smooth[mask,:],
                                     np.repeat(coords_smooth[mask,:][-1].reshape(1,2),
                                               int(movavg/2), axis=0)))
        if (movavg % 2) == 1:
            temp_coords = np.vstack((np.repeat(coords_smooth[mask,:][0].reshape(1,2),
                                               int(movavg/2), axis=0),
                                               coords_smooth[mask,:],
                                     np.repeat(coords_smooth[mask,:][-1].reshape(1,2),
                                               int(movavg/2), axis=0)))

        x_coords = np.convolve(temp_coords[:,0], np.ones(movavg),
                               mode='valid') / movavg
        y_coords = np.convolve(temp_coords[:,1], np.ones(movavg),
                               mode='valid') / movavg
        coords_smooth[mask, :] = np.column_stack((x_coords, y_coords))

        #same block of calculations done with the smoothed coordinates
        dist_smooth =  np.full((time_array.shape[0],), np.nan)
        velocities_smooth =  np.full((time_array.shape[0], ), np.nan)
        acceleration_smooth =  np.full((time_array.shape[0], ), np.nan)
        coords_norm_smooth = np.full((time_array.shape[0], 2), np.nan)
        dist_to_center_smooth =  np.full((time_array.shape[0], ), np.nan)

        x_smooth_diff = compute_derivative(coords_smooth[mask, 0])
        y_smooth_diff = compute_derivative(coords_smooth[mask, 1])
        dist_smooth[mask] = np.sqrt(x_smooth_diff**2 + y_smooth_diff**2)
        dist_smooth[dist_smooth < px_tolerances[0]] = 0.0
        dist_smooth[dist_smooth > px_tolerances[1]] = 0.0
        dist_smooth = dist_smooth * RESOLUTION

        # calculate velocities and acceleration
        v_x = compute_derivative(coords_smooth[mask, 0], x=time_array[mask,1])
        v_y = compute_derivative(coords_smooth[mask, 1], x=time_array[mask,1])
        velocities_smooth[mask] = np.sqrt(v_x**2 + v_y**2)
        velocities_smooth = velocities_smooth * RESOLUTION
        acceleration_smooth[mask] = compute_derivative(velocities_smooth[mask],
                                                       x=time_array[mask,1])

        coords_norm_smooth[mask, :] = (coords_smooth[mask, :] - (FRAME_SIZE/2))/ (FRAME_SIZE/2)
        diff_to_center_smooth = coords_smooth[mask, :] - np.array([(FRAME_SIZE/2), (FRAME_SIZE/2)])
        dist_to_center_smooth[mask] = np.sqrt(np.sum(diff_to_center_smooth**2, axis=1)) * RESOLUTION

        # define column names
        row_1.extend([bodypart]*8)
        row_2.extend(['x_smooth',
                      'y_smooth',
                      'x_norm_smooth',
                      'y_norm_smooth',
                      'dist_travelled_smooth',
                      'velocity_smooth',
                      'acceleration_smooth',
                      'dist_to_center_smooth'])

    results_structure = [row_1,
                         row_2]
    tuples = list(zip(*results_structure))
    results_structure = pd.MultiIndex.from_tuples(tuples)
    df = pd.DataFrame(columns=results_structure)

    df[[('Frame', 'Frame'), ('Time', 'Time')]] = time_array
    df[bodypart,'x'] = coords[:, 0]
    df[bodypart,'y'] = coords[:, 1]
    df[bodypart,'x_norm'] = coords_norm[:,0]
    df[bodypart,'y_norm'] = coords_norm[:,1]

    df[bodypart,'likelihood'] = likelihood
    df[bodypart,'dist_travelled'] = dist
    df[bodypart,'velocity'] = velocities
    df[bodypart,'acceleration'] = acceleration
    df[bodypart,'dist_to_center'] = dist_to_center

    if movavg is not None:
        df[bodypart, 'x_smooth'] = coords_smooth[:, 0]
        df[bodypart, 'y_smooth'] = coords_smooth[:, 1]
        df[bodypart, 'x_norm_smooth'] = coords_norm_smooth[:, 0]
        df[bodypart, 'y_norm_smooth'] = coords_norm_smooth[:, 1]
        df[bodypart, 'dist_travelled_smooth'] = dist_smooth
        df[bodypart, 'velocity_smooth'] = velocities_smooth
        df[bodypart, 'acceleration_smooth'] = acceleration_smooth
        df[bodypart, 'dist_to_center_smooth'] = dist_to_center_smooth

    return df

def get_vector_metrics(dataframe: pd.DataFrame, keypoints_to_vectors : list, mask=None, **kwargs):
    """ Compute all vector-based metrics 

    Args:
        dataframe ([type]): [description]
        keypoints_to_vectors (tuple): two-element tuple with lists of 
            str ontaining the two points to define the animal's heading.
            e.g. (['swim_bladder'], ['L_eye_top', 'R_eye_top', 'L_eye_bottom', 'R_eye_bottom'])
            the first set of keypoints determines the origin of the heading vector, and the second 
            set defines the its 'tip'.

    kwargs:
        'truncate' (int) : calculate vector metrics using fewer tail points by setting truncate 
            value to a number lower than 10 (10 is the total number of tail points). 
        'fps' (int): framerate of the video that originated the dlc result file
        'smooth' (int): window size, in frames (smooth > 3), with which to smooth the tail segment
            angle and heading angle data. Uses the savitzky-golay filter (polyorder=3). 
            This will help smooth the mean tail curvature metrics (mtc, mtc_velocity, mtc_accel)
    Returns:
        pd.DataFrame: full dataframe with the vector metrics results
    """

    ### I think it may make sense to change the output of this function to be a dictionary
    origin, point = keypoints_to_vectors

    scorer = dataframe.columns.get_level_values(0)[0]

    if 'truncate' in kwargs:
        TRUNCATE_TAIL_SEGS = kwargs['truncate']
    else: 
        TRUNCATE_TAIL_SEGS = 10

    if 'fps' in kwargs:
        if kwargs['fps'] is not None:
            FPS = kwargs['fps']
        else: 
            Warning('FPS is None. Velocity calculations will be computed using dt = 1 s')
            FPS = 1.0
    else:
        Warning('FPS was not given. Velocity calculations will be computed using dt = 1 s')
        FPS = 1.0

    if 'smooth' in kwargs:
        SMOOTH = kwargs['smooth']
    else: 
        SMOOTH = None

    N_TIME_PTS = len(dataframe)
    if mask is None:
        mask = np.full((N_TIME_PTS,), 1).astype(bool)

    frames = np.arange(N_TIME_PTS).reshape((-1,1))
    times = frames*(1/FPS)
    time_array = np.hstack([frames[mask],times[mask]]).reshape((-1,2))

    df = compute_base_and_heading(dataframe,
                                  time_array,
                                  origin,
                                  point,
                                  truncate=TRUNCATE_TAIL_SEGS,
                                  smooth=SMOOTH)
    df = compute_turn_angles(df, time_array=time_array)

    # add new columns to df so it later receives the outputs from compute_tail_curvature_metrics()
    results_structure = [[],[]]
    for i in range(1,TRUNCATE_TAIL_SEGS+1):
        results_structure[0].extend(['tail_vector_' + str(i), 'tail_vector_' + str(i)])
    for i in range(1,TRUNCATE_TAIL_SEGS+1):
        results_structure[0].extend(['tail_angle_' + str(i)])

    results_structure[1].extend(['x', 'y'] * TRUNCATE_TAIL_SEGS)
    results_structure[1].extend(['angle'] * TRUNCATE_TAIL_SEGS)
    results_structure[0].extend(['mtc', 'mtc_velocity', 'mtc_accel'])
    results_structure[1].extend(['mtc', 'mtc_velocity', 'mtc_accel'])
    tuples = list(zip(*results_structure))

    for column in tuples:
        df[column] = np.nan

    # prepare inputs for compute_tail_curvature_metrics()
    heading_vector = df.iloc[:,4:6].values
    tail_list = []

    for coord in ['x', 'y']:
        tail_list.append(('swim_bladder', coord))
    for i in range(1,TRUNCATE_TAIL_SEGS+1):
        for coord in ['x', 'y']:
            tail_list.append(('tail_'+str(i), coord))

    mtc_metrics = compute_tail_curvature_metrics(heading_vector,
                                                 time_array,
                                                 dataframe[scorer][tail_list].values,
                                                 truncate=TRUNCATE_TAIL_SEGS,
                                                 smooth=SMOOTH)

    for key, values in mtc_metrics.items():
        df[key] = values

    return df

def get_all_metrics(dataframe,
                    exp_metadata: dict,
                    bodyparts_dict: dict,
                    pcutoff=0.8,
                    mask=None,
                    px_tolerances=(0,1000),
                    return_fraction_lost=False,
                    **kwargs):

    """ Computes all point- and vector-based metrics from a dlc output dataframe

    Args:
        dataframe (pd.DataFrame): _description_
        exp_metadata (_type_): _description_
        bodyparts_dict (_type_): _description_
        pcutoff (float, optional): _description_. Defaults to 0.8.
        mask (np.ndarray, bool, optional): mask to filter particular time interval of the 
            dlc dataframe. Has to have the same length as the dataframe (len(mask)==len(dataframe)).
            Defaults to None.
        px_tolerances (tuple, optional): _description_. Defaults to (0,1000).
        return_fraction_lost (bool, optional): if True, also return the fraction of
            point_metrics datapoints that fell below pcutoff (see get_body_center).
            Only computed when bodyparts_dict['point_metrics'] has more than one
            bodypart; otherwise np.nan. Defaults to False.

    kwargs:
        'point_movavg': window size (in frames) to compue smoothed coordinates using a
             moving average. 
        'vector_smooth': window size, in frames (vector_smooth > 3), with which to smooth the tail 
            segment angle data. Uses the savitzky-golay filter (polyorder=3). 
            This will help smooth the mean tail curvature metrics (mtc, mtc_velocity, mtc_accel)
        'truncate_tail_pts': calculate vector metrics using fewer tail points by trunacting the 
            number of tail points (10 is the total number of tail points). 

    Raises:
        TypeError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    # process kwargs
    if 'point_movavg' in kwargs:
        MOVAVG = kwargs['point_movavg']
    else:
        MOVAVG = None

    if 'vector_smooth' in kwargs:
        SMOOTH = kwargs['vector_smooth']
    else:
        SMOOTH = None
    
    if 'truncate_tail_pts' in kwargs:
        TRUNCATE = kwargs['truncate_tail_pts']
    else:
        TRUNCATE = 10

    # store metadata 
    if isinstance(exp_metadata, dict):
        RESOLUTION = exp_metadata['experiment_specs']['resolution_mmpx']
        FPS = int(exp_metadata['camera_specs']['frame_rate'])
        FRAME_SIZE = int(exp_metadata['camera_specs']['width'])
    else:
        raise TypeError("""exp_metadata not type dict. This new version of the function takes
                        exp_metadata as dictionary.""")

    N_TIME_PTS = len(dataframe)
    if mask is not None:
        if len(mask) != len(dataframe):
            raise ValueError("mask has to have the same length as dataframe")
    if mask is None:
        mask = np.full((N_TIME_PTS,), 1).astype(bool)

    if len(bodyparts_dict['point_metrics']) > 1:
        mean_body_kpoint_df, fraction_lost = get_body_center(dataframe[mask],
                                              bodyparts_dict['point_metrics'],
                                              pcutoff=pcutoff,
                                              return_fraction_lost=True)

        dataframe = pd.concat([dataframe[mask], mean_body_kpoint_df], axis=1)

        point_metrics_df = get_point_metrics(dataframe,
                                             bodypart='mean_body_keypoint',
                                             mask=mask,
                                             px_tolerances=px_tolerances,
                                             resolution=RESOLUTION,
                                             frame_size=FRAME_SIZE,
                                             fps=FPS,
                                             movavg=MOVAVG)

    if len(bodyparts_dict['point_metrics']) == 1:
        # only used here to obtain fraction_lost with the same pcutoff-based
        # logic used for the multi-bodypart case above; its centroid df is unused
        _, fraction_lost = get_body_center(dataframe[mask],
                                           bodyparts_dict['point_metrics'],
                                           pcutoff=pcutoff,
                                           return_fraction_lost=True)

        point_metrics_df = get_point_metrics(dataframe[mask],
                                             bodypart=bodyparts_dict['point_metrics'],
                                             mask=mask,
                                             px_tolerances=px_tolerances,
                                             resolution=RESOLUTION,
                                             frame_size=FRAME_SIZE,
                                             fps=FPS,
                                             movavg=MOVAVG)

    vector_metrics_df = get_vector_metrics(dataframe[mask],
                                           keypoints_to_vectors=bodyparts_dict['vector_metrics'],
                                           fps=FPS,
                                           truncate=TRUNCATE,
                                           smooth=SMOOTH)

    all_metrics_df = pd.concat([point_metrics_df, vector_metrics_df], axis=1)

    # Sort the column index to prevent PerformanceWarning and improve lookup speed
    all_metrics_df.sort_index(axis=1, inplace=True)

    if return_fraction_lost:
        return all_metrics_df, fraction_lost
    return all_metrics_df

def bout_detector(
    all_metrics,
    frame_rate,
    smooth_window=9,
    min_peak_separation=0.05,
    min_peak_prominence=0.01,
    bout_window_start=60,
    bout_window_end=100,
    accel_threshold=5,
    mtc_std_threshold=0.5,
    num_repeats=8,
) -> dict:
    """
    detects the onset and offset of swim bouts using the dataframe output from get_all_metrics()
    Args:
        point_metrics (dict): Dictionary containing point-based kinematic data,
                              including 'distance_mm', 'velocity', and 'acceleration'.
        vector_metrics (dict): Dictionary containing vector-based metrics, including
                               'mean_tail_vel', 'mean_tail_angle', and 'delta_heading'.
        frame_rate (int): The video frame rate in Hz.
        smooth_window (int, optional): The size of the smoothing window. Defaults to 5.
        min_peak_separation (float, optional): Minimum time in seconds between movement peaks. 
            Defaults to 0.05.
        min_peak_prominence (float, optional): Minimun prominence of a peak to be considered 
            a valid bout. Defaults to 0.01.
        bout_window_start (int, optional): Number of frames to look back from a peak to 
            find the start. Defaults to 60.
        bout_window_end (int, optional): Number of frames to look forward from a peak to
            find the end. Defaults to 100.
        accel_threhold (float, optional): body acceleration threshold to find end of the bout. 
            Defaults to 5.
        mtc_std_threshold (float, optional): mean tail curvature standard deviation threshold 
            to find end of the bout. Defaults to 0.5.
        num_repeats (int, optional): Number of consecutive frames of low/zero movement to 
            define start/end. Defaults to 7.


    Returns:
        dict: A dictionary containing various metrics for each detected bout.
    """
    # Get main keypoint name
    keypt_name = all_metrics.columns.get_level_values(0)[5]
    # Grab point metrics
    distance_mm = all_metrics[(keypt_name, 'dist_travelled')].values
    acceleration = all_metrics[(keypt_name, 'acceleration')].values

    # Locate the bouts
    distance_mm_smoothed = quick_smooth(distance_mm, smooth_window)
    acceleration_smoothed = quick_smooth(acceleration, smooth_window)

    peaks, _ = find_peaks(distance_mm_smoothed,
                          distance=min_peak_separation * frame_rate,
                          prominence=min_peak_prominence)

    # Adjust peak locations to account for smoothing window
    locs = peaks + (smooth_window - 1) // 2

    # 3. Filter bouts
    # Remove bouts too close to start/end
    valid_indices = (locs > bout_window_start) & (locs + bout_window_end < len(distance_mm))
    trimmed_locs = locs[valid_indices]

    # Remove bouts with bad tracking (too many NaNs)
    bad_bouts = []
    for i, loc in enumerate(trimmed_locs):
        start_win = loc - bout_window_start
        end_win_start = loc + bout_window_end

        window_before = distance_mm[start_win:loc]
        window_after = distance_mm[loc:end_win_start]

        nan_ratio_before = np.sum(np.isnan(window_before)) / len(window_before)
        nan_ratio_after = np.sum(np.isnan(window_after)) / len(window_after)

        if nan_ratio_before >= 0.5 or nan_ratio_after >= 0.5:
            bad_bouts.append(i)

    trimmed_locs = np.delete(trimmed_locs, bad_bouts)

    # 4. Determine bout start
    bout_start_locs = []
    for loc in trimmed_locs:
        window_of_interest = distance_mm_smoothed[(loc - bout_window_start):loc]
        # To handle NaNs pythonically, we create a temporary version of the
        # window where NaNs are replaced with a value that won't affect logic (e.g., infinity).
        # This avoids modifying the original data with a "magic number".
        window_for_repeats = np.nan_to_num(window_of_interest, nan=np.inf)
        repeat_array = find_repeats(window_for_repeats)

        # Find last sequence of repeats or zero movement
        # The condition for `window_of_interest == 0` is checked on the original data.
        start_candidates = np.where((repeat_array >= num_repeats) & (window_of_interest == 0))[0]
        if start_candidates.size > 0:
            start_temp = (loc - bout_window_start) + np.max(start_candidates)
        else: # If not found, use max change in acceleration
            accel_change = acceleration_smoothed[(loc - bout_window_start):loc]
            max_i = np.argmax(accel_change) if accel_change.size > 0 else 0
            start_temp = (loc - bout_window_start) + max_i
        bout_start_locs.append(start_temp)

    bout_start_locs = np.array(bout_start_locs)

    # Filter out bouts where start is after peak
    valid_indices = (trimmed_locs - bout_start_locs) > 0
    trimmed_locs = trimmed_locs[valid_indices]
    bout_start_locs = bout_start_locs[valid_indices]

    # 5. Determine bout end (distance-based)
    bout_end_locs = []
    for i, loc in enumerate(trimmed_locs):
        # Strategy from `find_end` function
        window_of_interest = distance_mm_smoothed[loc:(loc + bout_window_end)].copy()

        # we divide acceleration by 10k, mtd_std by 100 to put them in the same order of magnitude
        # this is not necessary if you do not want to see the data in the same plot
        window_accel = np.absolute(acceleration_smoothed[loc:(loc + bout_window_end)])/10000
        window_mtc_std = all_metrics[('mtc', 'mtc')][loc:(loc + bout_window_end)] \
            .rolling(smooth_window).std()/100
        window_mtc_std = window_mtc_std.values.reshape((-1,))

        end_found = False
        # Iteratively search for minimum that meets criteria
        for _ in range(int(bout_window_end)):
            min_idx = np.nanargmin(window_of_interest)

            if np.logical_and(window_accel[min_idx] <= accel_threshold/10000,
                              window_mtc_std[min_idx] <= mtc_std_threshold/100):
                end_temp = loc + (min_idx if min_idx >= 5 else 5)
                end_found = True
                break
            else:
                window_of_interest[min_idx] = np.inf # Mark as checked and find next minimum
        if not end_found: # Fallback if no point meets the criteria

            end_temp = loc + np.nanargmin(distance_mm_smoothed[loc:(loc + bout_window_end)])

        bout_end_locs.append(end_temp)
    bout_end_locs = np.array(bout_end_locs)

    # remove or merge overlapped bouts
    bout_metric = process_bout_overlaps({'onset': bout_start_locs, 'offset':bout_end_locs})

    return bout_metric

def compute_single_bout_metrics(all_metrics_df:pd.DataFrame,
                         onset_offset:tuple,
                         FPS=1,
                         smooth_point=None,
                         smooth_vector= None,
                         smooth_vigor_window=43) -> dict:

    """ computes bout metrics of a single bout.

    Args:
        all_metrics_df (pd.DataFrame): frame-by-frame kinematic data (output from get_all_metrics)
        onset_offset (tuple): tuple where first and second elements comprise of bout 
            onset and offset, respectively.
        FPS (int, optional): recording framerate (in Hz). Defaults to 1.
        smooth_point (tuple, optional): parameters to smooth point metrics. First element
            is a string (at the moment only 'mean' is an option), second element is the time window
            in miliseconds to run the moving average with. Defaults to None.
        smooth_vector (tuple, optional): parameters to smooth vector metrics. First element
            is a string (at the moment only 'mean' is an option), second element is the time window
            in miliseconds to run the moving average ('mean') with. Defaults to None.
        smooth_vigor_window (int, optional): time window in miliseconds to run the vigor calculation
            with a moving average. Defaults to 43 ms.

    Returns:
        dict: bout metrics with keys as metrics names, values are their values.
    """

    # Get main keypoint name
    keypt_name = all_metrics_df.columns.get_level_values(0)[5]


    if smooth_point[0] == 'mean':
        window = int(smooth_point[1] / (1000*(1/FPS)))
        dt_values = quick_smooth(all_metrics_df[(keypt_name, 'dist_travelled')].values,
                                  smooth_val=window)
        velocity = quick_smooth(all_metrics_df[(keypt_name, 'velocity')].values,
                                 smooth_val=window)
        acceleration = quick_smooth(all_metrics_df[(keypt_name, 'acceleration')].values,
                                     smooth_val=window)

    if smooth_point is None:
        dt_values = all_metrics_df[(keypt_name, 'dist_travelled')]
        velocity = all_metrics_df[(keypt_name, 'velocity')]
        acceleration = all_metrics_df[(keypt_name, 'acceleration')]

    if smooth_vector[0] == 'mean':
        time_pts = all_metrics_df[('Time', 'Time')].values.reshape((-1,))
        window = int(smooth_vector[1] / (1000*(1/FPS)))
        mtc = all_metrics_df[('mtc', 'mtc')].rolling(window).mean().values.reshape((-1,))
        mtc_velocity = all_metrics_df[('mtc_velocity', 'mtc_velocity')] \
            .rolling(window).mean().values.reshape((-1,))
        turn_angles =  all_metrics_df[('turn_angles', 'turn_angles')] \
            .rolling(window).mean().values.reshape((-1,))
        yaw_speed = compute_derivative(turn_angles, x=time_pts)
        heading = all_metrics_df[('heading_degrees', 'heading_degrees')] \
            .rolling(window).mean().values.reshape((-1,))

    if smooth_vector is None:
        time_pts = all_metrics_df[('Time', 'Time')].values.reshape((-1,))
        mtc = all_metrics_df[('mtc', 'mtc')].values.reshape((-1,))
        mtc_velocity = all_metrics_df[('mtc_velocity', 'mtc_velocity')].values.reshape((-1,))
        turn_angles =  all_metrics_df[('turn_angles', 'turn_angles')].values.reshape((-1,)) #yaw
        yaw_speed = compute_derivative(turn_angles, x=time_pts)
        heading = all_metrics_df[('heading_degrees', 'heading_degrees')].values.reshape((-1,))

    onset = onset_offset[0]
    offset = onset_offset[1]

    # HERE THE COMPUTATIONS START
    # compute bout duration in miliseconds
    duration = all_metrics_df.iloc[offset, 1] - all_metrics_df.iloc[onset, 1]
    duration = duration * 1000

    # whole-body displacement related (point-based metrics)
    total_distance = np.nansum(dt_values[onset:offset])
    mean_speed = total_distance / (duration/1000)
    max_point_speed = np.nanmax(velocity[onset:offset])
    mean_accel = np.nanmean(acceleration[onset:offset])
    max_point_accel = np.nanmax(acceleration[onset:offset])

    # tail related (vector-based)
    mean_mtc = np.nanmean(mtc[onset:offset])
    max_point_mtc = np.nanmax(mtc[onset:offset])
    min_point_mtc = np.nanmin(mtc[onset:offset])
    max_abs_point_mtc = np.nanmax(np.abs(mtc[onset:offset]))
    bout_symmetry = (max_point_mtc + min_point_mtc) / (max_point_mtc - min_point_mtc)

    max_mtc_velocity = np.nanmax(mtc_velocity[onset:offset])
    mean_mtc_velocity = np.nanmean(np.abs(mtc_velocity[onset:offset]))

    tail_beat_fft = fft(detrend(mtc[onset:offset]), n=300)
    frequencies = fftfreq(300, 1/FPS)
    pos_freq_mask = frequencies > 0
    dominant_tailbeat_freq = frequencies[np.argmax(np.abs(tail_beat_fft[pos_freq_mask])**2)]

    # tail segment related (vector-based)
    tail_angle_names = [var for var in all_metrics_df.columns.get_level_values(0) \
                        if 'tail_angle_' in var]
    columns = zip(tail_angle_names, ['angle']*(len(tail_angle_names)))
    tail_max_amplitude = np.max(np.abs(all_metrics_df[columns].values[onset:offset]), axis=0)

    # vigor related
    # vigor degrees/s
    total_vigor = np.sum(np.abs(np.diff(mtc_velocity[onset:offset], prepend=0)))
    # max vigor calculated by finding the maximum instantaneous vigor using a window
    window = int(smooth_vigor_window / (1000*(1/FPS)))
    moving_vigor = quick_smooth(np.abs(np.diff(mtc_velocity[onset:offset])), smooth_val=window)
    max_point_vigor = np.nanmax(moving_vigor)
    mean_vigor = np.nanmean(moving_vigor)

    #heading related (vector-based)
    max_turn_angle = np.max(np.abs(turn_angles[onset:offset]))
    bout_end_angle = np.abs(heading[onset]  \
                            - heading[offset])
    bout_end_angle2 = 360 - bout_end_angle 
    bout_end_angle = np.min(np.abs([bout_end_angle, bout_end_angle2]), axis=0)

    max_yaw_speed = np.nanmax(yaw_speed)
    mean_yaw_speed = np.nanmean(yaw_speed)

    bout_dict = {'onset': onset_offset[0],
                 'offset': onset_offset[1],
                 'onset_s': onset_offset[0]/FPS,
                 'offset_s': onset_offset[1]/FPS}
    
    bout_dict.update({'duration': duration,                         # in miliseconts
                      'total_distance': total_distance,             # in milimeters
                      'mean_speed': mean_speed,                   # in mm.s^-1
                      'max_speed': max_point_speed,                 # in mm.s^-1
                      'mean_acceleration': mean_accel,            # in mm.s^-2
                      'max_acceleration': max_point_accel,          # in mm.s^-2
                      'mean_mtc': mean_mtc,                          # in degrees
                      'max_abs_mtc': max_abs_point_mtc,             # in degrees
                      'bout_symmetry': bout_symmetry,               # dimensionless 
                      'max_mtc_velocity': max_mtc_velocity,         # in degrees.s^-1
                      'neab_mtc_velocity': mean_mtc_velocity,        # in degrees.s^-1
                      'tailbeat_frequency': dominant_tailbeat_freq, # in Hz, dominant tailbeat freq
                      'total_vigor': total_vigor,                   # in degrees.s^-1
                      'max_vigor': max_point_vigor,                 # in degrees.s^-1
                      'mean_vigor': mean_vigor,                      # in degrees.s^-1
                      'mean_yaw_speed': mean_yaw_speed,             # in degrees.s^-1
                      'max_yaw_speed': max_yaw_speed,               # in degrees.s^-1
                      'max_turn_angle': max_turn_angle,             # in degrees
                      'bout_end_angle': bout_end_angle              # in degrees
                      })
    # add a variable for each tail segment max amplitude, in degrees (from absolute values)
    bout_dict.update({'max_tail_amplitude_'+str(i+1) : tail_max_amplitude[i] \
                       for i in range(len(tail_angle_names))})

    return bout_dict

def compute_bout_metrics(bouts_dict : dict, all_metrics_df : pd.DataFrame, FPS=1, **kwargs) -> dict:
    """
        Computes bout metrics of a set of swim bouts. 
    Args:
        bouts_dict (dict): dictionary containing the bout onset and offset indices 
            (output from bout_detector).
        all_metrics_df (pd.DataFrame): frame-by-frame kinematic data (output from get_all_metrics)

    Returns:
        dict: dictionary containing all bout metrics, easily transformed into a pd.Dataframe.
    """

    if 'smooth_point' in kwargs:
        smooth_point = kwargs['smooth_point']
    else:
        smooth_point = None

    if 'smooth_vector' in kwargs:
        smooth_vector = kwargs['smooth_vector']
    else:
        smooth_vector = None

    if 'smooth_vigor_window' in kwargs:
        smooth_vigor_window = kwargs['smooth_vigor_window']
    else:
        smooth_vigor_window = 40

    if 'end_time' in kwargs:
        end_time = kwargs['end_time']
    else:
        end_time = None  

    onsets = bouts_dict['onset']
    offsets = bouts_dict['offset']

    on_and_offsets = list(zip(onsets,offsets))
    bout_metrics = {}

    first_run = True
    for on_and_offset in on_and_offsets:
        single_bout_dict = compute_single_bout_metrics(all_metrics_df,
                                                       on_and_offset,
                                                       FPS=FPS,
                                                       smooth_point=smooth_point,
                                                       smooth_vector=smooth_vector,
                                                       smooth_vigor_window=smooth_vigor_window)
        if not first_run:
            for key, value in single_bout_dict.items():
                bout_metrics[key] = np.append(bout_metrics[key], value)
        if first_run:
            bout_metrics.update({key : np.array([value])
                                 for key, value in single_bout_dict.items()})
            first_run = False

    bout_metrics['IBI'] = compute_ibi(onsets, time_array=all_metrics_df[('Time', 'Time')].values, end_time=end_time)
    bout_metrics['bout_number'] = np.arange(1,len(onsets)+1)
    
    return bout_metrics

def compute_ibi(bout_onsets: list, time_array : np.ndarray, end_time=None) -> np.ndarray:
    """ Computes the interbout interval (IBI, i.e. time elapsed between start of current bout until
        the start of the next bout) for a set of bouts. Output unit of measure is the
        same as time_array units. 

    Args:
        bouts (list): list containing indices of where bouts start. 
        time_array (np.ndarray): time vector used to compute IBI
        end_time (_type_, optional): user-defined end time to calculate IBI of last bout.
            Since IBI is a paired measurement (the IBI of one bout depends on the existence of a 
            following swim bout), you can define a end time so the last bout has an IBI. 
            Defaults to None. None will output np.nan as the IBI of the last bout.
    
    Returns:
        np.ndarray: IBI values for each bout.
    """
    if end_time is not None:
        start_times = time_array[bout_onsets]
        start_times = np.append(start_times, end_time)
        return np.diff(start_times)
    else:
        ibi = np.diff(time_array[bout_onsets])
        ibi = np.append(ibi, np.nan)
        return ibi
