import os
import json

# Define excluded PIDs (as strings to match username format)
excluded_pids_str = os.environ.get("EXCLUDED_PIDS", "")
excluded_pids = {pid.strip() for pid in excluded_pids_str.split(",") if pid.strip()}


def get_pids_with_feature(feature_name, pid_map_data=None, excluded=None):
    """
    Get set of PIDs that have a specific feature enabled.

    Parameters:
    -----------
    feature_name : str
        Name of the feature to check (e.g., 'surveys_enabled', 'seizures_enabled',
        'sleep_enabled', 'spikes_enabled')
    pid_map_data : dict, optional
        The pid_map dictionary. If None, loads from pid_database_map.json
    excluded : set, optional
        Set of PIDs to exclude. If None, uses the global excluded_pids

    Returns:
    --------
    set : Set of PIDs (as strings) that have the feature enabled and are not excluded
    """
    if pid_map_data is None:
        with open("pid_database_map.json", "r") as f:
            pid_map_data = json.load(f)

    if excluded is None:
        excluded = excluded_pids

    eligible_pids = set()
    for db_name, pids in pid_map_data.items():
        for pid, info in pids.items():
            # Check if feature is enabled and PID is not excluded
            if pid not in excluded and info.get(feature_name, False):
                eligible_pids.add(pid)

    return eligible_pids
