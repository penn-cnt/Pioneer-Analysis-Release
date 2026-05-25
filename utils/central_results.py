import os
import json
import ast
import re
import numpy as np
import pandas as pd
import seaborn as sns
from .db_connector import get_db_engine
from .get_pids_with_feature import get_pids_with_feature
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from cycler import cycler
from databricks import sql as databricks_sql
from .databricks_config import (
    get_databricks_connection_params,
    get_databricks_seizure_annotations_table,
    get_databricks_spike_annotations_table,
    get_databricks_sparcnet_annotations_table
)
import pytz

load_dotenv()

def get_excluded_pids():
    excluded_pids_str = os.environ.get("EXCLUDED_PIDS", "")
    return {pid.strip() for pid in excluded_pids_str.split(",") if pid.strip()}

def setup_plotting():
    sns.set_context("talk")
    sns.set_style("whitegrid")
    COLORS = sns.color_palette("Dark2", 12)
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLORS)
    sns.set_palette(COLORS)
    # Set Inter as the default font for all figures
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "DejaVu Sans", "Helvetica", "Arial"]
    return COLORS

COLORS = setup_plotting()

TOPIC_COLOR_MAP = {
    "seizure": COLORS[0],
    "sleep": COLORS[1],
    "spike": COLORS[2],
    "general chat": COLORS[3],
}

PLOTS = {
    "conversations_svg": "plots/conversations_svg",
    "conversations_png": "plots/conversations_png",
    "surveys_svg": "plots/surveys_svg",
    "surveys_png": "plots/surveys_png",
    "usage_svg": "plots/usage_svg",
    "usage_png": "plots/usage_png",
}

def create_plot_dirs():
    for path in PLOTS.values():
        os.makedirs(path, exist_ok=True)

SVG_OUTPUT_DIR = PLOTS["conversations_svg"]
PNG_OUTPUT_DIR = PLOTS["conversations_png"]

def load_data():
    excluded_pids = get_excluded_pids()
    engine = get_db_engine()
    print("Database engine created successfully.")
    
    try:
        users_df = pd.read_sql("SELECT id, username, eeg_type FROM users", engine)
        users_df = users_df[~users_df["username"].isin(excluded_pids)]
        included_user_ids = set(users_df["id"])

        user_messages_df = pd.read_sql(
            "SELECT user_id, content, timestamp, message_type FROM user_messages",
            engine,
        )
        ai_messages_df = pd.read_sql(
            "SELECT user_id, content, timestamp FROM ai_messages",
            engine,
        )
        surveys_df = pd.read_sql("SELECT user_id, is_completed FROM surveys", engine)

        user_messages_df = user_messages_df[user_messages_df["user_id"].isin(included_user_ids)]
        ai_messages_df = ai_messages_df[ai_messages_df["user_id"].isin(included_user_ids)]
        surveys_df = surveys_df[surveys_df["user_id"].isin(included_user_ids)]

        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    with open("pid_database_map.json", "r") as f:
        pid_map = json.load(f)

    pid_info = {}
    for db_name, pids in pid_map.items():
        for pid, dates in pids.items():
            if pid not in excluded_pids:
                pid_info[pid] = dates
                
    return engine, users_df, user_messages_df, ai_messages_df, surveys_df, included_user_ids, pid_map, pid_info

def compute_interaction_volume(engine, included_user_ids=None):
    user_messages = pd.read_sql(
        "SELECT id, user_id, content, timestamp FROM user_messages",
        engine,
    )
    ai_messages = pd.read_sql(
        "SELECT id, user_id, content, timestamp FROM ai_messages",
        engine,
    )
    user_messages["timestamp"] = pd.to_datetime(user_messages["timestamp"], utc=True)
    ai_messages["timestamp"] = pd.to_datetime(ai_messages["timestamp"], utc=True)

    if included_user_ids:
        user_messages = user_messages[user_messages["user_id"].isin(included_user_ids)]
        ai_messages = ai_messages[ai_messages["user_id"].isin(included_user_ids)]

    print(f"[Usage] Total Patient messages: {len(user_messages)}")
    print(f"[Usage] Total AI messages: {len(ai_messages)}")
    return user_messages, ai_messages

def topic_classification_user_queries(engine, included_user_ids=None):
    user_msgs_with_types = pd.read_sql(
        """
        SELECT id, user_id, message_type
        FROM user_messages
        WHERE message_type IS NOT NULL
        """,
        engine,
    )
    if included_user_ids:
        user_msgs_with_types = user_msgs_with_types[
            user_msgs_with_types["user_id"].isin(included_user_ids)
        ]

    type_to_topic = {
        "general_chat": "general chat",
        "follow_up_answer": "general chat",
        "data_agent_seizure": "seizure",
        "data_agent_spike": "spike",
        "data_agent_sleep": "sleep",
    }
    topic_counts = {key: 0 for key in ["seizure", "sleep", "spike", "general chat"]}

    for _, row in user_msgs_with_types.iterrows():
        msg_types = row["message_type"]
        if msg_types is None:
            continue
        if isinstance(msg_types, (list, tuple, np.ndarray)):
            if len(msg_types) == 0:
                continue
        elif isinstance(msg_types, str):
            try:
                if msg_types.startswith("{"):
                    msg_types = re.findall(r'"([^"]*)"', msg_types) or re.findall(r"'([^']*)'", msg_types)
                else:
                    msg_types = ast.literal_eval(msg_types)
            except Exception:
                msg_types = [msg_types] if msg_types else []
        elif pd.isna(msg_types):
            continue
        else:
            msg_types = [msg_types]

        for msg_type in msg_types:
            msg_type_str = str(msg_type).strip()
            if msg_type_str in type_to_topic:
                topic_counts[type_to_topic[msg_type_str]] += 1

    topic_names = ["seizure", "sleep", "spike", "general chat"]
    topics_series = pd.Series(
        [topic_counts.get(name, 0) for name in topic_names],
        index=topic_names,
        name="count",
    )
    print("[Topics] Patient query distribution:")
    print(topic_counts)
    return topics_series

def compute_summary_statistics(users_df, user_messages_df, ai_messages_df, surveys_df, pid_info):
    # Rename users_df 'id' to 'user_id' for merging and 'username' to 'pid'
    summary_df = users_df.rename(columns={"id": "user_id", "username": "pid"})

    # Get message counts
    user_message_counts = user_messages_df.groupby('user_id').size().reset_index(name='messages_sent')
    ai_message_counts = ai_messages_df.groupby('user_id').size().reset_index(name='messages_received')

    # Get survey counts
    surveys_received = surveys_df.groupby('user_id').size().reset_index(name='surveys_received')
    surveys_completed = surveys_df[surveys_df['is_completed'] == True].groupby('user_id').size().reset_index(name='surveys_completed')

    # Merge message and survey counts into the summary table
    summary_df = pd.merge(summary_df, user_message_counts, on='user_id', how='left')
    summary_df = pd.merge(summary_df, ai_message_counts, on='user_id', how='left')
    summary_df = pd.merge(summary_df, surveys_received, on='user_id', how='left')
    summary_df = pd.merge(summary_df, surveys_completed, on='user_id', how='left')

    # Add enroll and discharge dates from pid_info
    summary_df['enroll_date'] = summary_df['pid'].map(lambda x: pid_info.get(x, {}).get('enroll_date'))
    summary_df['discharge_date'] = summary_df['pid'].map(lambda x: pid_info.get(x, {}).get('discharge_date'))

    # Convert dates to datetime objects
    summary_df['enroll_date'] = pd.to_datetime(summary_df['enroll_date'])
    summary_df['discharge_date'] = pd.to_datetime(summary_df['discharge_date'])

    # Calculate participation days
    summary_df['participation_days'] = (summary_df['discharge_date'] - summary_df['enroll_date']).dt.days

    # Fill NaN values with 0 for count columns
    count_cols = ['messages_sent', 'messages_received', 'surveys_received', 'surveys_completed']
    summary_df[count_cols] = summary_df[count_cols].fillna(0).astype(int)

    # Only consider users with participation_days > 0 to avoid division by zero
    valid_participation = summary_df['participation_days'] > 0

    # Average messages per day for user and AI
    summary_df['user_msgs_per_day'] = summary_df['messages_sent'] / summary_df['participation_days'].where(valid_participation, 1)
    summary_df['ai_msgs_per_day'] = summary_df['messages_received'] / summary_df['participation_days'].where(valid_participation, 1)

    # Survey response rate (if surveys_received > 0)
    summary_df['survey_response_rate'] = summary_df.apply(
        lambda row: row['surveys_completed'] / row['surveys_received'] if row['surveys_received'] > 0 else 0,
        axis=1
    )
    
    # Overall survey rate
    total_received = summary_df['surveys_received'].sum()
    total_completed = summary_df['surveys_completed'].sum()
    overall_survey_rate = total_completed / total_received if total_received > 0 else 0.0

    return summary_df, overall_survey_rate

def compute_seizure_statistics(engine, users_df, included_user_ids, summary_df):
    # Get PIDs with seizures_enabled feature
    seizure_enabled_pids = get_pids_with_feature('seizures_enabled')
    valid_users_df = users_df[users_df['username'].isin(seizure_enabled_pids)]
    valid_user_ids = set(valid_users_df['id'])
    
    # Load seizure events, filtering for included users and after exclusion
    # Only count events that were actually sent to patients (notification_sent_at IS NOT NULL)
    seizure_events_df = pd.read_sql(
        """
        SELECT id, user_id 
        FROM seizure_events 
        WHERE user_id = ANY(%(user_ids)s) 
        AND notification_sent_at IS NOT NULL
        """,
        engine,
        params={'user_ids': list(valid_user_ids & set(included_user_ids))}
    )
    
    # Get total count of all detected seizure events (including those not sent) for context
    total_detected_seizure_events = pd.read_sql(
        """
        SELECT COUNT(*) as total_count
        FROM seizure_events 
        WHERE user_id = ANY(%(user_ids)s)
        """,
        engine,
        params={'user_ids': list(valid_user_ids & set(included_user_ids))}
    ).iloc[0]['total_count']
    
    # Get all event IDs from our included users to fetch only relevant annotations
    all_seizure_event_ids = list(seizure_events_df['id'])
    
    # Load annotations for those specific events
    # This ensures seizure_annotations_df is also filtered by included users
    # Excluded seizure annotation IDs (accidental duplicates)
    EXCLUDED_SEIZURE_ANNOTATION_IDS = [39]
    
    if all_seizure_event_ids:
        seizure_annotations_df = pd.read_sql(
            "SELECT id, seizure_event_id, classified_response FROM seizure_annotations WHERE seizure_event_id = ANY(%(seizure_event_ids)s) AND id != ALL(%(excluded_ids)s)",
            engine,
            params={'seizure_event_ids': all_seizure_event_ids, 'excluded_ids': EXCLUDED_SEIZURE_ANNOTATION_IDS}
        )
    else:
        # Create an empty df if there are no seizure events for the included users
        seizure_annotations_df = pd.DataFrame(columns=['seizure_event_id', 'classified_response'])

    clinical_df = valid_users_df[['id', 'username']].rename(columns={'id': 'user_id', 'username': 'pid'})

    # Count total seizure events per user
    seizure_event_counts = seizure_events_df.groupby('user_id').size().reset_index(name='seizure_events_total')

    # Count seizure responses per user
    seizure_annotations_df = pd.merge(
        seizure_annotations_df,
        seizure_events_df.rename(columns={'id': 'seizure_event_id'}),
        on='seizure_event_id',
        how='left'
    )
    seizure_response_counts = seizure_annotations_df.groupby('user_id').size().reset_index(name='seizure_responses')

    # Merge counts into our dedicated clinical DataFrame
    clinical_df = pd.merge(clinical_df, seizure_event_counts, on='user_id', how='left')
    clinical_df = pd.merge(clinical_df, seizure_response_counts, on='user_id', how='left')

    # Clean up and calculate response rate
    clinical_df[['seizure_events_total', 'seizure_responses']] = clinical_df[['seizure_events_total', 'seizure_responses']].fillna(0).astype(int)
    clinical_df['seizure_response_rate'] = (
        clinical_df['seizure_responses'] / clinical_df['seizure_events_total']
    ).fillna(0).round(2)
    
    # Merge with summary_df to get participation days for rate calculations
    # We need to filter summary_df to only include the pids present in clinical_df
    pids_in_clinical_df = clinical_df['pid'].unique()
    filtered_summary_df = summary_df[summary_df['pid'].isin(pids_in_clinical_df)]
    clinical_summary_df = pd.merge(clinical_df, filtered_summary_df[['pid', 'participation_days']], on='pid', how='left')
    
    # Calculate events per day for each participant
    valid_participation_clinical = clinical_summary_df['participation_days'] > 0
    if valid_participation_clinical.sum() > 0:
        clinical_summary_df['seizure_events_per_day'] = clinical_summary_df['seizure_events_total'] / clinical_summary_df['participation_days'].where(valid_participation_clinical, 1)
    
    # Calculate overall rate
    total_seizure_events = clinical_summary_df['seizure_events_total'].sum()
    total_seizure_responses = clinical_summary_df['seizure_responses'].sum()
    overall_rate = total_seizure_responses / total_seizure_events if total_seizure_events > 0 else 0.0
    
    return clinical_summary_df, overall_rate

def compute_spike_statistics(engine, users_df, included_user_ids, summary_df):
    spikes_enabled_pids = get_pids_with_feature('spikes_enabled')
    valid_users_df = users_df[users_df['username'].isin(spikes_enabled_pids)]
    valid_user_ids = set(valid_users_df['id'])
    
    target_user_ids = list(valid_user_ids & set(included_user_ids))
    
    if not target_user_ids:
        return pd.DataFrame(), 0.0

    spike_events_df = pd.read_sql(
        """
        SELECT id, user_id
        FROM spike_events
        WHERE user_id = ANY(%(user_ids)s)
        AND notification_sent_at IS NOT NULL
        """,
        engine,
        params={"user_ids": target_user_ids},
    )
    
    if not spike_events_df.empty:
        event_ids = [int(x) for x in spike_events_df["id"].unique().tolist()]
        spike_annotations_df = pd.read_sql(
            """
            SELECT spike_event_id, user_id
            FROM spike_annotations
            WHERE spike_event_id = ANY(%(event_ids)s)
            """,
            engine,
            params={"event_ids": event_ids},
        )
    else:
        spike_annotations_df = pd.DataFrame(columns=["spike_event_id", "user_id"])

    spike_summary_df = valid_users_df[['id', 'username']].rename(columns={'id': 'user_id', 'username': 'pid'})

    if not spike_events_df.empty:
        spikes_per_user = spike_events_df.groupby("user_id").size().reset_index(name="spike_events_total")
    else:
        spikes_per_user = pd.DataFrame(columns=["user_id", "spike_events_total"])

    if not spike_annotations_df.empty:
        spike_responses_per_user = spike_annotations_df.groupby("user_id").size().reset_index(name="spike_responses")
    else:
        spike_responses_per_user = pd.DataFrame(columns=["user_id", "spike_responses"])

    spike_summary_df = pd.merge(spike_summary_df, spikes_per_user, on="user_id", how="left")
    spike_summary_df = pd.merge(spike_summary_df, spike_responses_per_user, on="user_id", how="left")

    spike_summary_df[["spike_events_total", "spike_responses"]] = spike_summary_df[["spike_events_total", "spike_responses"]].fillna(0).astype(int)
    
    spike_summary_df["spike_response_rate"] = 0.0
    mask = spike_summary_df["spike_events_total"] > 0
    spike_summary_df.loc[mask, "spike_response_rate"] = spike_summary_df.loc[mask, "spike_responses"] / spike_summary_df.loc[mask, "spike_events_total"]

    # Merge participation days
    spike_summary_df = pd.merge(spike_summary_df, summary_df[['pid', 'participation_days']], on='pid', how='left')
    
    spike_summary_df['spike_events_per_day'] = np.nan
    mask_days = spike_summary_df['participation_days'] > 0
    spike_summary_df.loc[mask_days, 'spike_events_per_day'] = spike_summary_df.loc[mask_days, 'spike_events_total'] / spike_summary_df.loc[mask_days, 'participation_days']

    # Overall rate
    total_events = spike_summary_df['spike_events_total'].sum()
    total_responses = spike_summary_df['spike_responses'].sum()
    overall_rate = total_responses / total_events if total_events > 0 else 0.0
    
    return spike_summary_df, overall_rate

def compute_patient_response_statistics(summary_df, seizure_df, spike_df):
    """
    Compute patient response statistics for daily surveys, seizure events, and spike events.
    
    Returns:
        dict: Dictionary with keys 'daily_surveys', 'seizure_events', 'spike_events'
              Each containing 'completed' and 'missed' counts
    """
    # Daily surveys
    total_surveys = summary_df['surveys_received'].sum()
    completed_surveys = summary_df['surveys_completed'].sum()
    missed_surveys = total_surveys - completed_surveys
    
    # Seizure events
    total_seizures = seizure_df['seizure_events_total'].sum() if not seizure_df.empty else 0
    responded_seizures = seizure_df['seizure_responses'].sum() if not seizure_df.empty else 0
    missed_seizures = total_seizures - responded_seizures
    
    # Spike events
    total_spikes = spike_df['spike_events_total'].sum() if not spike_df.empty else 0
    responded_spikes = spike_df['spike_responses'].sum() if not spike_df.empty else 0
    missed_spikes = total_spikes - responded_spikes
    
    return {
        'daily_surveys': {
            'completed': completed_surveys,
            'missed': missed_surveys,
            'total': total_surveys
        },
        'seizure_events': {
            'completed': responded_seizures,
            'missed': missed_seizures,
            'total': total_seizures
        },
        'spike_events': {
            'completed': responded_spikes,
            'missed': missed_spikes,
            'total': total_spikes
        }
    }


def compute_survey_response_by_time_of_day(engine, included_user_ids):
    """
    Compute survey response rates broken down by time of day (morning, afternoon, evening).
    
    Returns:
        dict: Dictionary with keys 'morning', 'afternoon', 'evening'
              Each containing 'completed', 'missed', 'total' counts
    """
    # Load surveys - survey_type already contains morning/afternoon/evening
    surveys_df = pd.read_sql(
        """
        SELECT user_id, survey_type, is_completed
        FROM surveys
        WHERE user_id = ANY(%(user_ids)s)
        """,
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    
    # Group by survey_type (which is already morning/afternoon/evening)
    result = {}
    for period in ['morning', 'afternoon', 'evening']:
        period_surveys = surveys_df[surveys_df['survey_type'] == period]
        total = len(period_surveys)
        completed = len(period_surveys[period_surveys['is_completed'] == True])
        missed = total - completed
        
        result[period] = {
            'completed': completed,
            'missed': missed,
            'total': total
        }
    
    return result


def compute_event_verification_stats(engine, included_user_ids):
    """
    Compute verification/response statistics for seizure and spike events.
    Categories: Yes, Unsure, No, No Response
    
    Returns:
        dict: Dictionary with keys 'seizure_events' and 'spike_events'
              Each containing counts for each response category
    """
    result = {}
    
    # Seizure Events
    seizure_events_df = pd.read_sql(
        """
        SELECT id, user_id
        FROM seizure_events
        WHERE user_id = ANY(%(user_ids)s)
        AND notification_sent_at IS NOT NULL
        """,
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    
    # Excluded seizure annotation IDs (accidental duplicates)
    EXCLUDED_SEIZURE_ANNOTATION_IDS = [39]
    
    if not seizure_events_df.empty:
        event_ids = list(seizure_events_df['id'])
        seizure_annotations_df = pd.read_sql(
            """
            SELECT id, seizure_event_id, classified_response
            FROM seizure_annotations
            WHERE seizure_event_id = ANY(%(event_ids)s)
            AND id != ALL(%(excluded_ids)s)
            """,
            engine,
            params={'event_ids': event_ids, 'excluded_ids': EXCLUDED_SEIZURE_ANNOTATION_IDS}
        )
    else:
        seizure_annotations_df = pd.DataFrame(columns=['seizure_event_id', 'classified_response'])
    
    # Count responses by category
    total_seizures = len(seizure_events_df)
    responded_seizures = len(seizure_annotations_df)
    no_response_seizures = total_seizures - responded_seizures
    
    # Group by response type
    yes_count = len(seizure_annotations_df[seizure_annotations_df['classified_response'] == 'yes'])
    uncertain_count = len(seizure_annotations_df[seizure_annotations_df['classified_response'] == 'uncertain'])
    no_count = len(seizure_annotations_df[seizure_annotations_df['classified_response'] == 'no'])
    
    result['seizure_events'] = {
        'yes': yes_count,
        'unsure': uncertain_count,  # Map 'uncertain' from DB to 'unsure' for display
        'no': no_count,
        'no_response': no_response_seizures,
        'total': total_seizures
    }
    
    # Spike Events
    spike_events_df = pd.read_sql(
        """
        SELECT id, user_id
        FROM spike_events
        WHERE user_id = ANY(%(user_ids)s)
        AND notification_sent_at IS NOT NULL
        """,
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    
    if not spike_events_df.empty:
        event_ids = list(spike_events_df['id'])
        spike_annotations_df = pd.read_sql(
            """
            SELECT spike_event_id, raw_response
            FROM spike_annotations
            WHERE spike_event_id = ANY(%(event_ids)s)
            """,
            engine,
            params={'event_ids': event_ids}
        )
    else:
        spike_annotations_df = pd.DataFrame(columns=['spike_event_id', 'raw_response'])
    
    # Classify responses based on raw_response text
    def classify_spike_response(raw_text):
        if pd.isna(raw_text):
            return 'no_response'
        
        text_lower = str(raw_text).lower()
        
        # Check for "unsure" or "uncertain" responses
        if any(word in text_lower for word in ['unsure', 'uncertain', "don't know", "not sure", "maybe"]):
            return 'unsure'
        # Check for "no" responses
        elif any(phrase in text_lower for phrase in ['no,', 'no.', 'did not', 'i did not', 'nope', 'negative']):
            return 'no'
        # Check for "yes" responses
        elif any(word in text_lower for word in ['yes', 'yeah', 'yep', 'correct', 'i did', 'affirmative']):
            return 'yes'
        else:
            # Default to unsure if we can't clearly classify
            return 'unsure'
    
    spike_annotations_df['classified'] = spike_annotations_df['raw_response'].apply(classify_spike_response)
    
    # Count responses
    total_spikes = len(spike_events_df)
    no_response_spikes = total_spikes - len(spike_annotations_df)
    
    # Group by response type
    yes_count_spike = len(spike_annotations_df[spike_annotations_df['classified'] == 'yes'])
    unsure_count_spike = len(spike_annotations_df[spike_annotations_df['classified'] == 'unsure'])
    no_count_spike = len(spike_annotations_df[spike_annotations_df['classified'] == 'no'])
    
    result['spike_events'] = {
        'yes': yes_count_spike,
        'unsure': unsure_count_spike,
        'no': no_count_spike,
        'no_response': no_response_spikes,
        'total': total_spikes
    }
    
    return result


def compute_data_agent_statistics(engine, included_user_ids):
    """
    Compute comprehensive statistics for data agent interactions.
    
    Analyzes:
    - User requests to the data agent (by topic: seizure, spike, sleep)
    - Data agent responses
    - Response patterns and accuracy (based on review status if available)
    - Response times
    
    Returns:
        dict: Dictionary with keys:
            - 'user_requests': breakdown of user requests by topic
            - 'ai_responses': data agent response statistics
            - 'request_response_pairs': matched request-response analysis
            - 'response_times': response latency statistics
            - 'review_stats': review/accuracy statistics if available
    """
    result = {}
    
    # 1. USER REQUESTS TO DATA AGENT (by topic)
    user_msgs_df = pd.read_sql(
        """
        SELECT id, user_id, content, message_type, timestamp
        FROM user_messages
        WHERE user_id = ANY(%(user_ids)s)
        AND message_type IS NOT NULL
        """,
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    user_msgs_df['timestamp'] = pd.to_datetime(user_msgs_df['timestamp'])
    
    # Parse message_type to identify ALL data agent topics in a message
    def extract_all_data_agent_topics(msg_type):
        """Extract ALL data agent topics from a message (returns list of topics)"""
        # Handle None
        if msg_type is None:
            return []
        
        # Handle numpy arrays and lists directly
        if isinstance(msg_type, (list, tuple, np.ndarray)):
            items = list(msg_type)
        elif isinstance(msg_type, str):
            # Check if it's a scalar NA
            try:
                if pd.isna(msg_type):
                    return []
            except (ValueError, TypeError):
                pass
            
            msg_type_str = str(msg_type)
            
            # Handle array-like strings
            if msg_type_str.startswith(('{', '[')):
                try:
                    if msg_type_str.startswith('{'):
                        items = re.findall(r'"([^"]*)"', msg_type_str) or re.findall(r"'([^']*)'", msg_type_str)
                        # Also handle unquoted items like {data_agent_seizure,data_agent_spike}
                        if not items:
                            items = [x.strip() for x in msg_type_str.strip('{}[]').split(',')]
                    else:
                        items = ast.literal_eval(msg_type_str)
                except:
                    items = [msg_type_str]
            else:
                items = [msg_type_str]
        else:
            # Try to check if it's a scalar NA
            try:
                if pd.isna(msg_type):
                    return []
            except (ValueError, TypeError):
                pass
            items = [str(msg_type)]
        
        # Extract ALL data agent topics (not just the first one)
        topics = []
        for item in items:
            item_str = str(item).strip()
            if item_str == 'data_agent_seizure':
                topics.append('seizure')
            elif item_str == 'data_agent_spike':
                topics.append('spike')
            elif item_str == 'data_agent_sleep':
                topics.append('sleep')
        
        return topics if topics else None
    
    # Extract all topics for each message
    user_msgs_df['data_agent_topics'] = user_msgs_df['message_type'].apply(extract_all_data_agent_topics)
    
    # Filter to only messages with data agent topics
    data_agent_messages = user_msgs_df[user_msgs_df['data_agent_topics'].notna()].copy()
    
    # Explode the topics so each topic gets its own row for counting
    data_agent_requests_exploded = data_agent_messages.explode('data_agent_topics')
    
    # Count by topic (this now counts ALL topics, even if multiple in one message)
    request_counts = data_agent_requests_exploded['data_agent_topics'].value_counts().to_dict()
    total_requests = len(data_agent_requests_exploded)
    
    # For the 'by_user' count, we'll count unique messages per user (not exploded)
    data_agent_requests = data_agent_messages.copy()  # Keep unexploded version for other uses
    
    result['user_requests'] = {
        'by_topic': request_counts,
        'total': total_requests,
        'by_user': data_agent_requests.groupby('user_id').size().to_dict()
    }
    
    # 2. DATA AGENT RESPONSES
    ai_msgs_df = pd.read_sql(
        """
        SELECT id, user_id, content, message_type, message_info, timestamp,
               is_reviewed, review_status, is_flagged
        FROM ai_messages
        WHERE user_id = ANY(%(user_ids)s)
        AND message_type = 'data_agent'
        """,
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    ai_msgs_df['timestamp'] = pd.to_datetime(ai_msgs_df['timestamp'])
    
    total_responses = len(ai_msgs_df)
    
    # Parse message_info to get topic/details
    def extract_response_topic(msg_info):
        if msg_info is None or pd.isna(msg_info):
            return 'unknown'
        
        msg_info_str = str(msg_info).lower()
        if 'seizure' in msg_info_str:
            return 'seizure'
        elif 'spike' in msg_info_str:
            return 'spike'
        elif 'sleep' in msg_info_str:
            return 'sleep'
        return 'unknown'
    
    ai_msgs_df['response_topic'] = ai_msgs_df['message_info'].apply(extract_response_topic)
    response_by_topic = ai_msgs_df['response_topic'].value_counts().to_dict()
    
    result['ai_responses'] = {
        'total': total_responses,
        'by_topic': response_by_topic,
        'by_user': ai_msgs_df.groupby('user_id').size().to_dict()
    }
    
    # 3. REVIEW/ACCURACY STATISTICS
    reviewed_count = ai_msgs_df['is_reviewed'].sum() if 'is_reviewed' in ai_msgs_df.columns else 0
    flagged_count = ai_msgs_df['is_flagged'].sum() if 'is_flagged' in ai_msgs_df.columns else 0
    
    # Review status breakdown
    if 'review_status' in ai_msgs_df.columns:
        review_status_counts = ai_msgs_df['review_status'].value_counts().to_dict()
        confirmed_count = review_status_counts.get('confirmed', 0)
        rejected_count = review_status_counts.get('rejected', 0)
    else:
        review_status_counts = {}
        confirmed_count = 0
        rejected_count = 0
    
    # Calculate accuracy rate (based on reviewed messages)
    if reviewed_count > 0:
        accuracy_rate = confirmed_count / reviewed_count
    else:
        accuracy_rate = None  # No reviewed messages yet
    
    result['review_stats'] = {
        'total_reviewed': int(reviewed_count),
        'confirmed': confirmed_count,
        'rejected': rejected_count,
        'not_reviewed': total_responses - int(reviewed_count),
        'flagged': int(flagged_count),
        'accuracy_rate': accuracy_rate,
        'review_status_breakdown': review_status_counts
    }
    
    # 4. REQUEST-RESPONSE PAIRING ANALYSIS
    # For each user, match data agent requests with subsequent responses
    # Use the exploded version for pairing since we want each topic counted
    matched_pairs = []
    unmatched_requests = 0
    
    # Work with exploded version for pairing
    for user_id in data_agent_requests_exploded['user_id'].unique():
        user_requests = data_agent_requests_exploded[data_agent_requests_exploded['user_id'] == user_id].sort_values('timestamp')
        user_responses = ai_msgs_df[ai_msgs_df['user_id'] == user_id].sort_values('timestamp')
        
        response_idx = 0
        for _, req_row in user_requests.iterrows():
            req_time = req_row['timestamp']
            req_topic = req_row['data_agent_topics']  # This is now a single topic from the exploded df
            
            # Find the next response after this request
            matched = False
            while response_idx < len(user_responses):
                resp_row = user_responses.iloc[response_idx]
                resp_time = resp_row['timestamp']
                
                if resp_time > req_time:
                    # Found matching response
                    response_time_seconds = (resp_time - req_time).total_seconds()
                    matched_pairs.append({
                        'user_id': user_id,
                        'request_topic': req_topic,
                        'response_topic': resp_row['response_topic'],
                        'response_time_seconds': response_time_seconds,
                        'is_reviewed': resp_row.get('is_reviewed', False),
                        'review_status': resp_row.get('review_status', None),
                        'is_flagged': resp_row.get('is_flagged', False)
                    })
                    response_idx += 1
                    matched = True
                    break
                response_idx += 1
            
            if not matched:
                unmatched_requests += 1
    
    matched_df = pd.DataFrame(matched_pairs) if matched_pairs else pd.DataFrame()
    
    result['request_response_pairs'] = {
        'total_matched': len(matched_pairs),
        'unmatched_requests': unmatched_requests
    }
    
    # 5. RESPONSE TIME STATISTICS
    if not matched_df.empty:
        response_times = matched_df['response_time_seconds']
        result['response_times'] = {
            'mean_seconds': response_times.mean(),
            'median_seconds': response_times.median(),
            'min_seconds': response_times.min(),
            'max_seconds': response_times.max(),
            'std_seconds': response_times.std(),
            'by_topic': matched_df.groupby('request_topic')['response_time_seconds'].agg(['mean', 'median', 'count']).to_dict()
        }
    else:
        result['response_times'] = {
            'mean_seconds': None,
            'median_seconds': None,
            'min_seconds': None,
            'max_seconds': None,
            'std_seconds': None,
            'by_topic': {}
        }
    
    # Store the matched DataFrame for potential further analysis
    result['matched_df'] = matched_df
    
    return result

def compute_system_response_latency_by_type(engine, included_user_ids):
    """
    Compute system (AI) response latencies categorized by message type.
    
    Categorizes AI messages into:
    - General Chat: normal messages (excluding follow_up_intro and initial greetings)
    - Data Agent: data_agent message type
    
    For follow_up_1 messages, response time is calculated from survey completion time.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['category', 'response_time_seconds']
    """
    import pandas as pd
    
    # Load user messages
    user_msgs_df = pd.read_sql(
        "SELECT user_id, timestamp FROM user_messages WHERE user_id = ANY(%(user_ids)s)",
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    user_msgs_df['sender'] = 'user'
    
    # Load AI messages with type and info for categorization
    ai_msgs_df = pd.read_sql(
        "SELECT user_id, timestamp, message_type, message_info, content FROM ai_messages WHERE user_id = ANY(%(user_ids)s)",
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    ai_msgs_df['sender'] = 'ai'
    
    # Load survey completion data for follow-up timing
    surveys_df = pd.read_sql(
        "SELECT user_id, created_at, completed_at FROM surveys WHERE user_id = ANY(%(user_ids)s) AND is_completed = true",
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    surveys_df['completed_at'] = pd.to_datetime(surveys_df['completed_at'])
    
    # Categorize AI messages
    def categorize_message(row):
        # Data Agent messages
        if row['message_type'] == 'data_agent':
            return 'Data Agent'
        
        # General Chat: normal messages, excluding specific automated messages
        if row['message_type'] == 'normal':
            # Exclude follow_up_intro and follow_up_expiry messages (automated survey follow-up messages)
            if pd.notna(row['message_info']):
                msg_info = str(row['message_info'])
                if 'follow_up_intro' in msg_info or 'follow_up_expiry' in msg_info:
                    return None
            
            # Content-based exclusions only
            if pd.notna(row['content']):
                content = str(row['content'])
                # Exclude initial greeting
                if content.startswith("Hi there! I'm Pioneer-AI"):
                    return None
                # Exclude these two specific automated sleep report messages
                if "Can you tell me about anything that affected how well you slept last night?" in content:
                    return None
                if "I'll send this sleep report each day after the completion of your morning survey" in content:
                    return None
            return 'General Chat'
        
        # Include general_chat message type
        if row['message_type'] == 'general_chat':
            return 'General Chat'
            
        return None
    
    ai_msgs_df['category'] = ai_msgs_df.apply(categorize_message, axis=1)
    
    # Filter to only categorized messages
    ai_msgs_categorized = ai_msgs_df[ai_msgs_df['category'].notna()].copy()
    
    # Create conversation timeline
    conversation_df = pd.concat([
        user_msgs_df[['user_id', 'timestamp', 'sender']],
        ai_msgs_categorized[['user_id', 'timestamp', 'sender', 'category', 'message_info']]
    ])
    
    # Convert timestamp and sort
    conversation_df['timestamp'] = pd.to_datetime(conversation_df['timestamp'])
    conversation_df.sort_values(by=['user_id', 'timestamp'], inplace=True)
    
    # Calculate time difference between consecutive messages
    conversation_df['time_diff_seconds'] = (
        conversation_df.groupby('user_id')['timestamp']
        .diff()
        .dt.total_seconds()
    )
    
    # Identify previous sender
    conversation_df['previous_sender'] = (
        conversation_df.groupby('user_id')['sender'].shift(1)
    )
    
    # For follow_up_1 messages, calculate time from survey completion instead
    def calculate_response_time(row):
        # Only process AI messages
        if row['sender'] != 'ai' or pd.isna(row['category']):
            return row['time_diff_seconds']
        
        # Check if this is a follow_up_1 message
        if pd.notna(row['message_info']) and 'follow_up_1' in str(row['message_info']):
            # Find the most recent completed survey before this message
            user_surveys = surveys_df[surveys_df['user_id'] == row['user_id']]
            recent_surveys = user_surveys[user_surveys['completed_at'] < row['timestamp']]
            
            if len(recent_surveys) > 0:
                # Get the most recent survey completion
                most_recent_survey = recent_surveys.loc[recent_surveys['completed_at'].idxmax()]
                # Calculate time from survey completion to AI message
                time_diff = (row['timestamp'] - most_recent_survey['completed_at']).total_seconds()
                return time_diff
        
        # For all other messages, use the standard time difference
        return row['time_diff_seconds']
    
    conversation_df['adjusted_time_diff'] = conversation_df.apply(calculate_response_time, axis=1)
    
    # AI responses: current sender is AI, previous sender was user (or it's a follow_up)
    ai_responses = conversation_df[
        (conversation_df['sender'] == 'ai') & 
        ((conversation_df['previous_sender'] == 'user') | 
         (conversation_df['message_info'].notna() & conversation_df['message_info'].str.contains('follow_up_1', na=False)))
    ].copy()
    
    # Return categorized response times
    result_df = ai_responses[['category', 'adjusted_time_diff']].copy()
    result_df.columns = ['category', 'response_time_seconds']
    result_df = result_df.dropna()
    
    return result_df

def compute_user_response_latency_by_category(engine, included_user_ids):
    """
    Compute user response latencies categorized by message/event type.
    
    Categories:
    - General Chat: User responding to normal AI messages (includes follow-up responses)
    - Surveys: User starting scheduled surveys  
    - Seizure Events: User responding to seizure notifications
    - Spike Events: User responding to spike notifications
    
    Returns:
        tuple: (included_df, excluded_df)
            - included_df: DataFrame with columns ['category', 'response_time_seconds', 'responded', 'source', 'details']
            - excluded_df: DataFrame with columns ['exclusion_reason', 'source', 'details']
    """
    all_response_times = []
    all_excluded = []
    
    # Load users to map user_id to username (needed for 001/002 special handling)
    users_df = pd.read_sql(
        """
        SELECT id, username 
        FROM users 
        WHERE id = ANY(%(user_ids)s)
        """,
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    user_id_to_username = dict(zip(users_df['id'], users_df['username']))
    
    # 1. GENERAL CHAT: User responding to AI messages
    
    # Load AI messages (what AI sent to user)
    ai_msgs_df = pd.read_sql(
        """
        SELECT id, user_id, timestamp, message_type, message_info, content 
        FROM ai_messages 
        WHERE user_id = ANY(%(user_ids)s)
        """,
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    ai_msgs_df['timestamp'] = pd.to_datetime(ai_msgs_df['timestamp'])
    
    # Load user messages (what user sent)
    user_msgs_df = pd.read_sql(
        """
        SELECT user_id, timestamp 
        FROM user_messages 
        WHERE user_id = ANY(%(user_ids)s)
        """,
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    user_msgs_df['timestamp'] = pd.to_datetime(user_msgs_df['timestamp'])
    
    # Categorize and filter AI messages with exclusion tracking
    def categorize_ai_message(row):
        """Determine category or exclusion reason for AI message"""
        # Scheduled surveys - handled separately in surveys table EXCEPT for patients 001 and 002
        if row['message_type'] == 'scheduled':
            username = user_id_to_username.get(row['user_id'], '')
            if username in ['001', '002']:
                # For patients 001 and 002, scheduled messages should be included as general_chat
                return 'INCLUDE_general_chat'
            else:
                # For other patients, exclude scheduled (they're in surveys table)
                return 'EXCLUDE_scheduled_survey'
        
        # Check message_info first to handle follow-ups
        if row['message_type'] == 'normal' and pd.notna(row['message_info']):
            msg_info = str(row['message_info'])
            # Only exclude follow_up_intro and follow_up_expiry
            if 'follow_up_intro' in msg_info:
                return 'EXCLUDE_followup_intro'
            if 'follow_up_expiry' in msg_info:
                return 'EXCLUDE_followup_expiry'
            # follow_up_1/2/3 should be INCLUDED as general_chat (even if they're sleep-related)
            # So we mark them explicitly and skip content-based exclusions
            if any(x in msg_info for x in ['follow_up_1', 'follow_up_2', 'follow_up_3']):
                return 'INCLUDE_general_chat'  # Include these regardless of content
        
        # Seizure/spike detection - handled separately in events tables
        if row['message_type'] == 'seizure_detection':
            return 'EXCLUDE_seizure_detection'
        if row['message_type'] == 'spike_detection':
            return 'EXCLUDE_spike_detection'
        
        # Content-based exclusions (only after we've handled follow-ups)
        if pd.notna(row['content']):
            content = str(row['content'])
            
            # Initial greetings
            if content.startswith("Hi there! I'm Pioneer-AI"):
                return 'EXCLUDE_initial_greeting'
            if content.startswith("Hello! I'm your personal AI assistant"):
                return 'EXCLUDE_initial_greeting'
            if content.startswith("Hello! I'm Pioneer-AI"):
                return 'EXCLUDE_initial_greeting'
            
            # Thank you / acknowledgment messages
            if content.startswith("Thanks for answering this follow-up question!"):
                return 'EXCLUDE_acknowledgment'
            if content.startswith("Thanks for answering these follow-up questions!"):
                return 'EXCLUDE_acknowledgment'
            if content.startswith("Thank you for letting me know. I've recorded that you did not experience a seizure"):
                return 'EXCLUDE_acknowledgment'
            if content.startswith("Thank you for letting me know. I've recorded that you are unsure if you experienced a seizure"):
                return 'EXCLUDE_acknowledgment'
            if content.startswith("Thank you for letting me know. I've recorded that you experienced a seizure"):
                return 'EXCLUDE_acknowledgment'
            if content.startswith("Thank you for letting me know, this information helps improve our detection system"):
                return 'EXCLUDE_acknowledgment'
            if content.startswith("Thank you for letting me know!"):
                return 'EXCLUDE_acknowledgment'
            
            # Intro/informational messages
            if content.startswith("I'm also able to access your monitoring data"):
                return 'EXCLUDE_informational'
            if content.startswith("If there are any features"):
                return 'EXCLUDE_informational'
            if content.startswith("Finally, toward the end of your stay"):
                return 'EXCLUDE_informational'
            if content.startswith("Finally, at the end of your stay, please remember to complete the exit survey"):
                return 'EXCLUDE_informational'
            
            # Sleep report content exclusions
            if "Can you tell me about anything that affected how well you slept last night?" in content:
                return 'EXCLUDE_sleep_report'
            if "I'll send this sleep report each day after the completion of your morning survey" in content:
                return 'EXCLUDE_sleep_report'
        
        # Exclude sleep reports by type
        if row['message_type'] == 'sleep_report':
            return 'EXCLUDE_sleep_report'
        
        # Exclude data agent
        if row['message_type'] == 'data_agent':
            return 'EXCLUDE_data_agent'
        
        # Include general_chat, normal, seizure_response_confirmation, spike_response_confirmation
        if row['message_type'] in ['general_chat', 'normal', 'seizure_response_confirmation', 'spike_response_confirmation']:
            return 'INCLUDE_general_chat'
        
        return 'EXCLUDE_other'
    
    ai_msgs_df['categorization'] = ai_msgs_df.apply(categorize_ai_message, axis=1)
    
    # Track excluded AI messages
    excluded_ai = ai_msgs_df[ai_msgs_df['categorization'].str.startswith('EXCLUDE')].copy()
    for _, msg in excluded_ai.iterrows():
        all_excluded.append({
            'message_id': msg['id'],
            'exclusion_reason': msg['categorization'].replace('EXCLUDE_', ''),
            'source': 'ai_messages',
            'message_type': msg['message_type'],
            'message_info': msg['message_info'],
            'timestamp': msg['timestamp'],
            'content_preview': str(msg['content'])[:100] if pd.notna(msg['content']) else None
        })
    
    
    # GLOBAL TRACKING: Prevent double-counting user responses
    # Track which user message timestamps have been consumed as responses
    # This prevents the same user message from being counted multiple times
    consumed_user_msg_timestamps = {}  # {user_id: set of consumed timestamps}
    
    # Initialize for each user
    for user_id in user_msgs_df['user_id'].unique():
        consumed_user_msg_timestamps[user_id] = set()
    
    
    # 1. FOLLOW-UPS: User answering survey follow-up questions (part of general_chat)
    # NOTE: Process these FIRST so they can claim their response timestamps
    
    follow_ups_df = pd.read_sql(
        """
        SELECT sf.id, sf.survey_id, sf.sent_at, sf.answered_at, sf.is_answered, sf.follow_up_text, s.user_id
        FROM survey_follow_ups sf
        JOIN surveys s ON sf.survey_id = s.id
        WHERE s.user_id = ANY(%(user_ids)s)
        """,
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    follow_ups_df['sent_at'] = pd.to_datetime(follow_ups_df['sent_at'])
    follow_ups_df['answered_at'] = pd.to_datetime(follow_ups_df['answered_at'])
    
    for _, followup in follow_ups_df.iterrows():
        if followup['is_answered'] and pd.notna(followup['answered_at']):
            # User answered the follow-up
            user_id = followup['user_id']
            response_timestamp = followup['answered_at']
            
            response_time = (response_timestamp - followup['sent_at']).total_seconds()
            all_response_times.append({
                'message_id': followup['id'],
                'category': 'general_chat',
                'response_time_seconds': response_time,
                'response_time_minutes': response_time / 60,
                'responded': True,
                'sent_timestamp': followup['sent_at'],
                'response_timestamp': response_timestamp,
                'source': 'survey_follow_ups',
                'message_content': str(followup['follow_up_text'])[:200] if pd.notna(followup['follow_up_text']) else 'Follow-up question',
                'details': f"survey_id={followup['survey_id']}"
            })
            
            # Mark this timestamp as consumed to prevent double-counting in general chat
            if user_id in consumed_user_msg_timestamps:
                consumed_user_msg_timestamps[user_id].add(response_timestamp)
        else:
            # User didn't answer the follow-up
            all_response_times.append({
                'message_id': followup['id'],
                'category': 'general_chat',
                'response_time_seconds': np.nan,
                'response_time_minutes': np.nan,
                'responded': False,
                'sent_timestamp': followup['sent_at'],
                'response_timestamp': None,
                'source': 'survey_follow_ups',
                'message_content': str(followup['follow_up_text'])[:200] if pd.notna(followup['follow_up_text']) else 'Follow-up question',
                'details': f"survey_id={followup['survey_id']}"
            })
    
    # 2. GENERAL CHAT: User responding to AI messages
    # NOTE: Processed AFTER follow-ups so it skips already-consumed responses
    
    # Process included general chat messages
    # Strategy: Process chronologically and mark user messages as "consumed"
    # Each user message can only be a response to ONE AI message (the most recent before it)
    general_chat_ai_msgs = ai_msgs_df[ai_msgs_df['categorization'] == 'INCLUDE_general_chat'].copy()
    
    # Sort AI messages by timestamp
    general_chat_ai_msgs_sorted = general_chat_ai_msgs.sort_values('timestamp')
    
    # For each AI message (in chronological order)
    for _, ai_msg in general_chat_ai_msgs_sorted.iterrows():
        user_id = ai_msg['user_id']
        ai_timestamp = ai_msg['timestamp']
        
        # Find the first unconsumed user message after this AI message
        user_responses = user_msgs_df[
            (user_msgs_df['user_id'] == user_id) & 
            (user_msgs_df['timestamp'] > ai_timestamp)
        ].sort_values('timestamp')
        
        response_found = False
        for _, user_msg in user_responses.iterrows():
            user_timestamp = user_msg['timestamp']
            # Check if this user message has already been consumed
            if user_timestamp not in consumed_user_msg_timestamps[user_id]:
                # This is the first unconsumed user message after the AI message
                response_time = (user_timestamp - ai_timestamp).total_seconds()
                all_response_times.append({
                    'message_id': ai_msg['id'],
                    'category': 'general_chat',
                    'response_time_seconds': response_time,
                    'response_time_minutes': response_time / 60,
                    'responded': True,
                    'sent_timestamp': ai_timestamp,
                    'response_timestamp': user_timestamp,
                    'source': 'ai_messages',
                    'message_content': str(ai_msg['content'])[:200] if pd.notna(ai_msg['content']) else None,
                    'details': f"msg_type={ai_msg['message_type']}, msg_info={ai_msg['message_info']}"
                })
                # Mark this user message as consumed
                consumed_user_msg_timestamps[user_id].add(user_timestamp)
                response_found = True
                break
        
        if not response_found:
            # No unconsumed user message found after this AI message
            all_response_times.append({
                'message_id': ai_msg['id'],
                'category': 'general_chat',
                'response_time_seconds': np.nan,
                'response_time_minutes': np.nan,
                'responded': False,
                'sent_timestamp': ai_timestamp,
                'response_timestamp': None,
                'source': 'ai_messages',
                'message_content': str(ai_msg['content'])[:200] if pd.notna(ai_msg['content']) else None,
                'details': f"msg_type={ai_msg['message_type']}, msg_info={ai_msg['message_info']}"
            })
    
    
    # 3. SURVEYS: User starting scheduled surveys
    
    surveys_df = pd.read_sql(
        """
        SELECT id, user_id, survey_type, sent_at, started_at, is_completed 
        FROM surveys 
        WHERE user_id = ANY(%(user_ids)s)
        """,
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    surveys_df['sent_at'] = pd.to_datetime(surveys_df['sent_at'])
    surveys_df['started_at'] = pd.to_datetime(surveys_df['started_at'])
    
    for _, survey in surveys_df.iterrows():
        if pd.notna(survey['started_at']):
            # User started the survey
            response_time = (survey['started_at'] - survey['sent_at']).total_seconds()
            all_response_times.append({
                'message_id': survey['id'],
                'category': 'survey',
                'response_time_seconds': response_time,
                'response_time_minutes': response_time / 60,
                'responded': True,
                'sent_timestamp': survey['sent_at'],
                'response_timestamp': survey['started_at'],
                'source': 'surveys',
                'message_content': f"Survey: {survey['survey_type']}",
                'details': f"survey_id={survey['id']}, completed={survey['is_completed']}"
            })
        else:
            # User never started the survey
            all_response_times.append({
                'message_id': survey['id'],
                'category': 'survey',
                'response_time_seconds': np.nan,
                'response_time_minutes': np.nan,
                'responded': False,
                'sent_timestamp': survey['sent_at'],
                'response_timestamp': None,
                'source': 'surveys',
                'message_content': f"Survey: {survey['survey_type']}",
                'details': f"survey_id={survey['id']}, completed={survey['is_completed']}"
            })
    
    # 4. SEIZURE EVENTS: User responding to seizure notifications
    
    # Get user timezones and databricks_guids for conversion and routing
    user_timezones = {}
    user_guids = {}
    if len(users_df) > 0:
        users_with_info = pd.read_sql(
            """
            SELECT id, timezone, databricks_guid
            FROM users 
            WHERE id = ANY(%(user_ids)s)
            """,
            engine,
            params={'user_ids': list(included_user_ids)}
        )
        user_timezones = dict(zip(users_with_info['id'], users_with_info['timezone']))
        user_guids = dict(zip(users_with_info['id'], users_with_info['databricks_guid']))
    
    # Patient GUIDs that use the sparcnet_annotations table (configured via env var)
    SPARCNET_GUIDS = {
        g.strip() for g in os.environ.get("SPARCNET_GUIDS", "").split(",") if g.strip()
    }
    
    # Get local seizure events with databricks_event_id
    seizure_events_df = pd.read_sql(
        """
        SELECT id, user_id, notification_sent_at, detected_at, original_event_id, databricks_event_id
        FROM seizure_events
        WHERE user_id = ANY(%(user_ids)s) 
        AND notification_sent_at IS NOT NULL
        AND databricks_event_id IS NOT NULL
        """,
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    seizure_events_df['notification_sent_at'] = pd.to_datetime(seizure_events_df['notification_sent_at'])
    seizure_events_df['detected_at'] = pd.to_datetime(seizure_events_df['detected_at'])
    
    # Query BOTH Databricks tables for annotations using databricks_event_id
    seizure_annotations = {}  # {databricks_event_id: created_at}
    
    if len(seizure_events_df) > 0:
        # Get all databricks_event_ids
        databricks_event_ids = [f"'{eid}'" for eid in seizure_events_df['databricks_event_id'].tolist()]
        event_ids_str = ','.join(databricks_event_ids)
        
        # Query sparcnet_annotations
        try:
            conn_params = get_databricks_connection_params()
            annotations_table = get_databricks_sparcnet_annotations_table()
            
            with databricks_sql.connect(**conn_params) as connection:
                with connection.cursor() as cursor:
                    query = f"""
                        SELECT databricks_event_id, created_at
                        FROM {annotations_table}
                        WHERE databricks_event_id IN ({event_ids_str})
                    """
                    cursor.execute(query)
                    
                    for row in cursor.fetchall():
                        seizure_annotations[row[0]] = row[1]
                    
                    print(f"Found {len(seizure_annotations)} matches in sparcnet_annotations")
        except Exception as e:
            print(f"Warning: Could not fetch sparcnet annotations from Databricks: {e}")
        
        # Query seizure_annotations (regular table)
        try:
            conn_params = get_databricks_connection_params()
            annotations_table = get_databricks_seizure_annotations_table()
            
            with databricks_sql.connect(**conn_params) as connection:
                with connection.cursor() as cursor:
                    query = f"""
                        SELECT databricks_event_id, created_at
                        FROM {annotations_table}
                        WHERE databricks_event_id IN ({event_ids_str})
                    """
                    cursor.execute(query)
                    
                    matches_before = len(seizure_annotations)
                    for row in cursor.fetchall():
                        # Only add if not already found in sparcnet (sparcnet takes priority)
                        if row[0] not in seizure_annotations:
                            seizure_annotations[row[0]] = row[1]
                    
                    new_matches = len(seizure_annotations) - matches_before
                    print(f"Found {new_matches} additional matches in seizure_annotations")
        except Exception as e:
            print(f"Warning: Could not fetch seizure annotations from Databricks: {e}")
    
    # Process seizure events
    for _, seizure in seizure_events_df.iterrows():
        databricks_event_id = seizure['databricks_event_id']
        response_timestamp = seizure_annotations.get(databricks_event_id)
        
        if response_timestamp is not None:
            # User responded to seizure notification
            # Use created_at from Databricks directly (already in UTC)
            response_timestamp = pd.to_datetime(response_timestamp)
            
            # Ensure it's timezone-aware
            if response_timestamp.tzinfo is None:
                response_timestamp = response_timestamp.tz_localize(pytz.UTC)
            
            response_time = (response_timestamp - seizure['notification_sent_at']).total_seconds()
            all_response_times.append({
                'message_id': seizure['id'],
                'category': 'seizure_event',
                'response_time_seconds': response_time,
                'response_time_minutes': response_time / 60,
                'responded': True,
                'sent_timestamp': seizure['notification_sent_at'],
                'response_timestamp': response_timestamp,
                'source': 'seizure_events',
                'message_content': f"Seizure detected at {seizure['detected_at']}" if pd.notna(seizure['detected_at']) else 'Seizure event',
                'details': f"event_id={seizure['id']}, databricks_event_id={databricks_event_id}"
            })
        else:
            # User didn't respond to seizure notification
            all_response_times.append({
                'message_id': seizure['id'],
                'category': 'seizure_event',
                'response_time_seconds': np.nan,
                'response_time_minutes': np.nan,
                'responded': False,
                'sent_timestamp': seizure['notification_sent_at'],
                'response_timestamp': None,
                'source': 'seizure_events',
                'message_content': f"Seizure detected at {seizure['detected_at']}" if pd.notna(seizure['detected_at']) else 'Seizure event',
                'details': f"event_id={seizure['id']}, databricks_event_id={databricks_event_id}"
            })
    
    # 5. SPIKE EVENTS: User responding to spike notifications
    
    # Get local spike events with original_event_id
    spike_events_df = pd.read_sql(
        """
        SELECT id, user_id, notification_sent_at, detected_at, original_event_id
        FROM spike_events
        WHERE user_id = ANY(%(user_ids)s) 
        AND notification_sent_at IS NOT NULL
        AND original_event_id IS NOT NULL
        """,
        engine,
        params={'user_ids': list(included_user_ids)}
    )
    spike_events_df['notification_sent_at'] = pd.to_datetime(spike_events_df['notification_sent_at'])
    spike_events_df['detected_at'] = pd.to_datetime(spike_events_df['detected_at'])
    
    # Query Databricks for spike annotations
    spike_annotations = {}
    if len(spike_events_df) > 0:
        try:
            conn_params = get_databricks_connection_params()
            annotations_table = get_databricks_spike_annotations_table()
            
            with databricks_sql.connect(**conn_params) as connection:
                with connection.cursor() as cursor:
                    # Get all annotations for our events
                    event_ids = spike_events_df['original_event_id'].unique().tolist()
                    event_ids_str = ','.join([str(int(eid)) for eid in event_ids])
                    
                    query = f"""
                        SELECT local_spike_event_id, created_at
                        FROM {annotations_table}
                        WHERE local_spike_event_id IN ({event_ids_str})
                    """
                    cursor.execute(query)
                    
                    # Store annotations in dict: {original_event_id: created_at}
                    for row in cursor.fetchall():
                        spike_annotations[row[0]] = row[1]
        except Exception as e:
            print(f"Warning: Could not fetch spike annotations from Databricks: {e}")
    
    # Process spike events
    for _, spike in spike_events_df.iterrows():
        original_event_id = spike['original_event_id']
        response_timestamp = spike_annotations.get(original_event_id)
        
        if response_timestamp is not None:
            # User responded to spike notification
            # Use created_at from Databricks directly (already in UTC)
            response_timestamp = pd.to_datetime(response_timestamp)
            
            # Ensure it's timezone-aware
            if response_timestamp.tzinfo is None:
                response_timestamp = response_timestamp.tz_localize(pytz.UTC)
            
            response_time = (response_timestamp - spike['notification_sent_at']).total_seconds()
            all_response_times.append({
                'message_id': spike['id'],
                'category': 'spike_event',
                'response_time_seconds': response_time,
                'response_time_minutes': response_time / 60,
                'responded': True,
                'sent_timestamp': spike['notification_sent_at'],
                'response_timestamp': response_timestamp,
                'source': 'spike_events',
                'message_content': f"Spike event detected at {spike['detected_at']}" if pd.notna(spike['detected_at']) else 'Spike event',
                'details': f"event_id={spike['id']}, databricks_event_id={original_event_id}"
            })
        else:
            # User didn't respond to spike notification
            all_response_times.append({
                'message_id': spike['id'],
                'category': 'spike_event',
                'response_time_seconds': np.nan,
                'response_time_minutes': np.nan,
                'responded': False,
                'sent_timestamp': spike['notification_sent_at'],
                'response_timestamp': None,
                'source': 'spike_events',
                'message_content': f"Spike event detected at {spike['detected_at']}" if pd.notna(spike['detected_at']) else 'Spike event',
                'details': f"event_id={spike['id']}, databricks_event_id={original_event_id}"
            })
    
    # Create final DataFrames
    
    # Included messages
    included_df = pd.DataFrame(all_response_times)
    
    # Categorize response times into bins
    def categorize_time(seconds):
        if pd.isna(seconds):
            return 'no_response'
        elif seconds < 60:
            return '0-1min'
        elif seconds < 120:
            return '1-2min'
        elif seconds < 300:
            return '2-5min'
        elif seconds < 600:
            return '5-10min'
        elif seconds < 1800:
            return '10-30min'
        elif seconds < 3600:
            return '30-60min'
        elif seconds < 7200:
            return '1-2hr'
        elif seconds < 21600:
            return '2-6hr'
        elif seconds < 43200:
            return '6-12hr'
        elif seconds < 86400:
            return '12-24hr'
        else:
            return '>24hr'
    
    included_df['time_bin'] = included_df['response_time_seconds'].apply(categorize_time)
    
    # Reorder columns for included_df (for CSV export)
    included_column_order = [
        'message_id',
        'category',
        'response_time_seconds',
        'response_time_minutes',
        'responded',
        'source',
        'message_content',
        'sent_timestamp',
        'response_timestamp',
        'details',
        'time_bin'
    ]
    # Only reorder columns that exist
    included_column_order = [col for col in included_column_order if col in included_df.columns]
    # Add any remaining columns not in the order list
    remaining_cols = [col for col in included_df.columns if col not in included_column_order]
    included_df = included_df[included_column_order + remaining_cols]
    
    # Excluded messages
    excluded_df = pd.DataFrame(all_excluded)
    
    # Reorder columns for excluded_df (for CSV export)
    if len(excluded_df) > 0:
        excluded_column_order = [
            'message_id',
            'exclusion_reason',
            'source',
            'message_type',
            'message_info',
            'timestamp',
            'content_preview'
        ]
        # Only reorder columns that exist
        excluded_column_order = [col for col in excluded_column_order if col in excluded_df.columns]
        # Add any remaining columns not in the order list
        remaining_cols = [col for col in excluded_df.columns if col not in excluded_column_order]
        excluded_df = excluded_df[excluded_column_order + remaining_cols]
    
    return included_df, excluded_df
