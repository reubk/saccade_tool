import streamlit as st
import pandas as pd
import numpy as np
from functools import reduce
import io
from itertools import combinations
import plotly.express as px
import xlsxwriter as xw
from typing import Dict, Any, List, Union, Optional
import warnings

# Suppress FutureWarnings from pandas
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Data Processing Functions ---

@st.cache_data
def load_data(uploaded_file: Any) -> Optional[pd.DataFrame]:
    """
    Reads an uploaded Excel file into a pandas DataFrame.
    Caches the result to prevent re-reading the file on every interaction.
    """
    try:
        return pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

@st.cache_data
def preprocess_data(_df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Takes the raw DataFrame and applies all initial cleaning and preparation steps.
    
    This includes renaming columns based on user mapping, filtering invalid trials,
    converting data types, and calculating saccade latency. The _df argument
    is a convention to indicate that the cached function receives the raw DataFrame.
    """
    df = _df.copy()
    
    # Create a mapping from the source file's column names to our internal names.
    rename_map = {v: k for k, v in config['column_mapping'].items() if v != '-'}
    
    # Validate that all selected source columns actually exist in the uploaded file.
    missing_source_cols = [src for src in rename_map.keys() if src not in df.columns]
    if missing_source_cols:
        st.error(f"Column Mismatch: The following columns were not found in your file: {missing_source_cols}")
        return None
    
    df = df.rename(columns=rename_map)
    
    # Basic data cleaning and filtering.
    df = df[~df['CURRENT_SAC_DIRECTION'].isin(['.', 'UP', 'DOWN'])]
    df = df[df['TARGET_ONSET_TIME'].notna()]
    df = df[df['TARGET_ONSET_TIME'] != 'UNDEFINED']

    # If the paradigm is 'interleaved', filter for the specific task ('antisaccade' or 'prosaccade').
    if config['task_type'] == 'interleaved':
        task_instruction = 'AWAY' if config['current_task'] == 'antisaccade' else 'TOWARD'
        df = df[df['BLOCK_INSTRUCTION'] == task_instruction]
    
    # Ensure all key numeric columns are properly typed, coercing errors to NaN.
    numeric_cols = [
        'CURRENT_SAC_AMPLITUDE', 'CURRENT_SAC_AVG_VELOCITY', 
        'CURRENT_SAC_PEAK_VELOCITY', 'TARGET_ONSET_TIME', 'CURRENT_SAC_START_TIME'
    ]
    # Also ensure the GAP column is numeric if it's been mapped by the user.
    if 'GAP' in df.columns:
        numeric_cols.append('GAP')
        
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
    for col in existing_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop rows where any of these critical columns are not valid numbers.
    df.dropna(subset=existing_numeric_cols, inplace=True)
    
    # Calculate the raw latency first.
    df['LATENCY'] = df['CURRENT_SAC_START_TIME'] - df['TARGET_ONSET_TIME']
    
    # If a GAP column exists, subtract its value to get the true latency.
    # This is now independent of the "group by gap" checkbox.
    if 'GAP' in df.columns:
        df['LATENCY'] = df['LATENCY'] - df['GAP']
    
    # Now, apply the minimum latency filter to the corrected latency values.
    df = df[df['LATENCY'] > 120]
    
    return df

@st.cache_data
def calculate_error_and_gain(_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculates trial error and saccadic gain.
    
    Error logic depends on the task (pro- vs. anti-saccade). Gain is calculated
    if target amplitude information is provided.
    """
    df = _df.copy()
    
    # Standardize direction representation: 1 for Right, -1 for Left.
    # This makes subsequent logic much clearer and less error-prone.
    # Using .str.lower() adds robustness against case variations (e.g., 'Right' vs 'right').
    df['TargetDirectionNumeric'] = np.where(df['TARGET_DIRECTION'].str.lower() == 'right', 1, -1)
    df['SaccadeDirectionNumeric'] = np.where(df['CURRENT_SAC_DIRECTION'].str.lower() == 'right', 1, -1)
    
    # The logic for determining a correct trial is opposite for anti- and pro-saccades.
    if config['current_task'] == 'antisaccade':
        # For antisaccades, an error occurs if saccade is in the SAME direction as the target.
        df['ERROR'] = np.where(df['TargetDirectionNumeric'] == df['SaccadeDirectionNumeric'], 1, 0) # 1=incorrect, 0=correct
    else: # prosaccade
        # For prosaccades, an error occurs if saccade is in the OPPOSITE direction of the target.
        df['ERROR'] = np.where(df['TargetDirectionNumeric'] != df['SaccadeDirectionNumeric'], 1, 0) # 1=incorrect, 0=correct
    
    # Calculate GAIN if amplitude analysis is enabled.
    if config['use_amplitude'] and config.get('amplitude_map'):
        amp_map = {k: pd.to_numeric(v, errors='coerce') for k, v in config['amplitude_map'].items()}
        df['TARGET_DEGREES'] = df['AMPLITUDE'].astype(str).map(amp_map)
        df['GAIN'] = (df['CURRENT_SAC_AMPLITUDE'] / df['TARGET_DEGREES']).fillna(0)
    else:
        df['GAIN'] = np.nan

    return df

@st.cache_data
def find_primary_saccades(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the primary saccade for each trial using a multi-step process.

    For each trial, it first considers the first three saccades occurring after
    the 120ms latency cutoff. From this subset of three, it then selects the 
    very first saccade that has an amplitude of 2 degrees or greater.
    """
    df = _df.copy()
    
    # Sort all saccades by latency to ensure we process them in temporal order.
    df_sorted = df.sort_values(by=['DATA_FILE', 'TRIAL_LABEL', 'LATENCY'])
    
    # Group by trial and take the first 3 saccades for each one.
    # This creates a subset of the data containing only the initial saccades.
    first_three_saccades = df_sorted.groupby(['DATA_FILE', 'TRIAL_LABEL']).head(3)
    
    # From that subset, filter for saccades that meet the amplitude criterion.
    valid_amplitude_saccades = first_three_saccades[first_three_saccades['CURRENT_SAC_AMPLITUDE'] >= 2]
    
    # Finally, group by trial again and take the very first saccade from the valid amplitude list.
    # This is the primary saccade.
    primary_saccades = valid_amplitude_saccades.groupby(['DATA_FILE', 'TRIAL_LABEL']).head(1)
    
    return primary_saccades

@st.cache_data
def summarize_data_wide(_primary_saccades: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculates all descriptive statistics and pivots the data to produce a
    wide-format output with one row per participant.
    """
    df = _primary_saccades.copy()
    
    base_group = 'RECORDING_SESSION_LABEL'
    grouping_cols = config.get('grouping_cols', [])
    metrics = config.get('metrics', [])
    all_summaries = []

# Helper function to compute descriptive stats for a given metric.
    def calculate_descriptives(data: pd.DataFrame, metric: str, remove_outliers: bool) -> pd.Series:
        if metric == 'percentError':
            incorrects = (data['ERROR'] == 1).sum()
            total = len(data)
            return pd.Series({'percentError': (incorrects / total * 100) if total > 0 else 0})

        # For other metrics, use only correct trials.
        correct_data = data[data['ERROR'] == 0]
        if metric not in correct_data.columns or correct_data.empty:
            return pd.Series({f'{metric}_mean': np.nan, f'{metric}_std': np.nan})
        
        # Option to remove outliers: filter data to be within +/- 2 standard deviations of the mean.
        if remove_outliers:
            ll = correct_data[metric].mean() - 2 * correct_data[metric].std()
            ul = correct_data[metric].mean() + 2 * correct_data[metric].std()
            valid_data = correct_data[correct_data[metric].between(ll, ul)]
        else:
            valid_data = correct_data

        if valid_data.empty:
            return pd.Series({f'{metric}_mean': np.nan, f'{metric}_std': np.nan})
        
        return pd.Series({
            f'{metric}_mean': valid_data[metric].mean(),
            f'{metric}_std': valid_data[metric].std()
        })

    # Generate all possible combinations of conditions (for main effects, interactions, etc.).
    all_grouping_combos: List[tuple] = []
    for i in range(len(grouping_cols) + 1):
        all_grouping_combos.extend(combinations(grouping_cols, i))

    # Loop through each metric and each condition combination to build the final table.
    for metric in metrics:
        for combo in all_grouping_combos:
            current_grouping = [base_group] + list(combo)
            
            # Pass the outlier removal flag to the calculation function
            summary = df.groupby(current_grouping).apply(
                calculate_descriptives, metric, remove_outliers=config['use_outlier_removal']
                ).reset_index()
            
            # We melt and pivot to transform the data into the desired wide format.
            id_vars = [base_group] + list(combo)
            melted = summary.melt(id_vars=id_vars, var_name='stat', value_name='value')
            
            # Create clear, descriptive column names.
            # Create clear, descriptive column names (e.g., "gap_LATENCY_mean")
            if list(combo):
                condition_name = melted[list(combo)].astype(str).agg('_'.join, axis=1)
                melted['col_name'] = condition_name + '_' + melted['stat']
            else:
                melted['col_name'] = 'overall_' + melted['stat']

            pivoted = melted.pivot_table(index=base_group, columns='col_name', values='value').reset_index()
            all_summaries.append(pivoted)

    if not all_summaries:
        return pd.DataFrame(columns=pd.Index([base_group]))

    # Merge all the pivoted tables into a single master DataFrame.
    final_df: pd.DataFrame = reduce(lambda left, right: pd.merge(left, right, on=base_group, how='outer'), all_summaries)
    return final_df

@st.cache_data
def generate_plotting_data(_primary_saccades: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Generates a long-format DataFrame suitable for plotting.
    Calculates mean metric values per participant for each condition.
    """
    df = _primary_saccades.copy()
    grouping_cols = config.get('grouping_cols', [])
    metrics = config.get('metrics', [])
    
    # Base grouping for all calculations: per participant, per condition
    base_grouping = ['RECORDING_SESSION_LABEL'] + grouping_cols

    all_participant_summaries = []

    # Calculate percentError
    if 'percentError' in metrics:
        # Group by participant and conditions, then calculate error rate for each group
        error_summary = df.groupby(base_grouping).apply(
            lambda x: (x['ERROR'].sum() / len(x) * 100) if len(x) > 0 else 0
        ).reset_index(name='value')
        error_summary['metric'] = 'percentError'
        all_participant_summaries.append(error_summary)

    # Calculate other metrics on correct trials
    correct_trials = df[df['ERROR'] == 0]
    other_metrics = [m for m in metrics if m != 'percentError' and m in correct_trials.columns]
    
    if other_metrics:
        # Calculate the mean for each metric for each participant/condition group
        summary = correct_trials.groupby(base_grouping)[other_metrics].mean(numeric_only=True).reset_index()
        
        # Melt the summary to get a long format: [participant, condition_cols..., metric, value]
        melted_summary = summary.melt(
            id_vars=base_grouping,
            value_vars=other_metrics,
            var_name='metric',
            value_name='value'
        )
        all_participant_summaries.append(melted_summary)

    if not all_participant_summaries:
        return pd.DataFrame()

    plotting_df: pd.DataFrame = pd.concat(all_participant_summaries, ignore_index=True)

    if grouping_cols:
        plotting_df['condition'] = plotting_df[grouping_cols].astype(str).agg(' - '.join, axis=1)
    else:
        plotting_df['condition'] = 'Overall'
        
    return plotting_df

# --- Streamlit User Interface ---

st.set_page_config(layout="wide", page_title="Saccade Analysis")
st.title("Saccade Analysis Workbench")

# Initialize session state to hold data and results across reruns.
if 'df' not in st.session_state:
    st.session_state.df = None
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'plotting_df' not in st.session_state:
    st.session_state.plotting_df = None

with st.sidebar:
    st.header("Setup & Configuration")
    
    uploaded_file = st.file_uploader("1. Upload Saccade Data File", type=["xlsx", "xls"])

    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
    
    if st.session_state.df is not None:
        df_cols = ["-"] + st.session_state.df.columns.tolist()

        st.subheader("2. Paradigm Settings")
        task_type_options = {'Antisaccade only': 'antisaccade', 'Prosaccade only': 'prosaccade', 'Interleaved': 'interleaved'}
        task_type_selection = st.radio("Task Type:", list(task_type_options.keys()))
        
        st.subheader("3. Analysis Conditions")
        st.write("Select conditions to group the results by.")
        use_amplitude = st.checkbox("Target Amplitude")
        use_emotional_valence = st.checkbox("Emotional Valence")
        use_gap_condition = st.checkbox("Gap/Step Condition")

        st.subheader("4. Column Mapping")
        st.write("Match your file's columns to the required variables.")
        core_cols = {'RECORDING_SESSION_LABEL':'Participant ID','CURRENT_SAC_DIRECTION':'Saccade Direction','TARGET_ONSET_TIME': "Target Onset Time",'CURRENT_SAC_AMPLITUDE':'Saccade Amplitude','CURRENT_SAC_START_TIME':'Saccade Start Time','DATA_FILE':'EDF File Name','TRIAL_LABEL':'Trial Label','TARGET_DIRECTION': "Target Direction"}
        optional_cols = {'CURRENT_SAC_AVG_VELOCITY':'Saccade Avg Velocity','CURRENT_SAC_PEAK_VELOCITY':'Saccade Peak Velocity'}
        
        config: Dict[str, Any] = {'grouping_cols': []}
        
        if task_type_selection == 'Interleaved':
            core_cols['BLOCK_INSTRUCTION'] = 'Task Instruction (TOWARD/AWAY)'
        if use_amplitude:
            core_cols['AMPLITUDE'] = 'Target Amplitude Labels'
            config['grouping_cols'].append('AMPLITUDE')
        if use_emotional_valence:
            core_cols['IMAGE_TYPE'] = 'Stimulus Valence'
            config['grouping_cols'].append('IMAGE_TYPE')
        if use_gap_condition:
            core_cols['GAP'] = 'Gap/Step/Overlap Info'
            config['grouping_cols'].append('GAP')

        column_mapping: Dict[str, str] = {}
        st.write("**Core Variables:**")
        for internal_name, desc in core_cols.items():
            column_mapping[internal_name] = st.selectbox(f"{desc}:", df_cols, key=f"map_{internal_name}")
        
        st.write("**Optional Metrics:**")
        for internal_name, desc in optional_cols.items():
            column_mapping[internal_name] = st.selectbox(f"{desc}:", df_cols, key=f"map_{internal_name}")

        amplitude_map = {}
        if use_amplitude and column_mapping.get('AMPLITUDE', '-') != '-':
            st.subheader("5. Amplitude Mapping")
            st.write("Specify the degrees of visual angle for each amplitude label.")
            try:
                amp_labels = st.session_state.df[column_mapping['AMPLITUDE']].dropna().unique()
                for label in amp_labels:
                    amplitude_map[str(label)] = st.text_input(f"Degrees for '{label}':", value=str(label), key=f"amp_deg_{label}")
            except KeyError:
                st.warning("Please map the 'Target Amplitude Labels' column first.")

        st.subheader("6. Output Metrics")
        available_metrics = ['percentError', 'LATENCY', 'GAIN', 'CURRENT_SAC_AMPLITUDE', 'CURRENT_SAC_AVG_VELOCITY', 'CURRENT_SAC_PEAK_VELOCITY']
        if not use_amplitude:
            if 'GAIN' in available_metrics: available_metrics.remove('GAIN')
        
        default_metrics = ['percentError', 'LATENCY', 'GAIN']
        selected_metrics = st.multiselect("Select metrics to calculate:", available_metrics, default=[m for m in default_metrics if m in available_metrics])
        
        st.subheader("7. Analysis Settings")
        use_outlier_removal = st.checkbox("Remove metric outliers (> 2 SD from mean)", value=True)

        separate_by_task = False
        if task_type_selection == 'Interleaved':
            st.subheader("8. Output Format")
            separate_by_task = st.checkbox("Separate columns by task type", value=True)

if st.session_state.df is not None:
    st.header("Uploaded Data Preview")
    st.dataframe(st.session_state.df.head())

    if st.sidebar.button("Run Analysis", use_container_width=True):
        st.session_state.result_df = None
        st.session_state.plotting_df = None
        with st.spinner("Processing data... This may take a moment on the first run."):
            
            final_map = {k: v for k, v in column_mapping.items() if v != '-'}
            analysis_config: Dict[str, Any] = config.copy()
            analysis_config.update({
                'column_mapping': final_map,
                'metrics': selected_metrics,
                'use_amplitude': use_amplitude,
                'amplitude_map': amplitude_map,
                'task_type': task_type_options[task_type_selection],
                'use_outlier_removal': use_outlier_removal,
            })

            tasks_to_run = ['prosaccade', 'antisaccade'] if analysis_config['task_type'] == 'interleaved' else [task_type_options[task_type_selection]]
            
            all_results = {}
            all_plotting_data = {}
            for task in tasks_to_run:
                st.info(f"Analyzing {task.title()} task...")
                task_config = analysis_config.copy()
                task_config['current_task'] = task

                # This is the main analysis pipeline. Each step uses a cached function for speed.
                processed = preprocess_data(st.session_state.df, task_config)
                if processed is not None and not processed.empty:
                    error_gain = calculate_error_and_gain(processed, task_config)
                    primary = find_primary_saccades(error_gain)
                    if not primary.empty:
                        result = summarize_data_wide(primary, task_config)
                        if result is not None and not result.empty:
                            all_results[task] = result

                        plotting_data = generate_plotting_data(primary, task_config)
                        if plotting_data is not None and not plotting_data.empty:
                            all_plotting_data[task] = plotting_data
                        else: st.warning(f"Summarization failed for the {task} task.")
                    else: st.warning(f"No primary saccades were found for the {task} task after filtering.")
                else: st.warning(f"No data remained after preprocessing for the {task} task.")
            
            # Combine results from different tasks into one final DataFrame.
            if all_results:
                if len(all_results) > 1 and separate_by_task:
                    # Suffix columns with task type (e.g., "LATENCY_mean_antisaccade") and merge.
                    for task, df in all_results.items():
                        df.columns = [f"{col}_{task}" if col != 'RECORDING_SESSION_LABEL' else col for col in df.columns]
                    final_df = reduce(lambda left, right: pd.merge(left, right, on='RECORDING_SESSION_LABEL', how='outer'), all_results.values())
                else:
                    # Stack results vertically, adding a column to identify the task.
                    for task, df in all_results.items():
                        df.insert(0, 'task_type', task)
                    final_df = pd.concat(all_results.values(), ignore_index=True)
                
                # Final cleanup for a tidy output table.
                final_df.columns = [str(c) for c in final_df.columns]
                if 'RECORDING_SESSION_LABEL' in final_df.columns:
                     cols = ['RECORDING_SESSION_LABEL'] + [c for c in sorted(final_df.columns) if c != 'RECORDING_SESSION_LABEL']
                     final_df = final_df[cols]

                st.session_state.result_df = final_df
                st.success("Analysis Complete!")

                if all_plotting_data:
                    for task, df in all_plotting_data.items():
                        df['task'] = task
                    final_plotting_df = pd.concat(all_plotting_data.values(), ignore_index=True)
                    st.session_state.plotting_df = final_plotting_df

            else:
                st.error("Analysis failed. Please review your data file and column mappings.")

if st.session_state.result_df is not None and not st.session_state.result_df.empty:
    st.header("Results")
    st.dataframe(st.session_state.result_df)
    
    # Provide a download button for the results.
    @st.cache_data
    def convert_df_to_excel(_df: pd.DataFrame) -> bytes:
        """Converts a DataFrame to an in-memory Excel file (bytes)."""
        buffer = io.BytesIO()
        _df.to_excel(buffer, index=False, sheet_name='Saccade_Results', engine='openpyxl')
        return buffer.getvalue() # No need for seek(0) as getvalue() returns the entire buffer.

    excel_data = convert_df_to_excel(st.session_state.result_df)
    
    st.download_button(
        label="Download Results as Excel",
        data=excel_data,
        file_name="saccade_analysis_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

if st.session_state.plotting_df is not None and not st.session_state.plotting_df.empty:
    st.header("Metric Plots")
    
    plot_df = st.session_state.plotting_df
    
    # Let user select which metric to plot from the available ones.
    metric_to_plot = st.selectbox("Select a metric to visualize:", options=sorted(plot_df['metric'].unique()))
    
    if metric_to_plot:
        df_to_plot = plot_df[plot_df['metric'] == metric_to_plot]
        
        # Create the plot
        fig = px.box(
            df_to_plot, 
            x='condition', 
            y='value', 
            color='condition',
            facet_col='task' if 'task' in df_to_plot.columns and df_to_plot['task'].nunique() > 1 else None,
            points="all",
            labels={'value': metric_to_plot, 'condition': 'Condition'},
            title=f'Distribution of {metric_to_plot} by Condition'
        )
        fig.update_traces(quartilemethod="exclusive") # or "inclusive", "linear"
        st.plotly_chart(fig, use_container_width=True)