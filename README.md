Saccade Analysis Streamlit Application
=======================================

This application provides a graphical user interface for processing, analyzing, 
and summarizing eye-tracking data from antisaccade and prosaccade tasks.

It is designed to be modular and configurable, allowing researchers to:
- Upload raw saccade report files (xlsx or csv format).
- Map columns from their data file to the required internal variable names.
- Configure the analysis based on experimental conditions (e.g., target amplitude, 
  emotional valence, gap/step).
- Calculate key metrics like percent error, latency, and gain.
- Generate a wide-format summary table with one row per participant, suitable
  for statistical analysis in other software.
- NOTE: this program is designed to be run on the output of the Eyelink Data Viewer Saccade Report (base settings).
- When using a gap/step/overlap condition, the script subtracts the number (denoting milliseconds) from the indicated column in order to account for any gap in stimulus presentation. eg. a 200ms gap between fixation disappearance and stimulus appearance should be coded as 200 in the gap column; a step condition should be coded as 0; an overlap of 200ms should be coded as -200.

To run the app, use the following command in your terminal:
`streamlit run your_app_filename.py`

TODO: write EDF (eyelink data format) translator and saccade/event detection section