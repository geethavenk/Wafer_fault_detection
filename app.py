import streamlit as st
from src import utils
import pandas as pd
from src.pipelines.prediction_pipeline import PredictionPipeline
import io

# side bar with title
st.sidebar.title('Wafer Fault Detection')

# side bar about the author
if st.sidebar.button('About creator'):
    utils.about_me()

# File uploader for CSV or TXT files
file = st.file_uploader('Upload file', type=['csv', 'txt'])

if file is not None:
    # Read the uploaded file into a DataFrame
    df = pd.read_csv(file)

    # Create an instance of the PredictionPipeline
    obj = PredictionPipeline()

    # Only run the prediction when the "Predict" button is clicked
    if st.button('Predict'):
        # Predict the outcomes
        y_pred = obj.predict(df)
        
        # Display the predictions
        st.write(y_pred)

        # Convert predictions to CSV format
        pred_csv = y_pred.to_csv(index=False).encode('utf-8')

        # Create a file-like object from the bytes data
        pred_csv_io = io.BytesIO(pred_csv)

        # Provide an option to download the prediction file
        st.download_button(
            label="Download CSV file",
            data=pred_csv_io,
            file_name="predictions.csv",
            mime="text/csv"
        )