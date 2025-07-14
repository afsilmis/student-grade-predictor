import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.2rem;
        margin: 2rem 0;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
    }
    
    .stSlider > div > div {
        background-color: white;
        border-radius: 8px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model & features
@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load('student_perform_prediction.pkl')
        with open('selected_features.json') as f:
            selected_features = json.load(f)
        with open('outlier_bounds.json') as f:
            bounds = json.load(f)
        return model, selected_features, bounds
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None, None

model, selected_features, bounds = load_model_and_features()

# Header
st.markdown("""
<div class="main-header">
    <h1><i class="bi bi-mortarboard-fill"></i> Student Grade Predictor</h1>
    <p>Advanced AI-powered academic performance prediction system</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Please ensure all model files are available in the directory.")
    st.stop()

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Student Information")
    
    # Create form for inputs
    with st.form("student_form"):
        input_data = {}
        
        # Academic Performance Section
        st.markdown("""<h4><i class="bi bi-bar-chart"></i> Academic Performance</h4>""", unsafe_allow_html=True)
        acad_col1, acad_col2 = st.columns(2)
        
        with acad_col1:
            if 'G1' in selected_features:
                input_data['G1'] = st.number_input(
                    "First Period Grade",
                    min_value=0, max_value=20, value=10, step=1,
                    help="Grade from first period (0-20 scale)"
                )
        
        with acad_col2:
            if 'G2' in selected_features:
                input_data['G2'] = st.number_input(
                    "Second Period Grade",
                    min_value=0, max_value=20, value=10, step=1,
                    help="Grade from second period (0-20 scale)"
                )
        
        # Personal Information Section
        st.markdown("---")
        st.markdown("""<h4><i class="bi bi-person"></i> Personal Information</h4>""", unsafe_allow_html=True)
        personal_col1, personal_col2 = st.columns(2)
        
        with personal_col1:
            if 'age' in selected_features:
                input_data['age'] = st.number_input(
                    "Age",
                    min_value=15, max_value=22, value=17, step=1,
                    help="Student's age"
                )
            
            if 'sex_M' in selected_features:
                sex = st.selectbox(
                    "Gender",
                    ['Female', 'Male'],
                    help="Student's gender"
                )
                input_data['sex_M'] = 1 if sex == 'Male' else 0
        
        with personal_col2:
            if 'course_por' in selected_features:
                course = st.selectbox(
                    "Course",
                    ['Math', 'Portuguese Language'],
                    help="Subject being studied"
                )
                input_data['course_por'] = 0 if course == 'Math' else 1
        
        # Behavior & Attendance Section
        st.markdown("---")
        st.markdown("""<h4><i class="bi bi-journal-check"></i> Behavior & Attendance</h4>""", unsafe_allow_html=True)
        behavior_col1, behavior_col2 = st.columns(2)
        
        with behavior_col1:
            if 'absences' in selected_features:
                input_data['absences'] = st.number_input(
                    "Number of Absences",
                    min_value=0, max_value=100, value=0, step=1,
                    help="Total number of school absences"
                )
        
        with behavior_col2:
            if 'goout' in selected_features:
                input_data['goout'] = st.slider(
                    "Social Activity Level",
                    1, 5, 3,
                    help="How often student goes out with friends (1=Very Low, 5=Very High)"
                )
        
        # Handle any additional features that might be in the model
        for feature in selected_features:
            if feature not in input_data:
                st.info(f"Additional feature '{feature}' set to default value")
                input_data[feature] = 0
        
        # Predict button
        predict_button = st.form_submit_button(
            "🔮 Predict Grade",
            use_container_width=True
        )

with col2:
    st.markdown("### Prediction Insights")
    
    # Information box
    st.markdown("""
    <div class="info-box">
        <h4>💡 How it works</h4>
        <p>This AI model uses machine learning to predict final grades based on:</p>
        <ul>
            <li>Previous academic performance</li>
            <li>Student demographics</li>
            <li>Behavioral patterns</li>
            <li>Attendance records</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Grade scale reference
    st.markdown("### Grade Scale Reference")
    grade_scale = pd.DataFrame({
        'Grade Range': ['18-20', '16-17', '14-15', '12-13', '10-11', '0-9'],
        'Performance': ['Excellent', 'Very Good', 'Good', 'Satisfactory', 'Needs Improvement', 'Unsatisfactory'],
        'Color': ['#2E8B57', '#32CD32', '#FFD700', '#FFA500', '#FF6347', '#DC143C']
    })
    
    for i, row in grade_scale.iterrows():
        st.markdown(f"""
        <div style="background-color: {row['Color']}; color: white; padding: 0.75rem; 
                    border-radius: 5px; margin: 0.25rem 0; text-align: center;">
            <strong>{row['Grade Range']}</strong> - {row['Performance']}
        </div>
        """, unsafe_allow_html=True)

# Prediction logic
if predict_button:
    try:
        # Create DataFrame with correct feature order
        # Ensure all selected features are present and in the right order
        ordered_data = []
        for feature in selected_features:
            if feature in input_data:
                ordered_data.append(input_data[feature])
            else:
                # Default value for missing features
                ordered_data.append(0)
        
        input_df = pd.DataFrame([ordered_data], columns=selected_features)
        
        # Apply outlier bounds
        for col in input_df.columns:
            if col in bounds:
                lower = bounds[col]['lower']
                upper = bounds[col]['upper']
                input_df[col] = np.clip(input_df[col], lower, upper)
        
        # Convert to float
        input_df = input_df.astype(float)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_rounded = int(round(prediction))
        prediction_final = min(max(prediction_rounded, 0), 20)
        
        # Determine performance level and color
        if prediction_final >= 18:
            performance = "Excellent"
            color = "#2E8B57"
            emoji = "🌟"
        elif prediction_final >= 16:
            performance = "Very Good"
            color = "#32CD32"
            emoji = "🎯"
        elif prediction_final >= 14:
            performance = "Good"
            color = "#FFD700"
            emoji = "👍"
        elif prediction_final >= 12:
            performance = "Satisfactory"
            color = "#FFA500"
            emoji = "📚"
        elif prediction_final >= 10:
            performance = "Needs Improvement"
            color = "#FF6347"
            emoji = "⚠️"
        else:
            performance = "Unsatisfactory"
            color = "#DC143C"
            emoji = "🔴"
        
        # Display results
        st.markdown("---")
        st.markdown("## Prediction Results")
        
        col1, _, col2 = st.columns([1,0.2, 1.5])
        
        with col1:
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prediction_final,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Grade Prediction"},
                delta = {'reference': 10},
                gauge = {
                    'axis': {'range': [None, 20]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgray"},
                        {'range': [10, 15], 'color': "gray"},
                        {'range': [15, 20], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 10
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                font={'color': "darkblue", 'family': "Arial"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Main result
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color} 0%, {color}CC 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; 
                        color: white; margin: 2rem 0;">
                <h2><i class="bi bi-clipboard2-pulse"></i> Predicted Final Grade: {prediction_final}/20</h2>
                <h3>Performance Level: {performance}</h3>
            </div>
            """, unsafe_allow_html=True)

        # Additional insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Grade Prediction",
                value=f"{prediction_final}/20",
                delta=f"{prediction_final - 10} from average"
            )
        
        with col2:
            percentage = (prediction_final / 20) * 100
            st.metric(
                label="Percentage Score",
                value=f"{percentage:.1f}%",
                delta=f"{percentage - 50:.1f}% from passing"
            )
        
        with col3:
            st.metric(
                label="Performance Level",
                value=performance,
                delta=None
            )
        
        # Recommendations
        st.markdown("---")
        st.markdown("### Recommendations")
        
        if prediction_final < 10:
            st.error("🚨 **Immediate Action Required**: Consider extra tutoring and study support.")
        elif prediction_final < 14:
            st.warning("⚠️ **Improvement Needed**: Regular study schedule and attendance focus recommended.")
        else:
            st.success("✅ **Good Progress**: Continue current study habits and maintain consistency.")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please check that all required files are available and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎓 Student Grade Predictor | Powered by Machine Learning</p>
    <p><em>This prediction is based on historical data and should be used as a guide only.</em></p>
</div>
""", unsafe_allow_html=True)