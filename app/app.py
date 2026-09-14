# ============================================================
# EDUPRO — PREDICTIVE MODELING DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduPro Predictive Modeling Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# DASHBOARD BACKGROUND
# ============================================================

BACKGROUND_IMAGE_URL = (
    "https://images.unsplash.com/photo-1618044733300-9472054094ee"
    "?q=80&w=2071"
)

st.markdown(
    f"""
    <style>
    /* Force text color to black for light background visibility */
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, 
    .stApp label, .stApp span, .stApp div, .stApp [data-testid="stMetricValue"], 
    .stApp [data-testid="stMetricLabel"] {{
        color: black !important;
    }}

    .stApp {{
        background-image:
        linear-gradient(
            rgba(255, 255, 255, 0.88),
            rgba(255, 255, 255, 0.88)
        ),
        url("{BACKGROUND_IMAGE_URL}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    [data-testid="stHeader"] {{
        background: rgba(255, 255, 255, 0.0);
    }}

    [data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0.90);
    }}

    [data-testid="stMetric"],
    [data-testid="stAlert"],
    .stDataFrame,
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: rgba(255, 255, 255, 0.82);
        border-radius: 12px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 2. PROJECT PATHS
# ============================================================

# Deployment-safe project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_FOLDER = PROJECT_ROOT / "models"

ENROLLMENT_MODEL_PATH = (
    MODELS_FOLDER / "EduPro_Final_Enrollment_Model.joblib"
)

REVENUE_MODEL_PATH = (
    MODELS_FOLDER / "EduPro_Final_Revenue_Model.joblib"
)

# ============================================================
# 3. LOAD FINAL MODELS
# ============================================================

@st.cache_resource
def load_models():
    # --------------------------------------------------------
    # Check Enrollment Model
    # --------------------------------------------------------
    if not ENROLLMENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Enrollment model file not found:\n"
            f"{ENROLLMENT_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Check Revenue Model
    # --------------------------------------------------------
    if not REVENUE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Revenue model file not found:\n"
            f"{REVENUE_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load Models
    # --------------------------------------------------------
    enrollment_model = joblib.load(
        ENROLLMENT_MODEL_PATH
    )

    revenue_model = joblib.load(
        REVENUE_MODEL_PATH
    )

    return enrollment_model, revenue_model

try:
    enrollment_model, revenue_model = load_models()
except FileNotFoundError as e:
    st.error(
        f"❌ Model file error:\n\n{e}"
    )
    st.info(
        "ℹ️ Please verify that the required model files "
        "exist inside the project's models folder."
    )
    st.stop()
except Exception as e:
    st.error(
        "❌ Unable to load the final prediction models."
    )
    st.info(
        "ℹ️ Please verify the model files and restart "
        "the Streamlit application."
    )
    st.stop()

# ============================================================
# 4. ORIGINAL MODEL FEATURES
# ============================================================

original_features = [
    "CourseCategory",
    "CourseType",
    "CourseLevel",
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "TeacherRating",
    "YearsOfExperience",
    "Expertise"
]

# ============================================================
# 5. DAY 13 ENGINEERED FEATURES
# ============================================================

engineered_features = [
    "PricePerDay",
    "PriceSquared",
    "DurationSquared",
    "LogCoursePrice",
    "LogCourseDuration",
    "RatingGap",
    "AverageRating",
    "CourseQualityScore",
    "ExperienceRatingScore",
    "PriceRatingInteraction",
    "PricePerRatingPoint",
    "Category_Type",
    "Category_Level",
    "Type_Level",
    "Category_Expertise",
    "Level_Expertise",
    "ExperienceBand"
]

# ============================================================
# 6. COMPLETE ENROLLMENT MODEL FEATURES
# ============================================================

all_modeling_features = (
    original_features + engineered_features
)

# ============================================================
# 7. DAY 13 FEATURE ENGINEERING FUNCTION
# ============================================================

def create_engineered_features(df):
    df = df.copy()

    # --------------------------------------------------------
    # Numerical engineered features
    # --------------------------------------------------------
    df["PricePerDay"] = np.where(
        df["CourseDuration"] != 0,
        df["CoursePrice"] / df["CourseDuration"],
        0
    )

    df["PriceSquared"] = (
        df["CoursePrice"] ** 2
    )

    df["DurationSquared"] = (
        df["CourseDuration"] ** 2
    )

    df["LogCoursePrice"] = np.log1p(
        df["CoursePrice"]
    )

    df["LogCourseDuration"] = np.log1p(
        df["CourseDuration"]
    )

    df["RatingGap"] = (
        df["CourseRating"] -
        df["TeacherRating"]
    )

    df["AverageRating"] = (
        df["CourseRating"] +
        df["TeacherRating"]
    ) / 2

    df["CourseQualityScore"] = (
        df["CourseRating"] *
        df["TeacherRating"]
    )

    df["ExperienceRatingScore"] = (
        df["YearsOfExperience"] *
        df["TeacherRating"]
    )

    df["PriceRatingInteraction"] = (
        df["CoursePrice"] *
        df["CourseRating"]
    )

    df["PricePerRatingPoint"] = np.where(
        df["CourseRating"] != 0,
        df["CoursePrice"] / df["CourseRating"],
        0
    )

    # --------------------------------------------------------
    # Categorical interaction features
    # --------------------------------------------------------
    df["Category_Type"] = (
        df["CourseCategory"].astype(str)
        + "_"
        + df["CourseType"].astype(str)
    )

    df["Category_Level"] = (
        df["CourseCategory"].astype(str)
        + "_"
        + df["CourseLevel"].astype(str)
    )

    df["Type_Level"] = (
        df["CourseType"].astype(str)
        + "_"
        + df["CourseLevel"].astype(str)
    )

    df["Category_Expertise"] = (
        df["CourseCategory"].astype(str)
        + "_"
        + df["Expertise"].astype(str)
    )

    df["Level_Expertise"] = (
        df["CourseLevel"].astype(str)
        + "_"
        + df["Expertise"].astype(str)
    )

    # --------------------------------------------------------
    # Experience band
    # --------------------------------------------------------
    df["ExperienceBand"] = pd.cut(
        df["YearsOfExperience"],
        bins=[-np.inf, 5, 10, 20, np.inf],
        labels=[
            "Early",
            "Developing",
            "Experienced",
            "Highly_Experienced"
        ]
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------
    df["PricePerRatingPoint"] = (
        df["PricePerRatingPoint"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return df

# ============================================================
# 8. DEMAND CLASSIFICATION
# ============================================================

def classify_demand(enrollment_prediction):
    if enrollment_prediction < 150:
        return "Low"
    elif enrollment_prediction < 170:
        return "Moderate"
    elif enrollment_prediction < 185:
        return "High"
    else:
        return "Very High"

# ============================================================
# 9. DEMAND BUSINESS RECOMMENDATION
# ============================================================

def generate_recommendation(demand_level):
    if demand_level == "Low":
        return (
            "Consider reviewing course positioning, pricing, "
            "content, and instructor strategy before launch."
        )
    elif demand_level == "Moderate":
        return (
            "Demand appears moderate. Consider launching with "
            "targeted promotion and monitor early enrollment."
        )
    elif demand_level == "High":
        return (
            "Demand appears high. The course is a strong "
            "candidate for launch with appropriate instructor "
            "and marketing capacity."
        )
    else:
        return (
            "Demand appears very high. Prioritize launch planning "
            "and ensure sufficient instructor and platform capacity."
        )

# ============================================================
# 10. REVENUE CLASSIFICATION
# ============================================================

def classify_revenue(revenue):
    if revenue < 10000:
        return "Low"
    elif revenue < 25000:
        return "Moderate"
    elif revenue < 50000:
        return "High"
    else:
        return "Very High"

# ============================================================
# 11. REVENUE BUSINESS RECOMMENDATION
# ============================================================

def generate_revenue_recommendation(revenue_level):
    if revenue_level == "Low":
        return (
            "Review pricing, course positioning, target audience "
            "and promotional strategy before launch."
        )
    elif revenue_level == "Moderate":
        return (
            "Consider targeted marketing and monitor early "
            "revenue performance after launch."
        )
    elif revenue_level == "High":
        return (
            "Strong revenue opportunity. Prioritize launch planning "
            "and prepare appropriate instructor and platform capacity."
        )
    else:
        return (
            "Very strong predicted revenue potential. Prioritize "
            "launch planning and prepare sufficient instructor, "
            "marketing and platform capacity."
        )

# ============================================================
# 12. PREDICTION FUNCTION
# ============================================================

def predict_course(course_input):
    # --------------------------------------------------------
    # Revenue model uses original 9 features
    # --------------------------------------------------------
    revenue_input = course_input[
        original_features
    ].copy()

    # --------------------------------------------------------
    # Enrollment model uses Day 13 engineered features
    # --------------------------------------------------------
    engineered_input = create_engineered_features(
        course_input
    )

    # --------------------------------------------------------
    # Enrollment prediction
    # --------------------------------------------------------
    enrollment_prediction = enrollment_model.predict(
        engineered_input[all_modeling_features]
    )[0]

    # --------------------------------------------------------
    # Revenue prediction
    # --------------------------------------------------------
    revenue_prediction = revenue_model.predict(
        revenue_input
    )[0]

    # --------------------------------------------------------
    # Prevent negative predictions
    # --------------------------------------------------------
    enrollment_prediction = max(
        0,
        enrollment_prediction
    )

    revenue_prediction = max(
        0,
        revenue_prediction
    )

    return {
        "EnrollmentCount": enrollment_prediction,
        "CourseRevenue": revenue_prediction
    }

# ============================================================
# 13. HEADER
# ============================================================

st.title("📊 EduPro Predictive Modeling Dashboard")
st.caption("Course Demand & Revenue Forecasting")

st.divider()

# ============================================================
# 14. COURSE INPUT SECTION
# ============================================================

st.header("🎓 Course Information")
col1, col2, col3 = st.columns(3)

# ============================================================
# CATEGORICAL INPUTS
# ============================================================

with col1:
    # --------------------------------------------------------
    # Course Category
    # --------------------------------------------------------
    course_category = st.selectbox(
        "Course Category",
        [
            "Select Category",
            "Artificial Intelligence",
            "Data Science",
            "Design",
            "Machine Learning",
            "Project Management",
            "Marketing",
            "Digital Marketing",
            "Web Development",
            "Programming",
            "Business",
            "Finance",
            "Cybersecurity"
        ],
        index=0
    )

    # --------------------------------------------------------
    # Course Type
    # --------------------------------------------------------
    course_type = st.selectbox(
        "Course Type",
        [
            "Select Type",
            "Online",
            "Offline",
            "Hybrid"
        ],
        index=0
    )

    # --------------------------------------------------------
    # Course Level
    # --------------------------------------------------------
    course_level = st.selectbox(
        "Course Level",
        [
            "Select Level",
            "Beginner",
            "Intermediate",
            "Advanced"
        ],
        index=0
    )

# ============================================================
# NUMERICAL INPUTS
# ============================================================

with col2:
    # --------------------------------------------------------
    # Course Price
    # --------------------------------------------------------
    course_price = st.number_input(
        "Course Price",
        min_value=0.0,
        value=0.0,
        step=10.0
    )

    # --------------------------------------------------------
    # Course Duration
    # --------------------------------------------------------
    course_duration = st.number_input(
        "Course Duration",
        min_value=0.0,
        value=0.0,
        step=0.5
    )

    # --------------------------------------------------------
    # Course Rating
    # --------------------------------------------------------
    course_rating = st.number_input(
        "Course Rating",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.01
    )

# ============================================================
# TEACHER INPUTS / EXPERTISE
# ============================================================

with col3:
    # --------------------------------------------------------
    # Teacher Rating
    # --------------------------------------------------------
    teacher_rating = st.number_input(
        "Teacher Rating",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.01
    )

    # --------------------------------------------------------
    # Years of Experience
    # --------------------------------------------------------
    years_experience = st.number_input(
        "Years of Experience",
        min_value=0,
        value=0,
        step=1
    )

    # --------------------------------------------------------
    # Expertise
    # --------------------------------------------------------
    expertise = st.selectbox(
        "Expertise",
        [
            "Select Category",
            "Artificial Intelligence",
            "Data Science",
            "Design",
            "Machine Learning",
            "Project Management",
            "Marketing",
            "Digital Marketing",
            "Web Development",
            "Programming",
            "Business",
            "Finance",
            "Cybersecurity"
        ],
        index=0
    )

# ============================================================
# 15. PREDICTION BUTTON
# ============================================================

st.divider()
predict_button = st.button(
    "🔮 Predict Enrollment & Revenue",
    type="primary",
    use_container_width=True
)

# ============================================================
# 16. RUN PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CLEAR PREVIOUS RESULTS
    # --------------------------------------------------------
    st.session_state.pop("prediction", None)
    st.session_state.pop("demand_level", None)
    st.session_state.pop("recommendation", None)
    st.session_state.pop("revenue_level", None)
    st.session_state.pop("revenue_recommendation", None)
    st.session_state.pop("revenue_per_enrollment", None)
    st.session_state.pop("course_input", None)

    # --------------------------------------------------------
    # VALIDATION FLAG
    # --------------------------------------------------------
    validation_failed = False

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================
    if (course_category == "Select Category" or course_category == "" or course_category is None):
        st.warning("⚠️ Please select a Course Category.")
        validation_failed = True

    if (course_type == "Select Type" or course_type == "" or course_type is None):
        st.warning("⚠️ Please select a Course Type.")
        validation_failed = True

    if (course_level == "Select Level" or course_level == "" or course_level is None):
        st.warning("⚠️ Please select a Course Level.")
        validation_failed = True

    if course_duration <= 0:
        st.warning("⚠️ Please enter a Course Duration greater than 0.")
        validation_failed = True

    if course_rating <= 0:
        st.warning("⚠️ Please enter a Course Rating greater than 0.")
        validation_failed = True

    if teacher_rating <= 0:
        st.warning("⚠️ Please enter a Teacher Rating greater than 0.")
        validation_failed = True

    if years_experience <= 0:
        st.warning("⚠️ Please enter Years of Experience greater than 0.")
        validation_failed = True

    if (expertise == "Select Expertise" or expertise == "Select Category" or expertise == "" or expertise is None):
        st.warning("⚠️ Please select an Expertise.")
        validation_failed = True

    # ========================================================
    # STOP PREDICTION IF ANY VALIDATION FAILED
    # ========================================================
    if validation_failed:
        st.info(
            "ℹ️ Prediction was not performed. "
            "Please correct all required inputs above "
            "and click the prediction button again."
        )

    # ========================================================
    # PREDICTION ONLY AFTER ALL VALIDATION PASSES
    # ========================================================
    else:
        # ----------------------------------------------------
        # Create input dataframe
        # ----------------------------------------------------
        course_input = pd.DataFrame([{
            "CourseCategory": course_category,
            "CourseType": course_type,
            "CourseLevel": course_level,
            "CoursePrice": course_price,
            "CourseDuration": course_duration,
            "CourseRating": course_rating,
            "TeacherRating": teacher_rating,
            "YearsOfExperience": years_experience,
            "Expertise": expertise
        }])

        # ----------------------------------------------------
        # Generate predictions
        # ----------------------------------------------------
        try:
            prediction = predict_course(course_input)
            predicted_enrollment = prediction["EnrollmentCount"]
            predicted_revenue = prediction["CourseRevenue"]

            # ------------------------------------------------
            # Demand classification & Recommendation
            # ------------------------------------------------
            demand_level = classify_demand(predicted_enrollment)
            recommendation = generate_recommendation(demand_level)

            # ------------------------------------------------
            # Revenue classification & Recommendation
            # ------------------------------------------------
            revenue_level = classify_revenue(predicted_revenue)
            revenue_recommendation = generate_revenue_recommendation(revenue_level)

            # ------------------------------------------------
            # Revenue per expected enrollment
            # ------------------------------------------------
            if predicted_enrollment > 0:
                revenue_per_enrollment = (predicted_revenue / predicted_enrollment)
            else:
                revenue_per_enrollment = 0

            # ------------------------------------------------
            # Store results in session state
            # ------------------------------------------------
            st.session_state["prediction"] = prediction
            st.session_state["demand_level"] = demand_level
            st.session_state["recommendation"] = recommendation
            st.session_state["revenue_level"] = revenue_level
            st.session_state["revenue_recommendation"] = revenue_recommendation
            st.session_state["revenue_per_enrollment"] = revenue_per_enrollment
            st.session_state["course_input"] = course_input

            st.success("✅ Prediction completed successfully.")

        except Exception as e:
            # ------------------------------------------------
            # APPLICATION ERROR PROTECTION
            # ------------------------------------------------
            st.error(f"❌ Prediction failed unexpectedly: {e}")
            st.info(
                "ℹ️ Please verify the entered course information "
                "and try again. The application remained running "
                "without crashing."
            )

# ============================================================
# 17. DISPLAY RESULTS
# ============================================================

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]
    predicted_enrollment = prediction["EnrollmentCount"]
    predicted_revenue = prediction["CourseRevenue"]
    demand_level = st.session_state["demand_level"]
    recommendation = st.session_state["recommendation"]
    revenue_level = st.session_state["revenue_level"]
    revenue_recommendation = st.session_state["revenue_recommendation"]
    revenue_per_enrollment = st.session_state["revenue_per_enrollment"]
    course_input = st.session_state["course_input"]

    # ========================================================
    # OVERALL PREDICTION RESULTS
    # ========================================================
    st.divider()
    st.header("🎯 Prediction Results — Enrollment & Revenue")

    # --------------------------------------------------------
    # ENROLLMENT PREDICTION
    # --------------------------------------------------------
    st.subheader("📈 Enrollment Prediction")
    enrollment_col1, enrollment_col2 = st.columns(2)

    with enrollment_col1:
        st.metric(
            "Predicted Enrollment",
            f"{predicted_enrollment:,.2f} students"
        )

    with enrollment_col2:
        st.metric(
            "Demand Level",
            demand_level
        )

    # --------------------------------------------------------
    # REVENUE PREDICTION
    # --------------------------------------------------------
    st.subheader("💰 Revenue Prediction")
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric(
            "Predicted Course Revenue",
            f"₹{predicted_revenue:,.2f}"
        )

    with kpi2:
        st.metric(
            "Expected Enrollment",
            f"{predicted_enrollment:,.2f}"
        )

    with kpi3:
        st.metric(
            "Revenue / Expected Enrollment",
            f"₹{revenue_per_enrollment:,.2f}"
        )

    # ========================================================
    # ENROLLMENT POTENTIAL
    # ========================================================
    st.subheader("📈 Enrollment Potential")

    if demand_level == "Very High":
        st.success(f"Enrollment Potential: {demand_level}")
    elif demand_level == "High":
        st.info(f"Enrollment Potential: {demand_level}")
    elif demand_level == "Moderate":
        st.warning(f"Enrollment Potential: {demand_level}")
    else:
        st.error(f"Enrollment Potential: {demand_level}")

    # ========================================================
    # REVENUE POTENTIAL
    # ========================================================
    st.subheader("📊 Revenue Potential")

    if revenue_level == "Very High":
        st.success(f"Revenue Potential: {revenue_level}")
    elif revenue_level == "High":
        st.info(f"Revenue Potential: {revenue_level}")
    elif revenue_level == "Moderate":
        st.warning(f"Revenue Potential: {revenue_level}")
    else:
        st.error(f"Revenue Potential: {revenue_level}")

    # ========================================================
    # REVENUE INTERPRETATION
    # ========================================================
    st.subheader("🔎 Revenue Interpretation")

    st.info(
        f"""
        The model predicts approximately
        **₹{predicted_revenue:,.2f} in course revenue**.

        Based on the dashboard revenue thresholds, this
        corresponds to **{revenue_level} revenue potential**.
        """
    )

    # ========================================================
    # REVENUE RECOMMENDATION
    # ========================================================
    st.subheader("💡 Revenue Recommendation")
    st.success(revenue_recommendation)

    # ========================================================
    # PRICING INSIGHT
    # ========================================================
    st.subheader("💵 Pricing Insight")

    st.write(
        f"**Proposed Course Price:** "
        f"₹{course_input['CoursePrice'].iloc[0]:,.2f}"
    )
    st.write(
        f"**Predicted Course Revenue:** "
        f"₹{predicted_revenue:,.2f}"
    )
    st.write(
        f"**Revenue per Expected Enrollment:** "
        f"₹{revenue_per_enrollment:,.2f}"
    )

    st.caption(
        "Revenue is a model prediction based on the submitted "
        "course characteristics. It should not be interpreted "
        "as a causal estimate of the effect of changing price."
    )

    # ========================================================
    # DEMAND RESULTS
    # ========================================================
    st.divider()
    st.header("📈 Demand Analysis")

    demand_col1, demand_col2 = st.columns(2)

    with demand_col1:
        st.metric(
            "Predicted Enrollment",
            f"{predicted_enrollment:.2f}"
        )

    with demand_col2:
        st.metric(
            "Demand Level",
            demand_level
        )

    # ========================================================
    # DEMAND INTERPRETATION
    # ========================================================
    st.subheader("🔎 Demand Interpretation")

    st.info(
        f"""
        The model predicts approximately
        **{predicted_enrollment:.2f} enrollments**.

        Based on the dashboard demand thresholds, this
        corresponds to **{demand_level} demand**.
        """
    )

    # ========================================================
    # DEMAND BUSINESS RECOMMENDATION
    # ========================================================
    st.subheader("💡 Demand Recommendation")
    st.success(recommendation)

    # ========================================================
    # PREDICTION INPUT SUMMARY
    # ========================================================
    st.subheader("📋 Prediction Input Summary")

    display_input = course_input.T.reset_index()
    display_input.columns = ["Feature", "Value"]

    st.dataframe(
        display_input,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================
    st.subheader("🤖 Model Information")

    model_col1, model_col2 = st.columns(2)

    with model_col1:
        st.write("**Enrollment Prediction Model**")
        st.write("EduPro_Final_Enrollment_Model.joblib")
        st.caption(
            "Final Day 14 Enrollment Model using "
            "the original and engineered features."
        )

    with model_col2:
        st.write("**Revenue Prediction Model**")
        st.write("EduPro_Final_Revenue_Model.joblib")
        st.caption(
            "Final Day 14 Revenue Model using "
            "the original course features."
        )

    # ========================================================
    # MODEL STATUS
    # ========================================================
    st.subheader("✅ Prediction System Status")

    status_col1, status_col2 = st.columns(2)

    with status_col1:
        st.success("Enrollment model loaded successfully.")

    with status_col2:
        st.success("Revenue model loaded successfully.")

    st.caption(
        "Predictions are generated using the saved final models. "
        "The dashboard does not retrain the models."
    )

# ============================================================
# 18. FOOTER
# ============================================================

st.divider()
st.caption(
    "EduPro Predictive Modeling Project — "
    "Course Demand & Revenue Forecasting Dashboard"
)