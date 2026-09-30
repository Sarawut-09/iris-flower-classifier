from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

st.set_page_config(
    page_title="Iris flower classifier",
    page_icon=":material/local_florist:",
    layout="wide",
)

MODEL_PATH = Path(__file__).with_name("iris_model.pkl")
CLASS_NAMES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
FEATURE_NAMES = ["Sepal length", "Sepal width", "Petal length", "Petal width"]


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


@st.cache_data
def load_reference_averages():
    iris = load_iris(as_frame=True)
    return iris.data.mean().to_numpy()


if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)
reference_averages = load_reference_averages()

with st.sidebar:
    st.title("Iris Learn")
    st.caption("Input features")
    st.write("Adjust the sliders to change the flower measurements.")

    with st.form("iris_prediction"):
        sepal_length = st.slider("Sepal length (cm)", 4.3, 7.9, 6.0, 0.1)
        sepal_width = st.slider("Sepal width (cm)", 2.0, 4.4, 2.9, 0.1)
        petal_length = st.slider("Petal length (cm)", 1.0, 6.9, 4.5, 0.1)
        petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.5, 0.1)
        st.form_submit_button(
            "Predict species",
            icon=":material/search:",
            width="stretch",
        )

input_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=model.feature_names_in_,
)
predicted_class = int(model.predict(input_data)[0])
probabilities = model.predict_proba(input_data)[0]
confidence = float(probabilities[list(model.classes_).index(predicted_class)])
predicted_species = CLASS_NAMES[predicted_class]

st.title("Iris flower classifier")
st.caption("Explore flower measurements and see how the trained model classifies each Iris species.")

chart_column, result_column = st.columns([1.15, 0.85], gap="large")

with chart_column:
    st.subheader("Your input vs. dataset average")
    comparison = pd.DataFrame(
        {
            "Your input": [sepal_length, sepal_width, petal_length, petal_width],
            "Dataset average": reference_averages,
        },
        index=FEATURE_NAMES,
    )
    st.bar_chart(
        comparison,
        height=300,
        y_label="Measurement (cm)",
        sort=False,
        stack=False,
    )

with result_column:
    with st.container(border=True):
        st.caption("PREDICTED SPECIES")
        st.subheader(f"Iris {predicted_species}")
        st.metric("Confidence", f"{confidence:.1%}")

    st.subheader("Probability distribution")
    probability_data = pd.DataFrame(
        {
            "Species": [CLASS_NAMES[int(class_id)] for class_id in model.classes_],
            "Probability": probabilities,
        }
    )
    st.bar_chart(
        probability_data,
        x="Species",
        y="Probability",
        y_label="Probability",
        height=220,
    )

st.divider()
st.subheader("About Iris species")

species_columns = st.columns(3)
species_details = [
    ("Setosa", "Small petals", "Distinct measurements", "Usually easy to identify"),
    ("Versicolor", "Medium-sized petals", "Intermediate features", "Often overlaps with Virginica"),
    ("Virginica", "Largest petals", "Longer, wider petals", "Typically separated by petal size"),
]

for column, (species, detail_one, detail_two, detail_three) in zip(
    species_columns, species_details
):
    with column, st.container(border=True):
        st.markdown(f"### Iris {species}")
        st.markdown(f"- {detail_one}\n- {detail_two}\n- {detail_three}")