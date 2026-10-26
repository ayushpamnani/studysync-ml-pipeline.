import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="StudySync ML Data Pipeline",
    page_icon="📊",
    layout="wide"
)

st.title("📊 StudySync ML — Data Pipeline")
st.write(
    "Upload student data, clean it, preprocess features, "
    "train machine learning models, and compare their performance."
)

st.info(
    "Pipeline: CSV Upload → Validation → Cleaning → "
    "Preprocessing → Model Training → Evaluation"
)

uploaded_file = st.file_uploader(
    "Upload your student performance CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        # 1. Data ingestion
        st.header("1. Data Ingestion")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.dataframe(df.head(10), use_container_width=True)

        # 2. Data quality analysis
        st.header("2. Data Quality Analysis")

        missing = df.isnull().sum()

        quality_report = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": missing.values,
            "Missing Percentage": (
                missing.values / max(len(df), 1) * 100
            ).round(2)
        })

        st.dataframe(quality_report, use_container_width=True)

        # 3. Data cleaning
        st.header("3. Data Cleaning")

        clean_df = df.drop_duplicates().copy()

        # Convert infinite numerical values to missing values.
        numeric_columns = clean_df.select_dtypes(
            include=np.number
        ).columns

        clean_df[numeric_columns] = clean_df[
            numeric_columns
        ].replace([np.inf, -np.inf], np.nan)

        st.write("Rows after removing duplicates:", len(clean_df))

        # 4. Select prediction target
        st.header("4. Select Prediction Target")

        numeric_targets = clean_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if not numeric_targets:
            st.error(
                "No numerical target column found. "
                "Please upload a dataset with a numerical column "
                "such as exam_score."
            )
            st.stop()

        default_target = (
            "exam_score"
            if "exam_score" in numeric_targets
            else numeric_targets[-1]
        )

        target = st.selectbox(
            "Which column should the model predict?",
            numeric_targets,
            index=numeric_targets.index(default_target)
        )

        # Remove rows where the target is missing or infinite.
        clean_df = clean_df.dropna(subset=[target])

        if len(clean_df) < 20:
            st.error(
                "At least 20 usable rows with known target values "
                "are required for this demonstration."
            )
            st.stop()

        # Exclude target and constant columns from the features.
        X = clean_df.drop(columns=[target])
        y = clean_df[target]

        usable_columns = [
            col for col in X.columns
            if X[col].nunique(dropna=True) > 1
        ]

        X = X[usable_columns]

        if X.shape[1] == 0:
            st.error(
                "No usable feature columns remain. "
                "Upload a dataset with multiple informative columns."
            )
            st.stop()

        # 5. Split data before learning preprocessing statistics.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        numeric_features = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        # 6. Data preprocessing pipeline
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        transformers = []

        if numeric_features:
            transformers.append(
                ("numeric", numeric_pipeline, numeric_features)
            )

        if categorical_features:
            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers
        )

        st.header("5. Preprocessing")

        st.write("Numerical features:", numeric_features)
        st.write("Categorical features:", categorical_features)

        st.success(
            "Preprocessing pipeline configured. "
            "Imputation, scaling, and encoding will be fitted "
            "using training data only."
        )

        # 7. Train and compare models
        st.header("6. Train and Compare Models")

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42
            )
        }

        if st.button("Train and Compare Models"):

            results = []
            trained_models = {}

            progress = st.progress(0)

            for index, (name, estimator) in enumerate(
                models.items()
            ):

                pipeline = Pipeline([
                    ("preprocessing", preprocessor),
                    ("model", estimator)
                ])

                pipeline.fit(X_train, y_train)

                predictions = pipeline.predict(X_test)

                mae = mean_absolute_error(
                    y_test, predictions
                )

                rmse = np.sqrt(
                    mean_squared_error(y_test, predictions)
                )

                r2 = r2_score(y_test, predictions)

                results.append({
                    "Model": name,
                    "MAE": round(mae, 3),
                    "RMSE": round(rmse, 3),
                    "R2 Score": round(r2, 3)
                })

                trained_models[name] = pipeline

                progress.progress(
                    (index + 1) / len(models)
                )

            results_df = pd.DataFrame(results)

            st.subheader("Model Evaluation Results")

            st.dataframe(
                results_df,
                use_container_width=True
            )

            best_model_name = results_df.loc[
                results_df["RMSE"].idxmin(),
                "Model"
            ]

            st.success(
                f"Lowest test RMSE: {best_model_name}"
            )

            st.bar_chart(
                results_df.set_index("Model")[
                    ["MAE", "RMSE"]
                ]
            )

            st.subheader("7. Sample Predictions")

            best_model = trained_models[best_model_name]

            prediction_results = pd.DataFrame({
                "Actual": y_test.to_numpy(),
                "Predicted": best_model.predict(
                    X_test
                )
            })

            st.dataframe(
                prediction_results.head(10),
                use_container_width=True
            )

            csv_output = prediction_results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Prediction Results",
                data=csv_output,
                file_name="studySync_predictions.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error(
            f"An error occurred while processing the CSV: {error}"
        )

else:
    st.warning(
        "Upload a CSV file to start the data pipeline."
    )
