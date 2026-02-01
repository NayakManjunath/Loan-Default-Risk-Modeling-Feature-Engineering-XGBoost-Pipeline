import streamlit as st
import pandas as pd
from PIL import Image
import glob
import os   # <-- THIS WAS MISSING

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Default Model – SHAP Explainability",
    layout="wide"
)

st.title("📊 Credit Default Model – SHAP Explainability")

ARTIFACTS_DIR = "artifacts"

# --------------------------------------------------
# 1. Model Summary
# --------------------------------------------------
st.header("1. Model Summary")

c1, c2, c3 = st.columns(3)
c1.metric("Model", "XGBoost")
c2.metric("Problem Type", "Binary Classification")
c3.metric("Target", "Default")

# --------------------------------------------------
# 2. Global SHAP Summary
# --------------------------------------------------
st.header("2. Global Feature Importance (SHAP)")

summary_path = os.path.join(ARTIFACTS_DIR, "shap_summary.png")

if os.path.exists(summary_path):
    st.image(
        Image.open(summary_path),
        use_container_width=True
    )
else:
    st.warning("Global SHAP summary not found.")

# --------------------------------------------------
# 3. Feature Importance Table
# --------------------------------------------------
st.header("3. Top Driving Features")

importance_path = os.path.join(
    ARTIFACTS_DIR,
    "shap_feature_importance.csv"
)

if os.path.exists(importance_path):
    df_imp = pd.read_csv(importance_path)
    st.dataframe(df_imp, use_container_width=True)
else:
    st.warning("Feature importance CSV not found.")

# --------------------------------------------------
# 4. Feature Behavior (Dependence Plots)
# --------------------------------------------------
st.header("4. Feature Behavior")

dep_plots = sorted(
    glob.glob(os.path.join(ARTIFACTS_DIR, "shap_dependence_*.png"))
)

if dep_plots:
    cols = st.columns(2)
    for i, img_path in enumerate(dep_plots[:6]):
        with cols[i % 2]:
            st.image(
                Image.open(img_path),
                use_container_width=True
            )
else:
    st.info("No SHAP dependence plots found.")

# --------------------------------------------------
# 5. Drivers of Default Risk (OPTIONAL)
# --------------------------------------------------
st.header("5. Drivers of Default Risk")

pos_class_path = os.path.join(
    ARTIFACTS_DIR,
    "shap_summary_positive_class.png"
)

if os.path.exists(pos_class_path):
    st.image(
        Image.open(pos_class_path),
        use_container_width=True
    )
else:
    st.info(
        "Positive-class SHAP summary not found. "
        "This section is optional."
    )

# --------------------------------------------------
# # 6. Individual Prediction Explanations
# # --------------------------------------------------
# st.header("6. Individual Prediction Explanations")

# force_plots = sorted(
#     glob.glob(os.path.join(ARTIFACTS_DIR, "shap_force_row_*.png"))
# )

# if force_plots:
#     for img_path in force_plots[:3]:
#         st.image(
#             Image.open(img_path),
#             use_container_width=True
#         )
# else:
#     st.info("No individual SHAP force plots found.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption("Artifacts loaded from /artifacts")



# import streamlit as st
# import pandas as pd
# from PIL import Image
# import glob
# import os
# st.set_page_config(
#     page_title="Credit Default Model – Explainability",
#     layout="wide"
# )

# st.title("📊 Credit Default Model – SHAP Explainability")

# # ---------- Model Summary ----------
# st.header("1. Model Summary")

# col1, col2, col3 = st.columns(3)
# col1.metric("Model", "XGBoost")
# col2.metric("Problem Type", "Binary Classification")
# col3.metric("Target", "Default")

# # ---------- Global SHAP ----------
# st.header("2. Global Feature Importance (SHAP)")

# summary_img = Image.open("artifacts/shap_summary.png")
# st.image(summary_img, caption="Global SHAP Summary", use_container_width=True)

# # ---------- Feature Importance Table ----------
# st.header("3. Top Driving Features")

# imp_df = pd.read_csv("artifacts/shap_feature_importance.csv")
# st.dataframe(imp_df, use_container_width=True)

# # ---------- Dependence Plots ----------
# st.header("4. Feature Behavior")

# dep_plots = sorted(glob.glob("artifacts/shap_dependence_*.png"))

# cols = st.columns(2)
# for i, img_path in enumerate(dep_plots[:6]):
#     with cols[i % 2]:
#         st.image(Image.open(img_path), use_container_width=True)

# # ---------- Default Class Explanation ----------
# st.header("5. Drivers of Default Risk")

# pos_img = Image.open("artifacts/shap_dependence_{feature}.png")
# st.image(
#     pos_img,
#     caption="SHAP Summary – Default Class",
#     use_container_width=True
# )


# # ---------- Individual Explanations ----------
# st.header("6. Individual Prediction Explanations")

# force_plots = sorted(glob.glob("artifacts/shap_force_row_*.png"))

# for img in force_plots[:3]:
#     st.image(Image.open(img), use_container_width=True)

# # ---------- Footer ----------
# st.divider()
# st.caption("Artifacts loaded from /artifacts | Model version v1.0")
