import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score

MAP_NHAN = {"II": 0, "III": 1, "IV": 2}
TEN_LOP = ["II", "III", "IV"]

X_train = pd.read_csv("X_train_clean.csv")
X_test = pd.read_csv("X_test_clean.csv")
y_train_raw = pd.read_csv("y_train.csv").iloc[:, 0]
y_test_raw = pd.read_csv("y_test.csv").iloc[:, 0]
y_train = y_train_raw.map(MAP_NHAN)
y_test = y_test_raw.map(MAP_NHAN)

# 1. Phân nhóm 139 cột theo chuyên khoa (dùng chung cho cả Stacking và UI Tabs)
nhom_lam_sang = [
    "admission.ward", "admission.way", "visit.times",
    "body.temperature", "pulse", "respiration",
    "systolic.blood.pressure", "diastolic.blood.pressure", "map",
    "type.of.heart.failure", "Killip.grade", "type.II.respiratory.failure",
    "consciousness", "eye.opening", "verbal.response", "movement", "GCS",
    "respiratory.support.", "oxygen.inhalation", "fio2"
]
nhom_sieu_am = [
    "LVEF", "left.ventricular.end.diastolic.diameter.LV",
    "mitral.valve.EMS", "mitral.valve.AMS", "tricuspid.valve.return.velocity"
]
nhom_benh_nen = [
    "occupation", "gender", "ageCat", "weight", "height", "BMI",
    "myocardial.infarction", "congestive.heart.failure", "peripheral.vascular.disease",
    "cerebrovascular.disease", "dementia", "Chronic.obstructive.pulmonary.disease",
    "connective.tissue.disease", "peptic.ulcer.disease", "diabetes",
    "moderate.to.severe.chronic.kidney.disease", "hemiplegia", "malignant.lymphoma",
    "solid.tumor", "liver.disease", "AIDS", "CCI.score", "acute.renal.failure"
]
da_chia = set(nhom_lam_sang + nhom_sieu_am + nhom_benh_nen)
nhom_xet_nghiem = [c for c in X_train.columns if c not in da_chia]

nhom_cot = {
    "Lâm sàng & Sinh hiệu": nhom_lam_sang,
    "Siêu âm tim": nhom_sieu_am,
    "Bệnh nền & Nhân trắc": nhom_benh_nen,
    "Xét nghiệm máu & Sinh hóa": nhom_xet_nghiem,
}

cot_ordinal = {
    "Killip.grade": ["I", "II", "III", "IV"],
    "ageCat": ["(21,29]", "(29,39]", "(39,49]", "(49,59]", "(59,69]",
               "(69,79]", "(79,89]", "(89,110]"],
}

# 2. Huấn luyện mô hình Stacking 2 tầng
def tao_nhanh_cay(danh_sach_cot, max_depth=2, max_iter=150, min_samples_leaf=30, l2=5.0):
    c_chu = [c for c in danh_sach_cot if not pd.api.types.is_numeric_dtype(X_train[c])]
    c_so = [c for c in danh_sach_cot if c not in c_chu]
    c_ord = [c for c in c_chu if c in cot_ordinal]
    c_nom = [c for c in c_chu if c not in cot_ordinal]

    steps = []
    if c_so:
        steps.append(("so", "passthrough", c_so))
    if c_ord:
        steps.append(("ord", OrdinalEncoder(
            categories=[cot_ordinal[c] for c in c_ord],
            handle_unknown="use_encoded_value", unknown_value=np.nan), c_ord))
    if c_nom:
        steps.append(("nom", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan), c_nom))

    prep = ColumnTransformer(steps, remainder="drop").set_output(transform="pandas")
    cat_cols = [f"nom__{c}" for c in c_nom] if c_nom else None
    clf = HistGradientBoostingClassifier(
        class_weight="balanced", max_depth=max_depth, learning_rate=0.05,
        max_iter=max_iter, min_samples_leaf=min_samples_leaf, l2_regularization=l2,
        categorical_features=cat_cols, random_state=42
    )
    return Pipeline([("prep", prep), ("m", clf)])

def tao_nhanh_tuyen_tinh(danh_sach_cot):
    c_chu = [c for c in danh_sach_cot if not pd.api.types.is_numeric_dtype(X_train[c])]
    c_so = [c for c in danh_sach_cot if c not in c_chu]
    c_ord = [c for c in c_chu if c in cot_ordinal]
    c_nom = [c for c in c_chu if c not in cot_ordinal]

    steps = []
    if c_so:
        steps.append(("so", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", RobustScaler())]), c_so))
    if c_ord:
        steps.append(("ord", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(categories=[cot_ordinal[c] for c in c_ord]))]), c_ord))
    if c_nom:
        steps.append(("nom", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore"))]), c_nom))

    prep = ColumnTransformer(steps, remainder="drop")
    return Pipeline([("prep", prep),
                     ("m", LogisticRegression(class_weight="balanced", C=0.1, max_iter=5000))])

cv = StratifiedKFold(5, shuffle=True, random_state=42)
model = StackingClassifier(
    estimators=[
        ("lam_sang", tao_nhanh_cay(nhom_lam_sang, max_depth=2, max_iter=120)),
        ("sieu_am", tao_nhanh_cay(nhom_sieu_am, max_depth=2, max_iter=100, min_samples_leaf=20)),
        ("xet_nghiem", tao_nhanh_cay(nhom_xet_nghiem, max_depth=2, max_iter=150)),
        ("benh_nen", tao_nhanh_tuyen_tinh(nhom_benh_nen)),
        ("tong_hop", tao_nhanh_cay(X_train.columns.tolist(), max_depth=2, max_iter=150)),
    ],
    final_estimator=LogisticRegression(class_weight="balanced", C=0.5, max_iter=1000),
    cv=cv,
    stack_method="predict_proba",
    n_jobs=-1
)
model.fit(X_train, y_train)
pred_test = model.predict(X_test)
print("Test balanced_acc:", round(balanced_accuracy_score(y_test, pred_test), 3),
      "| macro-F1:", round(f1_score(y_test, pred_test, average="macro"), 3))

# 3. Tạo đặc tả UI cho TOÀN BỘ 139 cột
cot_chu = X_train.select_dtypes(exclude="number").columns.tolist()
cot_so = X_train.select_dtypes(include="number").columns.tolist()
thieu = X_train.isna().mean()

mac_dinh = {}
ui_spec = {}
for c in X_train.columns:
    ti_le_na = float(thieu[c])
    if c in cot_chu:
        mode_val = X_train[c].mode().iloc[0]
        mac_dinh[c] = None if ti_le_na > 0.30 else mode_val
        ui_spec[c] = {
            "kieu": "chon",
            "lua_chon": sorted(X_train[c].dropna().unique().tolist()),
            "mac_dinh": mode_val,
            "ti_le_thieu": ti_le_na,
        }
    elif set(X_train[c].dropna().unique()) <= {0, 1}:
        mode_val = int(X_train[c].mode().iloc[0])
        mac_dinh[c] = None if ti_le_na > 0.30 else mode_val
        ui_spec[c] = {
            "kieu": "chon",
            "lua_chon": [0, 1],
            "mac_dinh": mode_val,
            "ti_le_thieu": ti_le_na,
        }
    else:
        med_val = float(X_train[c].median())
        is_int = pd.api.types.is_integer_dtype(X_train[c])
        mac_dinh[c] = None if ti_le_na > 0.30 else (int(round(med_val)) if is_int else med_val)
        ui_spec[c] = {
            "kieu": "so",
            "la_so_nguyen": bool(is_int),
            "min": float(X_train[c].quantile(0.01)),
            "max": float(X_train[c].quantile(0.99)),
            "mac_dinh": int(round(med_val)) if is_int else med_val,
            "ti_le_thieu": ti_le_na,
        }

# Lưu 100 ca từ tập test (kèm nhãn thật) để test chức năng lấy ca ngẫu nhiên trên UI
mau_test = X_test.head(100).copy()
mau_test["__nhan_that__"] = y_test_raw.head(100).values

joblib.dump({
    "model": model,
    "cot_dau_vao": X_train.columns.tolist(),
    "cot_chu": cot_chu,
    "nhom_cot": nhom_cot,
    "mac_dinh": mac_dinh,
    "ui_spec": ui_spec,
    "mau_test": mau_test.to_dict(orient="records"),
    "ten_lop": TEN_LOP,
    "phien_ban_sklearn": sklearn.__version__,
}, "nyha_model.joblib")
print("Đã lưu nyha_model.joblib với đầy đủ", len(X_train.columns), "cột.")