import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              StackingClassifier)
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (confusion_matrix, f1_score, balanced_accuracy_score,
                             accuracy_score, cohen_kappa_score)

# 1. Đọc dữ liệu; mapping nhãn NYHA: II -> 0, III -> 1, IV -> 2
MAP_NHAN = {"II": 0, "III": 1, "IV": 2}
TEN_LOP = ["II", "III", "IV"]
X_train = pd.read_csv("X_train_clean.csv")
X_test = pd.read_csv("X_test_clean.csv")
y_train = pd.read_csv("y_train.csv").iloc[:, 0].map(MAP_NHAN)
y_test = pd.read_csv("y_test.csv").iloc[:, 0].map(MAP_NHAN)
assert y_train.notna().all() and y_test.notna().all(), "Nhãn có giá trị ngoài II/III/IV"
print("Phân bố train:", y_train.value_counts().sort_index().to_dict(),
      "| test:", y_test.value_counts().sort_index().to_dict())

# 2. Phân nhóm cột theo kiểu dữ liệu & thứ tự
cot_ordinal = {
    "Killip.grade": ["I", "II", "III", "IV"],
    "ageCat": ["(21,29]", "(29,39]", "(39,49]", "(49,59]", "(59,69]",
               "(69,79]", "(79,89]", "(89,110]"],
}
cot_chu = X_train.select_dtypes(exclude="number").columns.tolist()
cot_ord_list = [c for c in cot_chu if c in cot_ordinal]
cot_nominal = [c for c in cot_chu if c not in cot_ordinal]
cot_so = X_train.select_dtypes(include="number").columns.tolist()

# 3. Preprocessor cho các mô hình đơn lẻ (Baseline)
prep_A = ColumnTransformer([      # cho Logistic Regression & Random Forest
    ("so", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                     ("sc", RobustScaler())]), cot_so),
    ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("enc", OrdinalEncoder(categories=[cot_ordinal[c] for c in cot_ord_list]))]),
     cot_ord_list),
    ("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("enc", OneHotEncoder(handle_unknown="ignore"))]), cot_nominal),
])

prep_B = ColumnTransformer([      # cho HistGradientBoosting (giữ NaN số + tách rõ biến định danh)
    ("so", "passthrough", cot_so),
    ("ord", OrdinalEncoder(categories=[cot_ordinal[c] for c in cot_ord_list],
                           handle_unknown="use_encoded_value", unknown_value=np.nan), cot_ord_list),
    ("nom", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cot_nominal),
]).set_output(transform="pandas")
cat_cols_B = [f"nom__{c}" for c in cot_nominal] if cot_nominal else None

# 4. Chia 139 cột thành 4 nhóm chuyên khoa cho mô hình Stacking
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

stacking_model = StackingClassifier(
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

# 5. Danh sách tất cả mô hình để đối chiếu
models = {
    "Dummy (đoán lớp III)": Pipeline([("m", DummyClassifier(strategy="most_frequent"))]),
    "LogisticRegression": Pipeline([("prep", prep_A),
        ("m", LogisticRegression(class_weight="balanced", C=0.1, max_iter=5000))]),
    "RandomForest": Pipeline([("prep", prep_A),
        ("m", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     min_samples_leaf=3, random_state=42, n_jobs=-1))]),
    "HistGradientBoosting": Pipeline([("prep", prep_B),
        ("m", HistGradientBoostingClassifier(class_weight="balanced", max_depth=2,
                                             learning_rate=0.05, max_iter=150,
                                             min_samples_leaf=30, l2_regularization=5.0,
                                             categorical_features=cat_cols_B, random_state=42))]),
    "Stacking_ChuyenKhoa": stacking_model,
}

# 6. So sánh bằng cross-validation 5-fold trên TRAIN
scoring = {"accuracy": "accuracy", "balanced_acc": "balanced_accuracy", "macro_F1": "f1_macro",
           "macro_recall": "recall_macro", "macro_precision": "precision_macro",
           "roc_auc_ovr": "roc_auc_ovr"}
rows = {}
for ten, pipe in models.items():
    r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    rows[ten] = {k: r["test_" + k].mean() for k in scoring}

pd.set_option("display.width", 200)
print("\n=== Cross-validation 5-fold trên TRAIN ===")
print(pd.DataFrame(rows).T.round(3).to_string())

# 7. Đánh giá trên TEST: fit trên toàn bộ train, dự đoán test
print("\n=== Đánh giá trên TẬP TEST ===")
tong_ket = {}
for ten, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rec = [(pred[y_test == k] == k).mean() for k in range(3)]
    tong_ket[ten] = dict(accuracy=accuracy_score(y_test, pred),
                         balanced_acc=balanced_accuracy_score(y_test, pred),
                         macro_F1=f1_score(y_test, pred, average="macro"),
                         kappa_bac_hai=cohen_kappa_score(y_test, pred, weights="quadratic"),
                         recall_II=rec[0], recall_III=rec[1], recall_IV=rec[2])
print(pd.DataFrame(tong_ket).T.round(3).to_string())

# 8. Đánh giá riêng từng nhánh chuyên khoa bên trong Stacking trên TEST
print("\n=== Hiệu năng từng nhánh chuyên khoa (Tầng 1) trên TẬP TEST ===")
nhanh_ket = {}
for ten_nhanh, est in models["Stacking_ChuyenKhoa"].named_estimators_.items():
    pred_nhanh = est.predict(X_test)
    nhanh_ket[ten_nhanh] = dict(
        accuracy=accuracy_score(y_test, pred_nhanh),
        balanced_acc=balanced_accuracy_score(y_test, pred_nhanh),
        macro_F1=f1_score(y_test, pred_nhanh, average="macro"),
        kappa_bac_hai=cohen_kappa_score(y_test, pred_nhanh, weights="quadratic"),
    )
print(pd.DataFrame(nhanh_ket).T.round(3).to_string())

# 9. Ma trận nhầm lẫn trên TEST
for ten in ["LogisticRegression", "RandomForest", "HistGradientBoosting", "Stacking_ChuyenKhoa"]:
    pred = models[ten].predict(X_test)
    print(f"\n--- {ten} ---")
    print(pd.DataFrame(confusion_matrix(y_test, pred), index=[f"thật {c}" for c in TEN_LOP],
                       columns=[f"đoán {c}" for c in TEN_LOP]).to_string())