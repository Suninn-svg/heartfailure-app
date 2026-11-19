import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PATH = "heart_failure.csv"                        # đổi thành đường dẫn file của bạn
TARGET = "NYHA.cardiac.function.classification"   # nhãn: II / III / IV (mức độ suy tim)
RANDOM_STATE = 42
MISSING_THRESHOLD = 0.80                          # bỏ cột thiếu > 80%
COT_BO_THEM = []                                  # muốn bỏ thêm cột nào thì ghi tên vào đây

# 1. Đọc file
df = pd.read_csv(PATH)
print("Kích thước ban đầu:", df.shape)

# 2. Tách nhãn
assert df[TARGET].notna().all(), "Nhãn có giá trị thiếu"
y = df[TARGET]
print(y.value_counts(), "\n")
X = df.drop(columns=[TARGET])

# 3. Bỏ cột không dùng
cot_dinh_danh = ["Unnamed: 0", "inpatient.number"]
cot_hang_rong = ["leukemia", "cholinesterase", "body.temperature.blood.gas"]
cot_sau_xuat_vien = ["DestinationDischarge", "discharge.department", "dischargeDay",
                     "outcome.during.hospitalization"]
# các cột kết cục: death.within.*, re.admission.*, time.of.death..., return/time.to.emergency...
cot_ket_cuc = X.columns[X.columns.str.contains(r"death|re\.admission|emergency\.department")].tolist()

cot_bo = cot_dinh_danh + cot_hang_rong + cot_sau_xuat_vien + cot_ket_cuc + COT_BO_THEM
print("Các cột bị bỏ:", cot_bo, "\n")
X = X.drop(columns=cot_bo)

# 4. Sửa giá trị vô lý thành NaN (chỉ áp cho cột sinh hiệu)
cot_sinh_hieu = ["pulse", "respiration", "systolic.blood.pressure",
                 "diastolic.blood.pressure", "map", "weight"]
X[cot_sinh_hieu] = X[cot_sinh_hieu].replace(0, np.nan)
X.loc[X["height"] < 1.0, "height"] = np.nan
X["BMI"] = X["weight"] / X["height"] ** 2

# 5. NaN có nghĩa "không dùng" (không đặt "None": pandas đọc lại sẽ thành NaN)
X["respiratory.support."] = X["respiratory.support."].fillna("NoSupport")

# 6. Bỏ cột thiếu quá nhiều
ti_le_thieu = X.isna().mean()
cot_thieu_nhieu = ti_le_thieu[ti_le_thieu > MISSING_THRESHOLD].index.tolist()
print("Cột bỏ vì thiếu >", MISSING_THRESHOLD, ":", cot_thieu_nhieu, "\n")
X = X.drop(columns=cot_thieu_nhieu)

# 7. Chia tập (stratify theo nhãn)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# Kiểm tra nhanh
print("X_train:", X_train.shape, "| X_test:", X_test.shape)
print(pd.concat([y_train.value_counts(normalize=True).rename("train"),
                 y_test.value_counts(normalize=True).rename("test")], axis=1).round(3))
print("Nhãn còn nằm trong X?", TARGET in X.columns)
print("Còn cột kết cục trong X?",
      X.columns.str.contains(r"death|re\.admission|emergency").any())

# Xuất 4 file bàn giao (đã làm sạch, CHƯA impute/encode). Nhãn giữ dạng chữ: II, III, IV
X_train.to_csv("X_train_clean.csv", index=False)
X_test.to_csv("X_test_clean.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("Đã xuất 4 file.")