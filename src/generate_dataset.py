# generate_dataset.py
# Project 9: Hospital Appointment Operations Analysis
# This script creates the synthetic (made-up) dataset 09_hospital_appointments.csv
# It is an OPERATIONAL dataset only - there is no clinical / diagnosis information.
# Libraries used: numpy and pandas only.
# Run from the project root folder:  python src/generate_dataset.py

import os
import numpy as np
import pandas as pd

# create the data folder if it does not exist
os.makedirs("data", exist_ok=True)

# fixed seed so the same dataset is created every time
np.random.seed(42)

n_rows = 2000

# ------------------------------------------------------------------
# 1. Department settings
#    base wait (minutes), wait spread, base fee (INR), base no-show chance, share of bookings
# ------------------------------------------------------------------
dept_info = pd.DataFrame({
    "Department":   ["General Medicine", "Orthopedics", "Pediatrics", "Cardiology",
                     "Dermatology", "ENT", "Gynecology", "Ophthalmology"],
    "Code":         ["GM", "OR", "PD", "CD", "DM", "EN", "GY", "OP"],
    "Base_Wait":    [38, 32, 24, 45, 20, 22, 30, 26],
    "Wait_SD":      [12, 10, 7, 14, 6, 7, 9, 8],
    "Base_Fee":     [500, 800, 600, 1200, 700, 650, 900, 750],
    "NoShow_Prob":  [0.16, 0.14, 0.10, 0.09, 0.22, 0.18, 0.11, 0.13],
    "Share":        [0.22, 0.14, 0.13, 0.12, 0.10, 0.10, 0.10, 0.09]
})

# ------------------------------------------------------------------
# 2. Pick a department, date, type and reminder for every appointment
# ------------------------------------------------------------------
dept_choice = np.random.choice(dept_info.index, size=n_rows, p=dept_info["Share"])

# only working days (Monday to Saturday) from 1 Jan 2025 to 31 Mar 2025
all_days = pd.date_range("2025-01-01", "2025-03-31", freq="D")
working_days = all_days[all_days.dayofweek != 6]          # 6 = Sunday
# Mondays are a bit busier, so give Monday a higher weight
day_weights = np.where(working_days.dayofweek == 0, 1.5, 1.0)
day_weights = day_weights / day_weights.sum()
appt_dates = np.random.choice(working_days, size=n_rows, p=day_weights)

appt_type = np.random.choice(["New", "Follow-up"], size=n_rows, p=[0.42, 0.58])
sms = np.random.choice(["Yes", "No"], size=n_rows, p=[0.62, 0.38])

# ------------------------------------------------------------------
# 3. Build the main DataFrame
# ------------------------------------------------------------------
df = pd.DataFrame()
df["Appointment_ID"] = ["APT" + str(i) for i in range(10001, 10001 + n_rows)]
df["Patient_ID"] = ["PAT" + str(x).zfill(4) for x in np.random.randint(1, 1400, size=n_rows)]
df["Appointment_Date"] = pd.to_datetime(appt_dates)
df["Department"] = dept_info.loc[dept_choice, "Department"].values
df["Doctor_ID"] = [dept_info.loc[d, "Code"] + "-D" + str(np.random.randint(1, 5))
                   for d in dept_choice]
df["Appointment_Type"] = appt_type
df["SMS_Reminder"] = sms

# ------------------------------------------------------------------
# 4. Attendance status
#    no-show chance = department base chance, +10% if no SMS, +2% for new patients
# ------------------------------------------------------------------
noshow_prob = dept_info.loc[dept_choice, "NoShow_Prob"].values
noshow_prob = noshow_prob + np.where(sms == "No", 0.10, 0.0)
noshow_prob = noshow_prob + np.where(appt_type == "New", 0.02, 0.0)

rand_vals = np.random.rand(n_rows)
status = np.where(rand_vals < noshow_prob, "No-Show", "Attended")

# about 6% of the remaining appointments are cancelled
cancel_rand = np.random.rand(n_rows)
status = np.where((status == "Attended") & (cancel_rand < 0.06), "Cancelled", status)
df["Attendance_Status"] = status

# ------------------------------------------------------------------
# 5. Waiting time in minutes (only exists if the patient attended)
# ------------------------------------------------------------------
base_wait = dept_info.loc[dept_choice, "Base_Wait"].values
wait_sd = dept_info.loc[dept_choice, "Wait_SD"].values
wait = np.random.normal(base_wait, wait_sd)
wait = wait + np.where(appt_type == "New", 8, 0)          # new patients wait longer
wait = np.where(wait < 3, 3, wait)                        # nobody waits less than 3 minutes

# add some very long waits (about 3% of rows)
extreme = np.random.rand(n_rows) < 0.03
wait = np.where(extreme, wait + np.random.randint(60, 120, size=n_rows), wait)
wait = np.round(wait)

df["Waiting_Time_Min"] = wait
# no-show and cancelled patients never waited, so those are empty (NaN)
df.loc[df["Attendance_Status"] != "Attended", "Waiting_Time_Min"] = np.nan

# some attended rows have the waiting time missing because of data-entry problems (~5%)
attended_idx = df[df["Attendance_Status"] == "Attended"].index
missing_idx = np.random.choice(attended_idx, size=int(len(attended_idx) * 0.05), replace=False)
df.loc[missing_idx, "Waiting_Time_Min"] = np.nan

# ------------------------------------------------------------------
# 6. Consultation fee (INR)
# ------------------------------------------------------------------
base_fee = dept_info.loc[dept_choice, "Base_Fee"].values
fee = base_fee * np.where(appt_type == "New", 1.25, 1.0)
fee = fee + np.random.choice([-50, 0, 0, 50], size=n_rows)  # small variation
df["Consultation_Fee"] = fee.astype(int)

# ------------------------------------------------------------------
# 7. Add a few duplicate Appointment_ID rows (data-quality problem to detect later)
# ------------------------------------------------------------------
dup_rows = df.sample(n=15, random_state=7)
df = pd.concat([df, dup_rows], ignore_index=True)

# sort by date and ID so the file looks like a normal booking log
df = df.sort_values(["Appointment_Date", "Appointment_ID"]).reset_index(drop=True)

# ------------------------------------------------------------------
# 8. Save
# ------------------------------------------------------------------
df.to_csv("data/09_hospital_appointments.csv", index=False)

print("Dataset created: data/09_hospital_appointments.csv")
print("Rows:", df.shape[0], "| Columns:", df.shape[1])
print(df.head())
