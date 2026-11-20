# analysis.py
# Project 9: Hospital Appointment Operations Analysis
# Industry: Healthcare Operations
# Libraries used: numpy and pandas only
# NOTE: This is an OPERATIONAL analysis (attendance, waiting time, workload, fees).
#       No medical / clinical conclusions are made.
# Run from the project root folder:  python src/analysis.py

import os
import numpy as np
import pandas as pd

# create the outputs folder if it does not exist (fixes 'non-existent directory' error)
os.makedirs("outputs", exist_ok=True)

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 20)

# This list collects everything we want in the final text report
report = []


def add_section(title):
    """Print a heading and also store it in the report."""
    line = "=" * 70
    print("\n" + line)
    print(title)
    print(line)
    report.append("\n" + line)
    report.append(title)
    report.append(line)


# ======================================================================
# TASK 1: Load and inspect appointment records
# ======================================================================
add_section("TASK 1: Load and inspect the data")

df = pd.read_csv("data/09_hospital_appointments.csv")

print("First 5 rows:")
print(df.head())
print("\nShape (rows, columns):", df.shape)
print("\nColumn data types:")
print(df.dtypes)

# Appointment_Date is read as text, so convert it to a proper date
df["Appointment_Date"] = pd.to_datetime(df["Appointment_Date"])

print("\nSummary of numeric columns:")
print(df.describe())

print("\nDate range:", df["Appointment_Date"].min().date(), "to", df["Appointment_Date"].max().date())
print("Unique departments:", df["Department"].nunique())
print("Unique doctors:", df["Doctor_ID"].nunique())

report.append("Raw rows loaded: " + str(df.shape[0]))
report.append("Columns: " + str(df.shape[1]))
report.append("Date range: " + str(df["Appointment_Date"].min().date()) +
              " to " + str(df["Appointment_Date"].max().date()))

# ======================================================================
# TASK 2 (part 1): Identify missing waiting times
# ======================================================================
add_section("TASK 2: Missing waiting times (identify)")

print("Missing values in every column:")
print(df.isnull().sum())

# Is the missing waiting time because the patient did not attend?
missing_by_status = pd.crosstab(df["Attendance_Status"], df["Waiting_Time_Min"].isnull())
missing_by_status.columns = ["Has_Waiting_Time", "Missing_Waiting_Time"]
print("\nMissing waiting time by attendance status:")
print(missing_by_status)

report.append("Missing waiting time by attendance status:")
report.append(missing_by_status.to_string())

# ======================================================================
# TASK 3: Check duplicate Appointment_ID values
# (done before filling missing values so repeated rows do not affect the median)
# ======================================================================
add_section("TASK 3: Duplicate Appointment_ID check")

dup_count = df["Appointment_ID"].duplicated().sum()
print("Duplicate Appointment_ID rows:", dup_count)

# show every row that shares an ID with another row
dup_rows = df[df["Appointment_ID"].duplicated(keep=False)].sort_values("Appointment_ID")
print("\nRows involved in duplication:")
print(dup_rows)

# are they exact copies of the whole row?
exact_dup = df.duplicated().sum()
print("\nFully identical duplicate rows:", exact_dup)

rows_before = df.shape[0]
df = df.drop_duplicates(subset="Appointment_ID", keep="first").reset_index(drop=True)
rows_after = df.shape[0]
print("Rows before:", rows_before, "| Rows after removing duplicates:", rows_after)

report.append("Duplicate Appointment_ID rows found: " + str(dup_count))
report.append("Fully identical duplicate rows: " + str(exact_dup))
report.append("Rows before: " + str(rows_before) + ", rows after: " + str(rows_after))

# ======================================================================
# TASK 2 (part 2): Handle missing waiting times
# ======================================================================
add_section("TASK 2 (continued): Missing waiting times (handle)")

# Case A: No-Show / Cancelled -> patient never waited. NaN is correct, so we keep it empty.
# Case B: Attended but waiting time missing -> a data-entry gap. Fill using the median
#         waiting time of the same department (median is not affected by long waits).
attended_mask = df["Attendance_Status"] == "Attended"
missing_attended = (attended_mask & df["Waiting_Time_Min"].isnull()).sum()
print("Attended rows with missing waiting time:", missing_attended)

dept_median_wait = df[attended_mask].groupby("Department")["Waiting_Time_Min"].median()
print("\nDepartment median waiting time used for filling:")
print(dept_median_wait)

df["Waiting_Time_Clean"] = df["Waiting_Time_Min"]
fill_values = df["Department"].map(dept_median_wait)
fill_condition = attended_mask & df["Waiting_Time_Clean"].isnull()
df.loc[fill_condition, "Waiting_Time_Clean"] = fill_values[fill_condition]

print("\nAttended rows still missing a waiting time:",
      (attended_mask & df["Waiting_Time_Clean"].isnull()).sum())
print("No-Show / Cancelled rows left as NaN (not applicable):",
      (~attended_mask & df["Waiting_Time_Clean"].isnull()).sum())

report.append("Attended rows with missing waiting time (filled with department median): " +
              str(missing_attended))
report.append("No-Show / Cancelled rows kept as NaN because no waiting happened.")

# Helpful extra columns used in later tasks
df["Is_NoShow"] = (df["Attendance_Status"] == "No-Show").astype(int)
df["Is_Attended"] = (df["Attendance_Status"] == "Attended").astype(int)
attended_df = df[df["Attendance_Status"] == "Attended"].copy()

# ======================================================================
# TASK 4: Total appointments and attendance-status percentages
# ======================================================================
add_section("TASK 4: Total appointments and attendance status %")

total_appointments = df.shape[0]
print("Total appointments:", total_appointments)

status_counts = df["Attendance_Status"].value_counts()
status_pct = (df["Attendance_Status"].value_counts(normalize=True) * 100).round(2)
status_summary = pd.DataFrame({"Count": status_counts, "Percentage": status_pct})
print(status_summary)

status_summary.to_csv("outputs/04_attendance_status_summary.csv")
report.append("Total appointments: " + str(total_appointments))
report.append(status_summary.to_string())

# ======================================================================
# TASK 5: Average waiting time by department (attended appointments)
# ======================================================================
add_section("TASK 5: Average waiting time by department")

wait_by_dept = attended_df.groupby("Department").agg(
    Attended_Count=("Appointment_ID", "count"),
    Avg_Wait_Min=("Waiting_Time_Clean", "mean"),
    Median_Wait_Min=("Waiting_Time_Clean", "median"),
    Max_Wait_Min=("Waiting_Time_Clean", "max")
).round(2).sort_values("Avg_Wait_Min", ascending=False)
print(wait_by_dept)

wait_by_dept.to_csv("outputs/05_avg_waiting_time_by_department.csv")
report.append(wait_by_dept.to_string())

# ======================================================================
# TASK 6: Departments with the highest no-show percentage
# ======================================================================
add_section("TASK 6: No-show percentage by department")

noshow_by_dept = df.groupby("Department").agg(
    Total_Appointments=("Appointment_ID", "count"),
    No_Shows=("Is_NoShow", "sum"),
    NoShow_Pct=("Is_NoShow", "mean")
)
noshow_by_dept["NoShow_Pct"] = (noshow_by_dept["NoShow_Pct"] * 100).round(2)
noshow_by_dept = noshow_by_dept.sort_values("NoShow_Pct", ascending=False)
print(noshow_by_dept)
print("\nTop 3 departments by no-show %:")
print(noshow_by_dept.head(3))

noshow_by_dept.to_csv("outputs/06_noshow_by_department.csv")
report.append(noshow_by_dept.to_string())

# ======================================================================
# TASK 7: No-show rates for SMS reminder Yes vs No
# ======================================================================
add_section("TASK 7: SMS reminder vs attendance status")

reminder_counts = pd.crosstab(df["SMS_Reminder"], df["Attendance_Status"])
reminder_pct = (pd.crosstab(df["SMS_Reminder"], df["Attendance_Status"], normalize="index") * 100).round(2)
print("Counts:")
print(reminder_counts)
print("\nRow percentages:")
print(reminder_pct)

noshow_sms = df.groupby("SMS_Reminder")["Is_NoShow"].mean() * 100
noshow_sms = noshow_sms.round(2)
print("\nNo-show rate (%) by SMS reminder:")
print(noshow_sms)

sms_gap = noshow_sms["No"] - noshow_sms["Yes"]
print("\nNo-show rate is", round(sms_gap, 2), "percentage points higher when no SMS is sent.")

reminder_pct.to_csv("outputs/07_sms_reminder_vs_attendance.csv")
report.append("Counts:\n" + reminder_counts.to_string())
report.append("\nRow percentages:\n" + reminder_pct.to_string())
report.append("\nNo-show gap (No SMS minus SMS): " + str(round(sms_gap, 2)) + " percentage points")

# ======================================================================
# TASK 8: Average consultation fee by department
# ======================================================================
add_section("TASK 8: Average consultation fee by department")

fee_by_dept = df.groupby("Department").agg(
    Avg_Fee=("Consultation_Fee", "mean"),
    Min_Fee=("Consultation_Fee", "min"),
    Max_Fee=("Consultation_Fee", "max")
).round(2).sort_values("Avg_Fee", ascending=False)
print(fee_by_dept)

fee_by_dept.to_csv("outputs/08_avg_fee_by_department.csv")
report.append(fee_by_dept.to_string())

# ======================================================================
# TASK 9: Appointment volume by date and department
# ======================================================================
add_section("TASK 9: Appointment volume by date and department")

# date x department table
volume_table = pd.crosstab(df["Appointment_Date"], df["Department"])
volume_table["Total"] = volume_table.sum(axis=1)
print("First 10 days of the date x department table:")
print(volume_table.head(10))

daily_total = volume_table["Total"]
print("\nAverage appointments per working day:", round(daily_total.mean(), 2))
print("Busiest 5 days:")
print(daily_total.sort_values(ascending=False).head(5))
print("Quietest 5 days:")
print(daily_total.sort_values().head(5))

# by day of week
df["Day_Of_Week"] = df["Appointment_Date"].dt.day_name()
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
weekday_volume = df["Day_Of_Week"].value_counts().reindex(day_order)
# divide by number of such days so we compare average load per day
days_per_weekday = df.groupby("Day_Of_Week")["Appointment_Date"].nunique().reindex(day_order)
weekday_summary = pd.DataFrame({
    "Total_Appointments": weekday_volume,
    "Working_Days": days_per_weekday,
    "Avg_Per_Day": (weekday_volume / days_per_weekday).round(2)
})
print("\nVolume by day of week:")
print(weekday_summary)

# by month
df["Month"] = df["Appointment_Date"].dt.month_name()
month_dept = pd.crosstab(df["Month"], df["Department"])
print("\nVolume by month and department:")
print(month_dept)

# department workload and doctor workload
dept_workload = df.groupby("Department").agg(
    Total_Appointments=("Appointment_ID", "count"),
    Doctors=("Doctor_ID", "nunique")
)
dept_workload["Share_Pct"] = (dept_workload["Total_Appointments"] / total_appointments * 100).round(2)
dept_workload["Appts_Per_Doctor"] = (dept_workload["Total_Appointments"] / dept_workload["Doctors"]).round(1)
dept_workload = dept_workload.sort_values("Total_Appointments", ascending=False)
print("\nDepartment workload:")
print(dept_workload)

volume_table.to_csv("outputs/09_volume_by_date_and_department.csv")
weekday_summary.to_csv("outputs/09_volume_by_weekday.csv")
dept_workload.to_csv("outputs/09_department_workload.csv")
report.append("Average appointments per working day: " + str(round(daily_total.mean(), 2)))
report.append("\nBusiest 5 days:\n" + daily_total.sort_values(ascending=False).head(5).to_string())
report.append("\nVolume by day of week:\n" + weekday_summary.to_string())
report.append("\nDepartment workload:\n" + dept_workload.to_string())

# ======================================================================
# TASK 10: Waiting times for new vs follow-up appointments
# ======================================================================
add_section("TASK 10: Waiting time - New vs Follow-up")

type_wait = attended_df.groupby("Appointment_Type").agg(
    Attended_Count=("Appointment_ID", "count"),
    Avg_Wait_Min=("Waiting_Time_Clean", "mean"),
    Median_Wait_Min=("Waiting_Time_Clean", "median"),
    Max_Wait_Min=("Waiting_Time_Clean", "max")
).round(2)
print(type_wait)

type_gap = type_wait.loc["New", "Avg_Wait_Min"] - type_wait.loc["Follow-up", "Avg_Wait_Min"]
print("\nNew appointments wait", round(type_gap, 2), "minutes longer on average.")

# same comparison inside each department
type_dept_wait = attended_df.pivot_table(index="Department", columns="Appointment_Type",
                                         values="Waiting_Time_Clean", aggfunc="mean").round(2)
type_dept_wait["Difference"] = (type_dept_wait["New"] - type_dept_wait["Follow-up"]).round(2)
print("\nAverage wait by department and type:")
print(type_dept_wait)

type_wait.to_csv("outputs/10_waiting_new_vs_followup.csv")
type_dept_wait.to_csv("outputs/10_waiting_type_by_department.csv")
report.append(type_wait.to_string())
report.append("\nNew minus Follow-up average wait: " + str(round(type_gap, 2)) + " minutes")
report.append("\n" + type_dept_wait.to_string())

# ======================================================================
# TASK 11: NumPy statistics of waiting time
# ======================================================================
add_section("TASK 11: NumPy statistics of waiting time")

wait_array = attended_df["Waiting_Time_Clean"].to_numpy()

mean_wait = np.mean(wait_array)
median_wait = np.median(wait_array)
q1 = np.percentile(wait_array, 25)
q2 = np.percentile(wait_array, 50)
q3 = np.percentile(wait_array, 75)
std_wait = np.std(wait_array, ddof=1)      # ddof=1 -> sample standard deviation
min_wait = np.min(wait_array)
max_wait = np.max(wait_array)

print("Number of values :", wait_array.size)
print("Mean             :", round(mean_wait, 2))
print("Median           :", median_wait)
print("Q1 (25th pct)    :", q1)
print("Q2 (50th pct)    :", q2)
print("Q3 (75th pct)    :", q3)
print("Std deviation    :", round(std_wait, 2))
print("Min / Max        :", min_wait, "/", max_wait)

numpy_stats = pd.DataFrame({
    "Statistic": ["Count", "Mean", "Median", "Q1", "Q2", "Q3", "Std_Dev", "Min", "Max"],
    "Value": [wait_array.size, mean_wait, median_wait, q1, q2, q3, std_wait, min_wait, max_wait]
}).round(2)
numpy_stats.to_csv("outputs/11_numpy_waiting_time_statistics.csv", index=False)
report.append(numpy_stats.to_string(index=False))

# ======================================================================
# TASK 12: Unusually long waiting times using IQR
# ======================================================================
add_section("TASK 12: Outliers in waiting time (IQR method)")

iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
print("IQR         :", round(iqr, 2))
print("Lower limit :", round(lower_limit, 2))
print("Upper limit :", round(upper_limit, 2))

outlier_mask = (attended_df["Waiting_Time_Clean"] > upper_limit) | (attended_df["Waiting_Time_Clean"] < lower_limit)
outliers = attended_df[outlier_mask]
print("\nNumber of unusually long waits:", outliers.shape[0])
print("Percentage of attended appointments:", round(outliers.shape[0] / attended_df.shape[0] * 100, 2), "%")

outlier_by_dept = outliers.groupby("Department").agg(
    Outlier_Count=("Appointment_ID", "count"),
    Avg_Outlier_Wait=("Waiting_Time_Clean", "mean")
).round(2).sort_values("Outlier_Count", ascending=False)
print("\nOutliers by department:")
print(outlier_by_dept)

print("\nTop 10 longest waits:")
top_waits = outliers.sort_values("Waiting_Time_Clean", ascending=False).head(10)
print(top_waits[["Appointment_ID", "Appointment_Date", "Department", "Doctor_ID",
                 "Appointment_Type", "Waiting_Time_Clean"]])

outliers.to_csv("outputs/12_waiting_time_outliers.csv", index=False)
outlier_by_dept.to_csv("outputs/12_outliers_by_department.csv")
report.append("IQR = " + str(round(iqr, 2)) + ", lower limit = " + str(round(lower_limit, 2)) +
              ", upper limit = " + str(round(upper_limit, 2)))
report.append("Unusually long waits: " + str(outliers.shape[0]))
report.append("\n" + outlier_by_dept.to_string())

# ======================================================================
# TASK 13: Waiting-time categories
# ======================================================================
add_section("TASK 13: Waiting-time categories")

bins = [0, 15, 30, 45, 60, np.inf]
labels = ["Very Short (0-15)", "Short (16-30)", "Moderate (31-45)", "Long (46-60)", "Very Long (60+)"]
df["Waiting_Band"] = pd.cut(df["Waiting_Time_Clean"], bins=bins, labels=labels)
attended_df["Waiting_Band"] = pd.cut(attended_df["Waiting_Time_Clean"], bins=bins, labels=labels)

band_counts = attended_df["Waiting_Band"].value_counts().reindex(labels)
band_pct = (band_counts / band_counts.sum() * 100).round(2)
band_summary = pd.DataFrame({"Count": band_counts, "Percentage": band_pct})
print(band_summary)

band_by_dept = pd.crosstab(attended_df["Department"], attended_df["Waiting_Band"])
band_by_dept_pct = (pd.crosstab(attended_df["Department"], attended_df["Waiting_Band"],
                                normalize="index") * 100).round(2)
print("\nWaiting band by department (%):")
print(band_by_dept_pct)

band_summary.to_csv("outputs/13_waiting_time_bands.csv")
band_by_dept_pct.to_csv("outputs/13_waiting_bands_by_department_pct.csv")
report.append(band_summary.to_string())
report.append("\nBy department (%):\n" + band_by_dept_pct.to_string())

# ======================================================================
# TASK 14: Estimated total consultation fees for attended appointments
# ======================================================================
add_section("TASK 14: Estimated consultation fees (attended only)")

total_fee_attended = attended_df["Consultation_Fee"].sum()
print("Estimated total fees (attended appointments): Rs.", total_fee_attended)

fee_summary = attended_df.groupby("Department").agg(
    Attended_Appointments=("Appointment_ID", "count"),
    Avg_Fee=("Consultation_Fee", "mean"),
    Estimated_Total_Fee=("Consultation_Fee", "sum")
).round(2)
fee_summary["Fee_Share_Pct"] = (fee_summary["Estimated_Total_Fee"] / total_fee_attended * 100).round(2)
fee_summary = fee_summary.sort_values("Estimated_Total_Fee", ascending=False)
print(fee_summary)

# extra: fee value of appointments that were booked but not attended
noshow_fee_lost = df[df["Attendance_Status"] == "No-Show"]["Consultation_Fee"].sum()
cancel_fee_lost = df[df["Attendance_Status"] == "Cancelled"]["Consultation_Fee"].sum()
print("\nFee value of No-Show appointments  : Rs.", noshow_fee_lost)
print("Fee value of Cancelled appointments: Rs.", cancel_fee_lost)

fee_summary.to_csv("outputs/14_estimated_fee_summary.csv")
report.append("Estimated total fees (attended): Rs. " + str(total_fee_attended))
report.append(fee_summary.to_string())
report.append("\nFee value of No-Show appointments: Rs. " + str(noshow_fee_lost))
report.append("Fee value of Cancelled appointments: Rs. " + str(cancel_fee_lost))

# ======================================================================
# EXTRA ANALYSIS: department bottleneck score
# (departments that are slow AND have high no-shows AND heavy doctor load)
# ======================================================================
add_section("EXTRA: Department bottleneck summary")

dept_summary = pd.DataFrame({
    "Total_Appointments": dept_workload["Total_Appointments"],
    "Appts_Per_Doctor": dept_workload["Appts_Per_Doctor"],
    "Avg_Wait_Min": wait_by_dept["Avg_Wait_Min"],
    "NoShow_Pct": noshow_by_dept["NoShow_Pct"],
    "Avg_Fee": fee_by_dept["Avg_Fee"],
    "Estimated_Total_Fee": fee_summary["Estimated_Total_Fee"]
})

# rank each measure (1 = worst) and add the ranks
dept_summary["Wait_Rank"] = dept_summary["Avg_Wait_Min"].rank(ascending=False)
dept_summary["NoShow_Rank"] = dept_summary["NoShow_Pct"].rank(ascending=False)
dept_summary["Load_Rank"] = dept_summary["Appts_Per_Doctor"].rank(ascending=False)
dept_summary["Bottleneck_Score"] = dept_summary["Wait_Rank"] + dept_summary["NoShow_Rank"] + dept_summary["Load_Rank"]
dept_summary = dept_summary.sort_values("Bottleneck_Score")
print("(lower score = bigger bottleneck)")
print(dept_summary)

dept_summary.to_csv("outputs/department_summary.csv")
report.append("(lower score = bigger bottleneck)")
report.append(dept_summary.to_string())

# ======================================================================
# TASK 15: Five operational recommendations
# ======================================================================
add_section("TASK 15: Operational recommendations")

slowest_dept = wait_by_dept.index[0]
slowest_wait = wait_by_dept["Avg_Wait_Min"].iloc[0]
top_noshow_dept = noshow_by_dept.index[0]
top_noshow_pct = noshow_by_dept["NoShow_Pct"].iloc[0]
busiest_dept = dept_workload.index[0]
busiest_per_doc = dept_workload["Appts_Per_Doctor"].max()
busiest_weekday = weekday_summary["Avg_Per_Day"].idxmax()
busiest_weekday_avg = weekday_summary["Avg_Per_Day"].max()
long_wait_pct = band_pct["Long (46-60)"] + band_pct["Very Long (60+)"]
top_fee_dept = fee_summary.index[0]

recommendations = [
    "1. Send SMS reminders for every appointment. No-show rate is " + str(noshow_sms["No"]) +
    "% without a reminder versus " + str(noshow_sms["Yes"]) + "% with one, a gap of " +
    str(round(sms_gap, 2)) + " percentage points. Reminders cost little and protect doctor time.",

    "2. Target " + top_noshow_dept + " (highest no-show rate, " + str(top_noshow_pct) +
    "%) with a second reminder 24 hours before the visit and a simple confirm/cancel reply, so "
    "freed slots can be offered to waiting patients.",

    "3. Reduce waiting at " + slowest_dept + " (average " + str(slowest_wait) +
    " minutes). Review how slots are spaced and stagger arrival times; consider extra "
    "front-desk or triage support during its peak hours.",

    "4. Add capacity on " + busiest_weekday + " (average " + str(busiest_weekday_avg) +
    " appointments per day, the highest of the week) and rebalance load for " + busiest_dept +
    ", which has " + str(busiest_per_doc) + " appointments per doctor. Move some bookings to "
    "quieter days.",

    "5. Give new-patient appointments longer slots, since new patients wait about " +
    str(round(type_gap, 1)) + " minutes more than follow-ups. Also review the " +
    str(outliers.shape[0]) + " unusually long waits (above " + str(round(upper_limit, 1)) +
    " minutes) to find the doctors and days where delays pile up. "
    "Together, " + str(round(long_wait_pct, 1)) + "% of attended visits wait over 45 minutes."
]

for rec in recommendations:
    print(rec)
    print()
    report.append(rec)

# note about limits of the data
note = ("Note: This is an operational analysis only. No clinical or medical conclusions "
        "are made. Results are based on a synthetic dataset.")
print(note)
report.append("\n" + note)

# ======================================================================
# Save the cleaned dataset and the text report
# ======================================================================
df.to_csv("outputs/cleaned_hospital_appointments.csv", index=False)

report_file = open("outputs/analysis_report.txt", "w")
report_file.write("HOSPITAL APPOINTMENT OPERATIONS ANALYSIS - REPORT\n")
report_file.write("\n".join(report))
report_file.close()

print("\nAll tables saved in the outputs/ folder.")
