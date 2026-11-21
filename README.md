# Hospital Appointment Operations Analysis

**Project 9 · Healthcare Operations · Python (NumPy & Pandas)**

| | |
|---|---|
| **Student Name** | _Your Name_ |
| **Roll No / Enrollment No** | _Your ID_ |
| **Course** | _Course Name_ |
| **Instructor** | _Instructor Name_ |
| **Submission Date** | _DD-MM-YYYY_ |

---

## About the Project

This project studies hospital appointment records to find out where patients wait the longest,
which departments see the most missed appointments, and whether SMS reminders help.
The data is **operational only** (no diagnosis or clinical details), so no medical conclusions are drawn.

## Goals

- Clean the data: missing waiting times and duplicate Appointment_ID values
- Measure attendance, no-show and cancellation rates
- Compare departments on waiting time, workload, fees and no-shows
- Test the effect of SMS reminders using a crosstab
- Find unusually long waits with the IQR rule
- Group waiting times into bands and estimate fee income
- Give five practical recommendations

## Tools

- **Python 3.9+**
- **Pandas**: loading, cleaning, `groupby().agg()`, `crosstab()`, `pd.cut()`
- **NumPy**: `median`, `percentile`, `std`

## Project Structure

```
hospital_appointment_analysis/
├── data/                 # 09_hospital_appointments.csv
├── docs/                 # SRS document
├── outputs/              # all generated result files
├── generate_dataset.py   # builds the dataset
├── main.py               # full analysis, tasks 1-15
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone <your-repository-link>
cd hospital_appointment_analysis
pip install numpy pandas
python main.py
```

Results are saved in the `outputs/` folder and printed on screen.

## Approach

| Step | What I did |
|------|-----------|
| Inspect | `head()`, `shape`, `dtypes`, `describe()` |
| Duplicates | Checked with `duplicated()` and removed repeated IDs |
| Missing values | Blank waits for No-Show/Cancelled are expected; real gaps for attended patients filled with department median |
| Summaries | Attendance %, average wait, no-show %, average fee by department |
| Reminders | `crosstab()` of SMS_Reminder vs Attendance_Status |
| Volume | Appointments by date and department |
| Statistics | NumPy median, quartiles, standard deviation |
| Outliers | Q1 − 1.5×IQR and Q3 + 1.5×IQR |
| Categories | Waiting-time bands with `pd.cut()` |
| Fees | Sum of consultation fees for attended appointments |

## My Results

_Run `main.py`, then fill this table from `outputs/analysis_report.txt`._

| Measure | Result |
|---------|--------|
| Total appointments (after duplicates removed) | |
| Attended / No-Show / Cancelled % | |
| Department with highest no-show % | |
| Department with longest average wait | |
| No-show % with SMS vs without SMS | |
| New vs follow-up average wait | |
| Median / Q1 / Q3 / Std dev of waiting time | |
| IQR upper limit and number of outliers | |
| Estimated fees (attended only) | |

## Recommendations

_Write five recommendations in your own words, based on your results._

1.
2.
3.
4.
5.

## Limitations

- The dataset is synthetic, so the results show the method, not a real hospital.
- Only operational data is used; no clinical conclusions.

## Deliverables

- SRS document: `docs/`
- Source code: `main.py`, `generate_dataset.py`
- Repository link: _add link here_
