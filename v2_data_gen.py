"""
=============================================================================
  STUDENT GRADUATION DATASET GENERATOR
  -----------------------------------------------------------------------------
  Run this script on your PC to generate a synthetic but logically realistic
  student dataset for training a 4-year graduation classification model.

  HOW TO RUN:
      pip install numpy pandas
      python student_data_generator.py

  OUTPUT:
      student_graduation_dataset.csv  (saved in the same folder as this script)
=============================================================================
"""

import numpy as np
import pandas as pd
import random
import os

np.random.seed(42)
random.seed(42)

# =============================================================================
#  STEP 1 — ENTER YOUR COHORT INFORMATION
#  Add or remove years as needed. Format: { year: number_of_students }
# =============================================================================

def get_cohort_config():
    print("\n" + "="*60)
    print("  STUDENT GRADUATION DATASET GENERATOR")
    print("="*60)
    print("\n  Enter cohort info (year + number of students enrolled).")
    print("  Type 'done' when finished.\n")

    cohorts = {}
    while True:
        year_input = input("  Cohort year (or 'done' to finish): ").strip()
        if year_input.lower() == "done":
            if len(cohorts) == 0:
                print("  ⚠  Please enter at least one cohort year.")
                continue
            break
        try:
            year = int(year_input)
            if year < 2000 or year > 2100:
                print("  ⚠  Please enter a valid year (e.g. 2018).")
                continue
        except ValueError:
            print("  ⚠  Invalid year. Please enter a 4-digit year.")
            continue

        size_input = input(f"  Number of students enrolled in {year}: ").strip()
        try:
            size = int(size_input)
            if size <= 0:
                print("  ⚠  Size must be greater than 0.")
                continue
        except ValueError:
            print("  ⚠  Invalid number. Please enter a whole number.")
            continue

        cohorts[year] = size
        print(f"  ✓  Cohort {year} → {size} students added.\n")

    return cohorts


# =============================================================================
#  STEP 2 — MAJOR / COLLEGE CONFIGURATION
#  Adjust difficulty (0–1) per major. Higher = harder = lower avg GPA.
# =============================================================================

MAJORS = {
    "Computer Science":  {"college": "Engineering",     "difficulty": 0.75},
    "Mechanical Eng.":   {"college": "Engineering",     "difficulty": 0.80},
    "Nursing":           {"college": "Health Sciences", "difficulty": 0.70},
    "Business Admin":    {"college": "Business",        "difficulty": 0.55},
    "Psychology":        {"college": "Liberal Arts",    "difficulty": 0.45},
    "Biology":           {"college": "Sciences",        "difficulty": 0.72},
    "Mathematics":       {"college": "Sciences",        "difficulty": 0.78},
    "English":           {"college": "Liberal Arts",    "difficulty": 0.40},
    "Accounting":        {"college": "Business",        "difficulty": 0.60},
    "Education":         {"college": "Education",       "difficulty": 0.50},
}

MAJOR_LIST    = list(MAJORS.keys())
MAJOR_WEIGHTS = [0.15, 0.10, 0.12, 0.18, 0.10, 0.10, 0.06, 0.07, 0.07, 0.05]

N_SEMESTERS   = 8   # 8 semesters = 4 academic years


# =============================================================================
#  STEP 3 — HELPER FUNCTIONS
# =============================================================================

def clamp_gpa(g):
    """Keep GPA between 0.0 and 4.0."""
    return round(float(np.clip(g, 0.0, 4.0)), 2)


def generate_student(student_id, cohort_year, covid_years=None):
    """
    Generate one student record with realistic academic trajectory.

    Logic baked in:
      - Scholarship holders → higher GPA, lower dropout risk
      - First-gen students  → lower baseline GPA, lower grad rate
      - COVID years         → GPA dip in Sem 2-4, higher dropout risk
      - Academic probation  → compounds dropout probability
      - Financial stress    → increases part-time likelihood & dropout
    """
    if covid_years is None:
        covid_years = []

    # ── Demographics ─────────────────────────────────────────────────────────
    age           = int(np.random.choice(
                        [17,18,19,20,21,22,23,24,25],
                        p=[0.05,0.45,0.25,0.10,0.05,0.04,0.03,0.02,0.01]))
    gender        = random.choice(["M","F","F","F","M"])     # slight female majority
    first_gen     = int(np.random.random() < 0.35)
    financial_aid = int(np.random.random() < (0.65 if first_gen else 0.40))
    scholarship   = int(np.random.random() < (0.10 if first_gen else 0.25))
    residency     = np.random.choice(
                        ["in_state","out_of_state","international"],
                        p=[0.68, 0.22, 0.10])
    major         = np.random.choice(MAJOR_LIST, p=MAJOR_WEIGHTS)
    college       = MAJORS[major]["college"]
    difficulty    = MAJORS[major]["difficulty"]
    entry_sem     = np.random.choice(["Fall","Spring"], p=[0.82, 0.18])

    # ── Admission strength ───────────────────────────────────────────────────
    hs_gpa_base = np.random.normal(3.2 - 0.25*first_gen + 0.30*scholarship, 0.40)
    hs_gpa      = round(float(np.clip(hs_gpa_base, 1.5, 4.0)), 2)

    sat_base    = np.random.normal(1050 + 120*(hs_gpa-3.0) - 60*first_gen + 80*scholarship, 120)
    sat_score   = int(np.clip(round(sat_base / 10) * 10, 600, 1600))

    # Underlying academic strength (0–1), drives GPA trajectory
    strength    = np.clip((hs_gpa/4.0)*0.6 + (sat_score/1600)*0.4, 0.1, 1.0)
    strength   += np.random.normal(0, 0.05)

    # COVID disruption flag
    covid_hit = (cohort_year in covid_years) and (np.random.random() < 0.30)

    # ── Simulate 8 semesters ─────────────────────────────────────────────────
    records            = {}
    cumulative_credits = 0
    cumulative_gpa_sum = 0.0
    probation_count    = 0
    major_change_count = 0
    current_major      = major
    withdrawal_total   = 0
    failure_total      = 0
    dropped_out        = False
    dropout_semester   = None
    prev_gpa           = None

    for sem in range(1, N_SEMESTERS + 1):

        if dropped_out:
            for k in ["gpa","cumulative_gpa","credits_attempted","credits_earned",
                      "credits_failed","credits_withdrawn","on_track",
                      "total_credits_accumulated","enrollment_status",
                      "major_changed","academic_standing","courses_repeated"]:
                records[f"sem{sem}_{k}"] = np.nan
            continue

        # Enrollment status
        financial_stress  = (financial_aid==0 and scholarship==0 and residency=="out_of_state")
        part_time_prob    = 0.08 + 0.10*financial_stress + 0.05*(age>22) - 0.05*scholarship
        enrollment_status = "part_time" if np.random.random() < part_time_prob else "full_time"
        credits_attempted = 12 if enrollment_status == "part_time" else random.choice([15,15,15,16,18,12])

        # GPA this semester
        gpa_mean = (strength * 3.8) - (difficulty * 0.5) + 0.20*scholarship
        if covid_hit and sem in [2,3,4]:
            gpa_mean -= 0.40
        if prev_gpa is not None and prev_gpa < 2.0:
            gpa_mean -= 0.15
        if probation_count > 0:
            gpa_mean += 0.10

        sem_gpa  = clamp_gpa(np.random.normal(gpa_mean, 0.35))
        prev_gpa = sem_gpa

        # Credits from GPA
        fail_rate         = max(0, (2.5 - sem_gpa) / 5.0)
        withdraw_rate     = max(0, (2.8 - sem_gpa) / 8.0) + 0.02*financial_stress
        credits_failed    = int(np.round(credits_attempted * np.random.uniform(0, fail_rate*1.5)))
        credits_withdrawn = int(np.round((credits_attempted-credits_failed) * np.random.uniform(0, withdraw_rate*2)))
        credits_withdrawn = max(0, min(credits_withdrawn, credits_attempted - credits_failed))
        credits_earned    = max(0, credits_attempted - credits_failed - credits_withdrawn)

        cumulative_credits += credits_earned
        withdrawal_total   += credits_withdrawn
        failure_total      += credits_failed
        cumulative_gpa_sum += sem_gpa
        cumulative_gpa      = clamp_gpa(cumulative_gpa_sum / sem)

        academic_standing = "probation" if cumulative_gpa < 2.0 else "good"
        if academic_standing == "probation":
            probation_count += 1

        # Major change (early semesters only)
        major_changed = 0
        if sem <= 4 and sem_gpa < 2.5 and np.random.random() < 0.12:
            current_major  = np.random.choice([m for m in MAJOR_LIST if m != current_major])
            major_changed  = 1
            major_change_count += 1

        courses_repeated = 1 if credits_failed > 0 and np.random.random() < 0.6 else 0
        on_track         = int(cumulative_credits >= sem*15*0.85 and cumulative_gpa >= 2.0)

        records[f"sem{sem}_gpa"]                      = sem_gpa
        records[f"sem{sem}_cumulative_gpa"]            = cumulative_gpa
        records[f"sem{sem}_credits_attempted"]         = credits_attempted
        records[f"sem{sem}_credits_earned"]            = credits_earned
        records[f"sem{sem}_credits_failed"]            = credits_failed
        records[f"sem{sem}_credits_withdrawn"]         = credits_withdrawn
        records[f"sem{sem}_on_track"]                  = on_track
        records[f"sem{sem}_total_credits_accumulated"] = cumulative_credits
        records[f"sem{sem}_enrollment_status"]         = enrollment_status
        records[f"sem{sem}_major_changed"]             = major_changed
        records[f"sem{sem}_academic_standing"]         = academic_standing
        records[f"sem{sem}_courses_repeated"]          = courses_repeated

        # Dropout probability
        p_dropout = 0.0
        if cumulative_gpa < 1.5:   p_dropout += 0.45
        elif cumulative_gpa < 2.0: p_dropout += 0.25
        if financial_stress:       p_dropout += 0.08
        if probation_count >= 2:   p_dropout += 0.20
        if covid_hit and sem in [2,3,4]: p_dropout += 0.10
        if on_track == 0:          p_dropout += 0.05
        p_dropout -= 0.15 * scholarship
        p_dropout  = float(np.clip(p_dropout, 0.0, 0.90))

        if np.random.random() < p_dropout:
            dropped_out      = True
            dropout_semester = sem

    # ── Graduation outcome ────────────────────────────────────────────────────
    if not dropped_out and cumulative_credits >= 110 and cumulative_gpa >= 2.0:
        graduated_4yr = 1
    elif not dropped_out and cumulative_credits >= 105 and cumulative_gpa >= 2.0 and np.random.random() < 0.4:
        graduated_4yr = 1
    else:
        graduated_4yr = 0

    # ── Engineered features ───────────────────────────────────────────────────
    g = [records.get(f"sem{s}_gpa", np.nan) for s in range(1,5)]

    def safe_diff(a, b):
        return round(b - a, 2) if not (np.isnan(a) or np.isnan(b)) else np.nan

    s1_att = records.get("sem1_credits_attempted", 15) or 15
    s1_ern = records.get("sem1_credits_earned", 0) or 0
    credits_behind = max(0, 60 - (records.get("sem4_total_credits_accumulated", 0) or 0))

    base = {
        "student_id":               student_id,
        "cohort_year":              cohort_year,
        "entry_semester":           entry_sem,
        "age_at_enrollment":        age,
        "gender":                   gender,
        "first_gen_student":        first_gen,
        "financial_aid":            financial_aid,
        "scholarship":              scholarship,
        "residency":                residency,
        "declared_major":           major,
        "current_major_sem4":       current_major,
        "college_division":         college,
        "hs_gpa":                   hs_gpa,
        "sat_score":                sat_score,
        "major_change_count":       major_change_count,
        "total_withdrawals":        withdrawal_total,
        "total_failures":           failure_total,
        "probation_count":          probation_count,
        "dropout_semester":         dropout_semester if dropped_out else np.nan,
        # engineered
        "gpa_trend_sem1_to_sem2":           safe_diff(g[0], g[1]),
        "gpa_trend_sem2_to_sem3":           safe_diff(g[1], g[2]),
        "gpa_trend_sem3_to_sem4":           safe_diff(g[2], g[3]),
        "avg_gpa_first2_sems":              round((g[0]+g[1])/2, 2) if not any(np.isnan(x) for x in g[:2]) else np.nan,
        "avg_gpa_first4_sems":              round(float(np.nanmean(g)), 2),
        "credit_completion_rate_sem1":      round(s1_ern / s1_att, 3) if s1_att > 0 else np.nan,
        "credits_behind_pace_after_sem4":   credits_behind,
        # target — LAST column
        "graduated_4yr":            graduated_4yr,
    }

    return {**base, **records}


# =============================================================================
#  STEP 4 — BUILD DATASET & SAVE
# =============================================================================

def build_dataset(cohorts, covid_years=None):
    if covid_years is None:
        covid_years = []

    all_students = []
    global_id    = 1

    for year, n in sorted(cohorts.items()):
        print(f"  Generating {n} students for cohort {year}...")
        for _ in range(n):
            sid = f"STU-{year}-{str(global_id).zfill(5)}"
            all_students.append(generate_student(sid, year, covid_years))
            global_id += 1

    df = pd.DataFrame(all_students)

    # ── Column ordering ───────────────────────────────────────────────────────
    id_cols   = ["student_id","cohort_year","entry_semester"]
    demo_cols = ["age_at_enrollment","gender","first_gen_student","financial_aid",
                 "scholarship","residency","declared_major","current_major_sem4",
                 "college_division","hs_gpa","sat_score"]
    sem_cols  = []
    for s in range(1, N_SEMESTERS+1):
        sem_cols += [
            f"sem{s}_gpa", f"sem{s}_cumulative_gpa",
            f"sem{s}_credits_attempted", f"sem{s}_credits_earned",
            f"sem{s}_credits_failed", f"sem{s}_credits_withdrawn",
            f"sem{s}_on_track", f"sem{s}_total_credits_accumulated",
            f"sem{s}_enrollment_status", f"sem{s}_major_changed",
            f"sem{s}_academic_standing", f"sem{s}_courses_repeated",
        ]
    eng_cols  = ["gpa_trend_sem1_to_sem2","gpa_trend_sem2_to_sem3","gpa_trend_sem3_to_sem4",
                 "avg_gpa_first2_sems","avg_gpa_first4_sems",
                 "credit_completion_rate_sem1","credits_behind_pace_after_sem4"]
    misc_cols = ["major_change_count","total_withdrawals","total_failures",
                 "probation_count","dropout_semester"]
    target    = ["graduated_4yr"]

    df = df[id_cols + demo_cols + sem_cols + eng_cols + misc_cols + target]
    return df


def print_summary(df, cohorts):
    print(f"\n{'='*60}")
    print(f"  Dataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"{'='*60}")
    print("\n  Graduation rates by cohort:")
    for yr in sorted(cohorts.keys()):
        sub  = df[df.cohort_year == yr]
        rate = sub.graduated_4yr.mean() * 100
        print(f"    {yr} : {rate:.1f}%  ({int(sub.graduated_4yr.sum())}/{len(sub)} students)")
    print(f"\n  Overall graduation rate   : {df.graduated_4yr.mean()*100:.1f}%")
    print(f"  Students who dropped out  : {df.dropout_semester.notna().sum():,}")
    print(f"\n  GPA snapshot — Semester 1:")
    print(f"    Graduates avg GPA   : {df[df.graduated_4yr==1].sem1_gpa.mean():.2f}")
    print(f"    Non-grads avg GPA   : {df[df.graduated_4yr==0].sem1_gpa.mean():.2f}")
    print(f"\n  First-gen grad rate       : {df[df.first_gen_student==1].graduated_4yr.mean()*100:.1f}%")
    print(f"  Non-first-gen grad rate   : {df[df.first_gen_student==0].graduated_4yr.mean()*100:.1f}%")
    print(f"  Scholarship grad rate     : {df[df.scholarship==1].graduated_4yr.mean()*100:.1f}%")
    print(f"  No scholarship grad rate  : {df[df.scholarship==0].graduated_4yr.mean()*100:.1f}%")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":

    # ── Get cohort config from user ───────────────────────────────────────────
    cohorts = get_cohort_config()

    # ── Ask which years were COVID-impacted ───────────────────────────────────
    print("\n  Which years (if any) were disrupted by COVID or external events?")
    print("  Enter years separated by commas, or press Enter to skip.")
    print("  Example: 2020, 2021\n")
    covid_input = input("  COVID-affected years: ").strip()

    covid_years = []
    if covid_input:
        for y in covid_input.split(","):
            try:
                covid_years.append(int(y.strip()))
            except ValueError:
                pass

    # ── Output file path ──────────────────────────────────────────────────────
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    output_path  = os.path.join(script_dir, "student_graduation_dataset.csv")

    print(f"\n  Generating dataset...\n")
    df = build_dataset(cohorts, covid_years)
    df.to_csv(output_path, index=False)

    print_summary(df, cohorts)

    print(f"\n  ✅  File saved → {output_path}")
    print(f"\n  You're ready to build the model. See you on the other side! 🎓\n")