import csv
import random
import os
import string
from datetime import datetime

# ─────────────────────────────────────────────
#  NAME POOLS
# ─────────────────────────────────────────────
FIRST_NAMES_MALE = [
    "James", "Liam", "Noah", "Oliver", "Elijah", "William", "Benjamin", "Lucas",
    "Henry", "Alexander", "Mason", "Ethan", "Daniel", "Michael", "Logan", "Jackson",
    "Sebastian", "Jack", "Aiden", "Owen", "Samuel", "Ryan", "Nathan", "Carlos",
    "Miguel", "Diego", "Mateo", "Alejandro", "Luis", "Ricardo", "Ahmed", "Omar",
    "Ravi", "Arjun", "Rahul", "Wei", "Kai", "Jin", "Marcus", "Jordan"
]

FIRST_NAMES_FEMALE = [
    "Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia",
    "Harper", "Evelyn", "Abigail", "Emily", "Elizabeth", "Sofia", "Ella", "Madison",
    "Scarlett", "Victoria", "Luna", "Grace", "Chloe", "Penelope", "Layla", "Riley",
    "Zoey", "Nora", "Lily", "Eleanor", "Hannah", "Lillian", "Maria", "Valentina",
    "Camila", "Lucia", "Ana", "Fatima", "Sara", "Priya", "Ananya", "Mei"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Patel", "Shah", "Kumar", "Singh", "Sharma", "Chen", "Wang", "Kim", "Park", "Choi"
]

# ─────────────────────────────────────────────
#  DEPARTMENT → MAJORS & MINORS
# ─────────────────────────────────────────────
DEPT_MAJORS = {
    "Engineering":  ["Electrical Engineering", "Chemical Engineering",
                     "Aerospace Engineering", "Biomedical Engineering",
                     "Industrial Engineering"],
    "Medicine":     ["Pre-Medicine", "Nursing", "Pharmacy",
                     "Public Health", "Biomedical Sciences"],
    "Arts":         ["Fine Arts", "Music", "Theater",
                     "English Literature", "History", "Philosophy", "Graphic Design"],
    "Computers":    ["Computer Science", "Information Technology",
                     "Data Science", "Cybersecurity", "Software Engineering",
                     "Artificial Intelligence"],
    "Business":     ["Finance", "Marketing", "Management",
                     "Accounting", "Economics", "Entrepreneurship", "Human Resources"],
    "Civil":        ["Urban Planning", "Structural Engineering",
                     "Environmental Engineering", "Transportation Engineering",
                     "Geotechnical Engineering"],
    "Mechanical":   ["Robotics", "Thermodynamics", "Manufacturing Engineering",
                     "Automotive Engineering", "HVAC Engineering"]
}

# Minors can come from other departments
ALL_MINORS = [
    "Mathematics", "Statistics", "Psychology", "Sociology", "Communication",
    "Spanish", "Creative Writing", "Environmental Science", "Political Science",
    "Entrepreneurship", "Data Analytics", "Ethics", "Foreign Language",
    "Leadership Studies", "Health Science", "Film Studies", "Journalism",
    "Cybersecurity", "Project Management", "Public Relations"
]

LOCATIONS   = ["Edinburg", "Brownsville", "Harlingen"]
DEPARTMENTS = list(DEPT_MAJORS.keys())

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def generate_student_id(existing_ids: set) -> str:
    """Generate a unique 8-digit student ID."""
    while True:
        sid = str(random.randint(10000000, 99999999))
        if sid not in existing_ids:
            existing_ids.add(sid)
            return sid


def pick_grade() -> float:
    """
    Weighted grade distribution so data feels realistic:
    most students cluster around 2.0–3.5.
    """
    weights = [5, 10, 20, 30, 20, 10, 5]       # buckets across 1.0–4.0
    buckets = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5),
               (2.5, 3.0), (3.0, 3.5), (3.5, 3.8), (3.8, 4.0)]
    lo, hi = random.choices(buckets, weights=weights, k=1)[0]
    return round(random.uniform(lo, hi), 2)


def pick_hs_score(grade: float) -> int:
    """
    High school score loosely correlated with college GPA
    so the dataset has meaningful patterns for the model.
    """
    base = int(grade / 4.0 * 60) + 30          # maps 1.0→45, 4.0→90
    score = base + random.randint(-15, 15)
    return max(0, min(100, score))


def pick_graduated(gpa: float, hs_score: int, commute: str) -> str:
    """
    Graduation probability based on real academic logic:
      - GPA < 2.0  → cannot graduate (hard rule)
      - GPA drives base probability
      - High school score adds a secondary boost/penalty
      - Commuting students have a slight disadvantage
    """
    if gpa < 2.0:
        return "No"

    # Base probability from GPA
    if gpa < 2.5:
        prob = 0.25
    elif gpa < 3.0:
        prob = 0.50
    elif gpa < 3.5:
        prob = 0.65
    else:
        prob = 0.85

    # High school score adjustment (secondary factor)
    if hs_score >= 80:
        prob += 0.10
    elif hs_score < 60:
        prob -= 0.10

    # Commuters have slightly lower graduation rate
    if commute == "Yes":
        prob -= 0.05

    prob = max(0.0, min(1.0, prob))
    return "Yes" if random.random() < prob else "No"


def generate_student(year: int, existing_ids: set) -> dict:
    sex  = random.choice(["Male", "Female"])
    fname = random.choice(FIRST_NAMES_MALE if sex == "Male" else FIRST_NAMES_FEMALE)
    lname = random.choice(LAST_NAMES)
    dept  = random.choice(DEPARTMENTS)
    major = random.choice(DEPT_MAJORS[dept])
    minor = random.choice(ALL_MINORS)
    grade   = pick_grade()
    hs      = pick_hs_score(grade)
    commute = random.choice(["Yes", "No"])

    return {
        "student_id":           generate_student_id(existing_ids),
        "first_name":           fname,
        "last_name":            lname,
        "sex":                  sex,
        "department":           dept,
        "major":                major,
        "minor":                minor,
        "gpa":                  grade,
        "location":             random.choice(LOCATIONS),
        "commute":              commute,
        "high_school_score":    hs,
        "year":                 year,
        "graduated_in_4_years": pick_graduated(grade, hs, commute)
    }

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("   Student Academic Data Generator")
    print("=" * 50)

    # ── user inputs ──────────────────────────
    while True:
        try:
            year = int(input("\nEnter the academic year to generate data for (e.g. 2023): "))
            if 1900 <= year <= datetime.now().year + 5:
                break
            print("  Please enter a realistic year.")
        except ValueError:
            print("  Invalid input — please enter a 4-digit year.")

    while True:
        try:
            count = int(input("How many student records do you want to generate? "))
            if count > 0:
                break
            print("  Please enter a number greater than 0.")
        except ValueError:
            print("  Invalid input — please enter a whole number.")

    # ── generate records ─────────────────────
    print(f"\nGenerating {count} student records for year {year} ...")

    existing_ids: set = set()
    records = [generate_student(year, existing_ids) for _ in range(count)]

    # ── save to CSV ──────────────────────────
    filename = f"student_data_{year}.csv"
    fieldnames = [
        "student_id", "first_name", "last_name", "sex",
        "department", "major", "minor", "gpa",
        "location", "commute", "high_school_score", "year", "graduated_in_4_years"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # ── summary ──────────────────────────────
    print(f"\n✅  Done!  Saved → {filename}")
    print(f"   Total records  : {count}")
    print(f"   Academic year  : {year}")

    # quick breakdown
    dept_counts = {}
    for r in records:
        dept_counts[r["department"]] = dept_counts.get(r["department"], 0) + 1

    print("\n   Department breakdown:")
    for dept, cnt in sorted(dept_counts.items()):
        bar = "█" * (cnt * 20 // count)
        print(f"   {dept:<15} {cnt:>4} students  {bar}")

    print("\n   Sample rows (first 3):")
    print(f"   {'ID':<10} {'Name':<22} {'Dept':<14} {'GPA':<6} {'HS Score'}")
    print("   " + "-" * 60)
    for r in records[:3]:
        name = f"{r['first_name']} {r['last_name']}"
        print(f"   {r['student_id']:<10} {name:<22} {r['department']:<14} {r['gpa']:<6} {r['high_school_score']}")

    print(f"\n   Open '{filename}' in Excel or load it in Python to get started!\n")


if __name__ == "__main__":
    main()