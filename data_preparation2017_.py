import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

def clean_correct_column(df):
    if 'correct' not in df.columns:
        print("correct column not found, skip cleaning")
        return df

    df = df.copy()

    before_stats = df['correct'].value_counts(dropna=False).sort_index()
    print("\nDistribution of correct before cleaning:")
    for val, cnt in before_stats.items():
        print(f"   - {val}: {cnt:,} rows")

    df['correct'] = pd.to_numeric(df['correct'], errors='coerce')
    df['correct'] = df['correct'].apply(lambda x: x if x in (0, 1) else 0)

    after_stats = df['correct'].value_counts(dropna=False).sort_index()
    print("\nDistribution of correct after cleaning:")
    for val, cnt in after_stats.items():
        print(f"   - {val}: {cnt:,} rows")

    return df

def main():
    output_dir = "processed_data_2017"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder created: '{output_dir}'")
    else:
        print(f"Folder '{output_dir}' already exists")

    csv_path = r'./original_data/anonymized_full_release_competition_dataset.csv'
    print("\nReading raw data and detecting column names...")

    try:
        df_all = pd.read_csv(csv_path, low_memory=False, nrows=0)
        real_columns = df_all.columns.tolist()
        print(f"Original CSV columns:")
        for i, col in enumerate(real_columns):
            print(f"   [{i}] '{col}'")

        raw_columns = [
            'startTime', 'endTime', 'timeTaken','attemptCount', 'hintCount',
            'studentId', 'skill', 'problemId', 'correct'
        ]

        column_mapping = {
            'studentId': 'student_id',
            'problemId': 'exercise_id',
            'startTime': 'start_time',
            'endTime': 'end_time',
            'attemptCount': 'attempt_count',
            'hintCount': 'hint_count',
            'skill': 'skill',
            'correct': 'correct',
            'timeTaken': 'response_time'
        }

        print("Loading dataset...")
        df = pd.read_csv(
            csv_path,
            usecols=raw_columns,
            low_memory=False
        )
        print(f"Data loading finished, total {len(df):,} rows, {df.shape[1]} columns.")

        print("Removing rows with missing values...")
        before = len(df)
        df.dropna(subset=raw_columns, inplace=True)
        after = len(df)
        print(f"Removed {before - after:,} rows, remaining {after:,} rows")

        df.rename(columns=column_mapping, inplace=True)

        print("\nData types after column rename:")
        for col in df.columns:
            print(f"   - {col}: {df[col].dtype}")

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Data file '{csv_path}' not found.")
    except Exception as e:
        raise RuntimeError(f"Error reading data: {e}")

    df = clean_correct_column(df)

    print("Sort records by student_id and start_time...")
    df = df.sort_values(by=['student_id', 'start_time'], ascending=[True, True]).reset_index(drop=True)

    print("Calculating time_interval (current start time - previous start time)...")
    df['prev_start'] = df.groupby('student_id')['start_time'].shift(1)
    df['time_interval'] = df['start_time'] - df['prev_start']

    negative_ti = (df['time_interval'] < 0).sum()
    if negative_ti > 0:
        print(f"Warning: {negative_ti} negative time_interval entries, set to 0")
    else:
        print("No negative time_interval detected")

    df['time_interval'] = df['time_interval'].fillna(0).clip(lower=0)
    df.drop(columns=['prev_start'], inplace=True)

    final_columns = [
        'student_id', 'skill', 'exercise_id',
        'start_time', 'end_time', 'response_time',
        'correct', 'hint_count', 'attempt_count', 'time_interval'
    ]

    df_final = df[final_columns].copy()
    print("\nFilter students with less than 3 exercise records...")
    stu_count = df_final.groupby('student_id').size()
    valid_students = stu_count[stu_count >= 3].index
    df_final = df_final[df_final['student_id'].isin(valid_students)].copy()
    print(f"Filter completed, remaining students: {len(valid_students):,}, total rows: {len(df_final):,}")

    output_file = os.path.join(output_dir, 'processed_data17.csv')
    df_final.to_csv(output_file, index=False)
    print(f"Processed data saved to: {output_file}")

    print("\n" + "=" * 50)
    print("Starting data analysis...")
    print("=" * 50)

    student_counts = df['student_id'].value_counts().sort_values(ascending=False)
    total_students = len(student_counts)
    print(f"Total distinct students: {total_students:,}")

    plt.figure(figsize=(12, 6))
    top_students = student_counts.head(30)
    sns.barplot(x=top_students.values, y=top_students.index.astype(str), hue=top_students.index.astype(str), palette="viridis", legend=False)
    plt.title("Number of Exercises per Student (Top 30)", fontsize=14)
    plt.xlabel("Number of Exercises")
    plt.ylabel("Student ID")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "student_activity_top30.png"), dpi=150)
    plt.show()

    if 'skill' in df.columns and not df['skill'].isnull().all():
        skill_counts = df['skill'].value_counts()
        total_skills = len(skill_counts)
        print(f"Total distinct skills: {total_skills:,}")

        plt.figure(figsize=(10, 6))
        top_skills = skill_counts.head(20)
        sns.barplot(x=top_skills.values, y=top_skills.index, hue=top_skills.index, palette="magma", legend=False)
        plt.title("Skill Frequency Distribution (Top 20)", fontsize=14)
        plt.xlabel("Frequency")
        plt.ylabel("Skill Name")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "skill_frequency_top20.png"), dpi=150)
        plt.show()
    else:
        print("No valid skill data, skip skill analysis")

    plt.figure(figsize=(10, 6))
    plt.hist(student_counts.values, bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Exercise Counts per Student", fontsize=14)
    plt.xlabel("Number of Exercises per Student")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "student_count_histogram.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(10, 6))
    top_20_students = df['student_id'].value_counts().head(20).index
    top_students_data = df[df['student_id'].isin(top_20_students)].copy()

    top_students_data['hours'] = top_students_data['time_interval'] / 3600
    avg_intervals = (
        top_students_data[top_students_data['time_interval'] > 0]
        .groupby('student_id')['hours']
        .mean()
        .sort_values()
    )

    sns.barplot(
        x=avg_intervals.values,
        y=[f"Student {i + 1}" for i in range(len(avg_intervals))],
        hue=[f"Student {i + 1}" for i in range(len(avg_intervals))],
        palette="coolwarm",
        legend=False
    )

    plt.title('Avg. Time Interval per Student (Top 20 Active)', fontsize=12)
    plt.xlabel('Average Time Interval (hours)', fontsize=11)
    plt.ylabel('Student Rank', fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top20_time_interval.png"), dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()