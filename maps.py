import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV file
file_path = "ml_grouped_topics_questions.csv"  # Replace with your actual CSV file path
df = pd.read_csv(file_path)

# Ensure the relevant columns are correctly named in your CSV
# Assuming your CSV has columns "Topic" and "Question"
if 'Topic' not in df.columns:
    raise ValueError("CSV must have a column named 'Topic'.")

# Count questions per topic
topic_counts = df['Topic'].value_counts()

# Pie chart for topic distribution
plt.figure(figsize=(8, 8))
colors = plt.cm.Paired(range(len(topic_counts)))  # Use a visually pleasing colormap
explode = [0.1 if i == topic_counts.idxmax() else 0 for i in topic_counts.index]  # Highlight largest slice

plt.pie(topic_counts, labels=topic_counts.index, autopct='%1.1f%%',
        startangle=140, colors=colors, explode=explode, shadow=True, textprops={'fontsize': 12})

plt.show()
