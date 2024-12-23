import os
import json

factual = [
    "Who was the eldest among the Pandava brothers?",
    "What were the conditions of the dice game between the Pandavas and Kauravas, and what was wagered in each round?",
    "Detail the various divine weapons (astras) received by Arjuna during his training, from whom he received them, and under what circumstances?",
]

interpretive = [
    "Why did Krishna choose to be Arjuna's charioteer rather than fight in the war himself?",
    "How does Bhishma's vow of celibacy and his devotion to his father's happiness reflect the theme of duty versus personal desire in the Mahabharata?",
    "Analyze how the concept of Time (Kala) is portrayed as both a destroyer and preserver throughout the epic, using specific examples from different parvas?",
]

analytical = [
    "Describe the events leading up to and including the burning of the Khandava Forest. What were its immediate and long-term consequences?",
    "Compare and contrast the characters of Karna and Arjuna, analyzing their similarities, differences, and how their parallel journeys contribute to the epic's themes.",
    "Examine the role of women in the Mahabharata, focusing on Draupadi, Kunti, and Gandhari. How do their actions and choices drive the narrative and reflect the epic's views on gender, power, and dharma?",
]

input_directory = "results/final1"
output_directory = os.path.join(input_directory, "Question_Based")
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.endswith(".json"):
        input_filepath = os.path.join(input_directory, filename)

        with open(input_filepath, "r") as f:
            data = json.load(f)

        final_op = {}

        for category, questions in zip(
            ["Factual", "Interpretive", "Long Answer"],
            [factual, interpretive, analytical],
        ):
            category_data = {"Faithfulness": 0, "Contextual Relevancy": 0, "Answer Relevancy": 0}
            category_count = 0
            for q in questions:
                for item in data:
                    if q in item["question"]:
                        category_data["Faithfulness"] += item["metrics"][0]["score"]
                        category_data["Contextual Relevancy"] += item["metrics"][1]["score"]
                        category_data["Answer Relevancy"] += item["metrics"][2]["score"]
                        category_count += 1
            if category_count > 0:
                category_data["Faithfulness"] /= category_count
                category_data["Contextual Relevancy"] /= category_count
                category_data["Answer Relevancy"] /= category_count
            final_op[category] = category_data

        # Add final aggregate scores
        final_op["Final"] = {
            category: sum(metrics.values()) / 3 for category, metrics in final_op.items()
        }

        output_filepath = os.path.join(output_directory, filename)
        with open(output_filepath, "w") as f:
            json.dump(final_op, f, indent=4)