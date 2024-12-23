import os
import json

questions = [
    "Who was the eldest among the Pandava brothers?",
    "What were the conditions of the dice game between the Pandavas and Kauravas, and what was wagered in each round?",
    "Detail the various divine weapons (astras) received by Arjuna during his training, from whom he received them, and under what circumstances?",
    "Why did Krishna choose to be Arjuna's charioteer rather than fight in the war himself?",
    "How does Bhishma's vow of celibacy and his devotion to his father's happiness reflect the theme of duty versus personal desire in the Mahabharata?",
    "Analyze how the concept of Time (Kala) is portrayed as both a destroyer and preserver throughout the epic, using specific examples from different parvas?",
    "Describe the events leading up to and including the burning of the Khandava Forest. What were its immediate and long-term consequences?",
    "Compare and contrast the characters of Karna and Arjuna, analyzing their similarities, differences, and how their parallel journeys contribute to the epic's themes.",
    "Examine the role of women in the Mahabharata, focusing on Draupadi, Kunti, and Gandhari. How do their actions and choices drive the narrative and reflect the epic's views on gender, power, and dharma?",
]

input_directory = "results"  
output_directory = os.path.join(input_directory, "final1")
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.endswith(".json"): 
        input_filepath = os.path.join(input_directory, filename)

        with open(input_filepath, "r") as f:
            data = json.load(f)

        output_data = []
        for question in questions:
          
            for key, value in data['test_cases_lookup_map'].items():
                if question in key:
                    scores_and_names = [
                            {
                                "metric_name": metric["metric_data"]["name"],
                                "score": metric["metric_data"]["score"]
                            }
                            for metric in value["cached_metrics_data"]
                        ]
                    print(question, scores_and_names)
                    output_data.append({"question": question, "metrics": scores_and_names})
                    break
                
            
        print(output_data)
        output_filename = f"{filename}"
        output_filepath = os.path.join(output_directory, output_filename)
        with open(output_filepath, "w") as f:
            json.dump(output_data, f, indent=4)

print(f"Processed files saved in: {output_directory}")