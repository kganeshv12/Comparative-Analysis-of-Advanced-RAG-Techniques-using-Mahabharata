import os
import json

input_directory = "results/final1"
output_directory = os.path.join(input_directory, "final2")
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.endswith(".json"):
        input_filepath = os.path.join(input_directory, filename)

        with open(input_filepath, "r") as f:
            data = json.load(f)

        final_op = {}

        category_data = {"Faithfulness": 0, "Contextual Relevancy": 0, "Answer Relevancy": 0}
            
            
        for item in data:
            category_data["Faithfulness"] += item["metrics"][0]["score"]
            category_data["Contextual Relevancy"] += item["metrics"][1]["score"]
            category_data["Answer Relevancy"] += item["metrics"][2]["score"]
                
    
        category_data["Faithfulness"] /= 9
        category_data["Contextual Relevancy"] /= 9
        category_data["Answer Relevancy"] /= 9
            

        # Add final aggregate scores
        # final_op["Final"] = {
        #     category: metrics.values() for category, metrics in final_op.items()
        # }

        output_filepath = os.path.join(output_directory, filename)
        with open(output_filepath, "w") as f:
            json.dump(category_data, f, indent=4)