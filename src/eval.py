from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from pathlib import Path
import json
import os



def evaluate_1(rag_chain, retriever):

    os.environ["OPENAI_API_KEY"] = ""

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

    answer_list = []
    context_list = []
    for q in questions:
        answer = rag_chain.invoke(q)
        print(answer)
        answer_list.append(answer)
        context_list.append([docs.page_content for docs in retriever.get_relevant_documents(q)])

    test_cases = [
    LLMTestCase(
    input=questions[i],
    actual_output=answer_list[i],
    retrieval_context=context_list[i],
    )for i in range(0,len(questions))
    ]     
    print(test_cases)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7,  model = 'gpt-4o-mini', include_reason = True) 
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model = 'gpt-4o-mini', include_reason = True)
    contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7, model = 'gpt-4o-mini', include_reason = True)




    evaluate(test_cases, [
    faithfulness_metric,
    contextual_relevancy_metric,
    answer_relevancy_metric])



def evaluate_2(method, model):

    os.environ["OPENAI_API_KEY"] = ""

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

    answer_list = []
    context_list = []
    for q in questions:
        answer, context = method(model,q)
        print(answer)
        answer_list.append(answer)
        temp = list()
        temp.append(context)
        context_list.append(temp)
        print(context_list)
        print(type(context_list))
        print(temp)
        

    test_cases = [
    LLMTestCase(
    input=questions[i],
    actual_output=answer_list[i],
    retrieval_context= context_list[i],
    )for i in range(0,len(questions))
    ]     
    print(test_cases)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7,  model = 'gpt-4o-mini', include_reason = True) 
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model = 'gpt-4o-mini', include_reason = True)
    contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7, model = 'gpt-4o-mini', include_reason = True)

    evaluate(test_cases, [
    faithfulness_metric,
    contextual_relevancy_metric,
    answer_relevancy_metric])

    cache_file = Path(".deepeval-cache.json")
    if cache_file.exists():
        with open(cache_file, "r") as f:
            results = json.load(f)

        file_name = "./results/"+method.__name__+ "_" + model+".json"
        
        json.dump(results, open(file_name, "w"), indent=4)

    