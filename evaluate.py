from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
import time
import pandas as pd
import asyncio

from dotenv import load_dotenv
load_dotenv()

class Evaluator():
    def __init__(
            self,
            path_to_data,
            model,
            filename,
    ):
        self.path_to_data = path_to_data
        self.output = {}
        self.model = model
        self.file_name = filename

    def load_json(self):
        with open(self.path_to_data, "r", encoding="utf-8") as infile: 
            data = json.load(infile)
        self.output = data 
        return data
    
    def load_dataset(self):
        dataset = EvaluationDataset()
        data = self.load_json()

        for i in range(0, len(data["questions"])):
            testcase = LLMTestCase(
                input=data["questions"][i], 
                actual_output=data["answers"][i],
                expected_output=data["ground_truths"][i],
                retrieval_context=data["contexts"][i])
            dataset.add_test_case(testcase)
        return dataset
    
    async def eval(self):
        answer_relevancy = GEval(
            name="Answer Relevancy",
            criteria="Answer Relevancy - evaluating how relevant the actual output is compared to the input.",
            evaluation_params=[LLMTestCaseParams.INPUT, 
                            LLMTestCaseParams.ACTUAL_OUTPUT, 
                            LLMTestCaseParams.EXPECTED_OUTPUT,
                            LLMTestCaseParams.RETRIEVAL_CONTEXT],
            model=self.model
        )

        faithfulness = GEval(
            name="Faithfulness",
            criteria="Faithfulness - evaluating  whether the actual output factually aligns with the contents of the retrieval context.",
            evaluation_params=[LLMTestCaseParams.INPUT, 
                            LLMTestCaseParams.ACTUAL_OUTPUT, 
                            LLMTestCaseParams.EXPECTED_OUTPUT,
                            LLMTestCaseParams.RETRIEVAL_CONTEXT],
            model=self.model
        )

        context_precision = GEval(
            name="Context precision",
            criteria="Context precision - evaluating whether nodes in the retrieval context that are relevant to the given input are ranked higher than irrelevant ones.",
            evaluation_params=[LLMTestCaseParams.INPUT, 
                            LLMTestCaseParams.ACTUAL_OUTPUT, 
                            LLMTestCaseParams.EXPECTED_OUTPUT,
                            LLMTestCaseParams.RETRIEVAL_CONTEXT],
            model=self.model
        )

        context_recall = GEval(
            name="Context recall",
            criteria="Context recall -  evaluating the extent of which the retrieval context aligns with the expected output.",
            evaluation_params=[LLMTestCaseParams.INPUT, 
                            LLMTestCaseParams.ACTUAL_OUTPUT, 
                            LLMTestCaseParams.EXPECTED_OUTPUT,
                            LLMTestCaseParams.RETRIEVAL_CONTEXT],
            model=self.model
        )

        answer_relevancy_scores = []
        faithfulness_scores = []
        context_precision_scores = []
        context_recall_scores = []

        dataset = self.load_dataset()

        for i in range(0, len(dataset.test_cases)):
            answer_relevancy.measure(dataset.test_cases[i])
            faithfulness.measure(dataset.test_cases[i])
            context_precision.measure(dataset.test_cases[i])
            context_recall.measure(dataset.test_cases[i])
            answer_relevancy_scores.append(answer_relevancy.score)
            faithfulness_scores.append(faithfulness.score)
            context_precision_scores.append(context_precision.score)
            context_recall_scores.append(context_recall.score)
            time.sleep(3)
        
        self.output["answer_relevancy"] = answer_relevancy_scores
        self.output["faithfulness"] = faithfulness_scores
        self.output["context_precision"] = context_precision_scores
        self.output["context_recall"] = context_recall_scores

    def get_evaluate_output(self):
        df = pd.DataFrame.from_dict(self.output)
        df.to_csv("./output" + self.file_name)
        return df
        
    def get_relevance_score(self):
        average_answer_relevancy = sum(self.output["answer_relevancy"]) / len(self.output["answer_relevancy"])
        average_faithfulness = sum(self.output["faithfulness"]) / len( self.output["faithfulness"])
        average_context_precision = sum(self.output["context_precision"]) / len(self.output["context_precision"])
        average_context_recall = sum(self.output["context_precision"]) / len(self.output["context_precision"])

        print("Trung bình của các chỉ số:")
        print("Answer Relevancy:", average_answer_relevancy)
        print("Faithfulness:", average_faithfulness)
        print("Context Precision:", average_context_precision)
        print("Context Recall:", average_context_recall)

class CustomModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model
    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom model"
    
# Init
gemini_chat = ChatGoogleGenerativeAI(model='gemini-pro',google_api_key="AIzaSyD3NCZLaMXUpG1UvStJMN8eYB1QeleOg6Y",temperature=0.1)
model = CustomModel(gemini_chat)
path_to_evaluate_data = "./data/testset.json"
evaluator = Evaluator(path_to_data=path_to_evaluate_data, model=model)

# Evaluation
asyncio.run(evaluator.eval())

# Get Dataframe
result = evaluator.get_evaluate_output()
result.to_csv("./data/eval_class_19_4.csv")
evaluator.get_relevance_score()

