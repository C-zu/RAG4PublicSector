from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory
)
import json
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval, AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from langchain_community.chat_models import ChatOllama
import time
import pandas as pd
import asyncio
import os
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()
load_dotenv()

class Evaluator():
    def __init__(
            self,
            path_to_data,
            model,
            filename
    ):
        self.path_to_data = path_to_data
        self.output = {}
        self.model = model
        self.filename = filename

    def load_json(self):
        with open(self.path_to_data, "r", encoding="utf-8") as infile: 
            data = json.load(infile)
        sliced_dict = {key: value[:5] for key, value in data.items()}
        self.output = sliced_dict 
        return sliced_dict
    
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
        # answer_relevancy = GEval(
        #     name="Answer Relevancy",
        #     criteria="Evaluate how well the actual output addresses and aligns with the input query and retrieval context.",
        #     threshold=0.5,
        #     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        #     model=self.model
        # )

        # faithfulness = GEval(
        #     name="Faithfulness",
        #     criteria="Assess whether the actual output faithfully represents information found in the retrieval context.",
        #     threshold=0.5,
        #     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        #     model=self.model
        # )

        # context_precision = GEval(
        #     name="Contextual Precision",
        #     criteria="Evaluate how accurately the retrieval context identifies and ranks relevant nodes for the input query.",
        #     threshold=0.5,
        #     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        #     model=self.model
        # )

        # context_recall = GEval(
        #     name="Contextual Recall",
        #     criteria="Assess how well the retrieval context covers the necessary information to generate the expected output.",
        #     threshold=0.5,
        #     evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        #     model=self.model
        # )
        
        answer_relevancy = AnswerRelevancyMetric(
            threshold=0.5,
            model=self.model,
        )
        
        faithfulness = FaithfulnessMetric(
            threshold=0.5,
            model=self.model,
        )
        
        context_precision = ContextualPrecisionMetric(
            threshold=0.5,
            model=self.model,
        )
        
        context_recall = ContextualRecallMetric(
            threshold=0.5,
            model=self.model,
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
        df.to_csv("./output/" + self.filename)
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

class GeminiChatModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return ChatGoogleGenerativeAI(model=self.model,
                                      temperature=0,
                                      google_api_key="AIzaSyBg-vNAHIFM52uannoCK8sruEQn3zzh6Ec")
        # return ChatOllama(model=self.model,temperature=0)

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Gemini 1.0 Model"
    
# Init
# safety_settings = {
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
# }
# gemini_chat = ChatGoogleGenerativeAI(model='gemini-1.5-pro',google_api_key=os.getenv("GOOGLE_API_KEY"),safety_settings=safety_settings,temperature=0.1)
gemini_model = GeminiChatModel(model="gemini-1.5-pro")
path_to_evaluate_data = "./data/testset.json"
evaluator = Evaluator(path_to_data=path_to_evaluate_data, model=gemini_model,filename="eval_12_7.csv")

# Evaluation
asyncio.run(evaluator.eval())

# Get Dataframe
result = evaluator.get_evaluate_output()
evaluator.get_relevance_score()

