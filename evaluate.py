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
import time
import pandas as pd
import asyncio
import os
from pydantic import BaseModel
import instructor
import google.generativeai as genai
from dotenv import load_dotenv
import nest_asyncio
import pickle
nest_asyncio.apply()
load_dotenv()
genai.configure(api_key="AIzaSyC0mf55Y05nqbQLTfsCJTuqFk-K0OtLwC0")
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
            criteria="Evaluate how well the actual output addresses and aligns with the input query and retrieval context.",
            threshold=0.5,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT, LLMTestCaseParams.INPUT],
            model=self.model
        )

        faithfulness = GEval(
            name="Faithfulness",
            criteria="Assess whether the actual output faithfully represents information found in the retrieval context.",
            threshold=0.5,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
            model=self.model
        )

        context_precision = GEval(
            name="Contextual Precision",
            criteria="Evaluate how accurately the retrieval context identifies and ranks relevant nodes for the input query.",
            threshold=0.5,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
            model=self.model
        )

        context_recall = GEval(
            name="Contextual Recall",
            criteria="Assess how well the retrieval context covers the necessary information to generate the expected output.",
            threshold=0.5,
            evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
            model=self.model
        )
        correctness = GEval(
            name="Correctness",
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK"
            ],
            evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=self.model
        )
        # answer_relevancy = AnswerRelevancyMetric(
        #     threshold=0.5,
        #     model=self.model,
        # )
        
        # faithfulness = FaithfulnessMetric(
        #     threshold=0.5,
        #     model=self.model,
        # )
        
        # context_precision = ContextualPrecisionMetric(
        #     threshold=0.5,
        #     model=self.model,
        # )
        
        # context_recall = ContextualRecallMetric(
        #     threshold=0.5,
        #     model=self.model,
        # )

        answer_relevancy_scores = []
        faithfulness_scores = []
        context_precision_scores = []
        # with open('./context_precision_scores.pkl', 'rb') as f:
        #     context_precision_scores = pickle.load(f)
        context_recall_scores = []
        correctness_scores = []

        dataset = self.load_dataset()

        stop_processing = False
        for i in range(0, len(dataset.test_cases)):
            if stop_processing:
              break
            retries = 0
            while retries < 6:
                try:
                    # correctness.measure(dataset.test_cases[i])
                    # correctness_scores.append(correctness.score)
                    print(i)
                    answer_relevancy.measure(dataset.test_cases[i])
                    answer_relevancy_scores.append(answer_relevancy.score)
                    # faithfulness.measure(dataset.test_cases[i])
                    # faithfulness_scores.append(faithfulness.score)
                    # context_precision.measure(dataset.test_cases[i])
                    # context_precision_scores.append(context_precision.score)
                    # context_recall.measure(dataset.test_cases[i])
                    # context_recall_scores.append(context_recall.score)
                    # with open('correctness_scores.pkl', 'wb') as f:
                    #     pickle.dump(correctness_scores, f)
                    with open('answer_relevancy_scores.pkl', 'wb') as f:
                         pickle.dump(answer_relevancy_scores, f)
                    # with open('faithfulness_scores.pkl', 'wb') as f:
                    #      pickle.dump(faithfulness_scores, f)
                    # with open('context_precision_scores.pkl', 'wb') as f:
                    #      pickle.dump(context_precision_scores, f)
                    # with open('context_recall_scores.pkl', 'wb') as f:
                    #      pickle.dump(context_recall_scores, f)
                    break  # Exit the retry loop if successful
                except Exception as e:
                    if "'GenerateContentResponse' object has no attribute 'result'" in str(e):
                        answer_relevancy_scores.append(0)
                        with open('answer_relevancy_scores.pkl', 'wb') as f:
                            pickle.dump(answer_relevancy_scores, f)
                        break
                    if retries < 6:
                        print(f"Error encountered: {e}. Retrying in 30 seconds... (Attempt {retries}/{6})")
                        time.sleep(30)
                        retries += 1
                    else:
                        print(f"Error encountered: {e}. Max retries reached. Stopping the process.")
                        stop_processing = True
                        break
        self.output["answer_relevancy"] = answer_relevancy_scores
        # self.output["faithfulness"] = faithfulness_scores
        # self.output["context_precision"] = context_precision_scores
        # self.output["context_recall"] = context_recall_scores
        # self.output["correctness"] = correctness_scores

    def get_evaluate_output(self):
        df = pd.DataFrame.from_dict(self.output)
        df.to_csv("./output/" + self.filename,index=False)
        return df
        
    def get_relevance_score(self):
        average_answer_relevancy = sum(self.output["answer_relevancy"]) / len(self.output["answer_relevancy"])
        # average_faithfulness = sum(self.output["faithfulness"]) / len( self.output["faithfulness"])
        # average_context_precision = sum(self.output["context_precision"]) / len(self.output["context_precision"])
        # average_context_recall = sum(self.output["context_recall"]) / len(self.output["context_recall"])
        # average_correctness = sum(self.output["correctness"]) / len(self.output["correctness"])

        print("Trung bình của các chỉ số:")
        print("Answer Relevancy:", average_answer_relevancy)
        # print("Faithfulness:", average_faithfulness)
        # print("Context Precision:", average_context_precision)
        # print("Context Recall:", average_context_recall)
        # print("Correctness:", average_correctness)

class CustomChatModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        safety_settings
    ):
        self.model = model
        self.safety_settings = safety_settings

    def load_model(self):
        return ChatGoogleGenerativeAI(model=self.model,
                                      temperature=0,
                                      kwargs={"trust_remote_code": True},
                                      safety_settings=self.safety_settings,
                                      google_api_key='AIzaSyBEnlQJeeYq-je17SKyiyXrXuLIXEWREKU')
        # return ChatOllama(model=self.model,temperature=0)

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Model"

class CustomGeminiFlash(DeepEvalBaseLLM):
    def __init__(self,safety_settings):
        self.model = genai.GenerativeModel(model_name="models/gemini-1.5-flash",safety_settings=safety_settings)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()
        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini 1.5 Flash"
# Init
safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}
# gemini_chat = ChatGoogleGenerativeAI(model='gemini-1.5-pro',google_api_key=os.getenv("GOOGLE_API_KEY"),safety_settings=safety_settings,temperature=0.1)
# gemini_model = CustomChatModel(model="gemini-1.0-pro",safety_settings=safety_settings)
gemini_model = CustomGeminiFlash(safety_settings=safety_settings)
path_to_evaluate_data = "./data/testset_992.json"
evaluator = Evaluator(path_to_data=path_to_evaluate_data, model=gemini_model,filename="eval_992_ar.csv")

# Evaluation
asyncio.run(evaluator.eval())

# Get Dataframe
result = evaluator.get_evaluate_output()
evaluator.get_relevance_score()

