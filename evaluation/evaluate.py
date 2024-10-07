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
from openai import OpenAI
nest_asyncio.apply()
load_dotenv()
genai.configure(api_key="AIzaSyCijbw4C_WoEUPVPtss6uZ_qe9-GnbGaoY")
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
            evaluation_steps=[
                "Check if 'actual output' directly answers the 'input'",
                "Penalize irrelevant information that does not contribute to answering the 'input'",
                "You can overlook minor details if they don't detract from the relevancy of the answer",
                "Focus on whether the core response addresses the key aspects of the 'input'"
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
            model=self.model,
        )

        faithfulness = GEval(
            name="Faithfulness",
            evaluation_steps=[
                "Check if the facts in 'actual output' are directly supported by the 'retrieval context'",
                "Penalize any facts or information in 'actual output' that are not grounded in the 'retrieval context'",
                "You should also penalize any hallucination or fabricated details in 'actual output'",
                "Vague language is acceptable as long as it does not introduce unsupported information"
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
            model=self.model,
        )

        # context_precision = GEval(
        #     name="Context Precision",
        #     criteria="""Evaluate how precisely the retrieval context relates to the input query, avoiding irrelevant information and providing focused, relevant details.""",
        #     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        #     model=self.model,
        # )
        # context_recall = GEval(
        #     name="Context Recall",
        #     criteria="""Assess how well the retrieval context covers the necessary information to match the expected output, avoiding omissions and irrelevant details.""",
        #     evaluation_params=[LLMTestCaseParams.RETRIEVAL_CONTEXT, LLMTestCaseParams.EXPECTED_OUTPUT],
        #     model=self.model,
        # )
        correctness = GEval(
            name="Correctness",
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK"
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=self.model,
        )
        # answer_relevancy = AnswerRelevancyMetric(
        #     threshold=0.7,
        #     model=self.model,
        # )
        
        # faithfulness = FaithfulnessMetric(
        #     threshold=0.7,
        #     model=self.model,
        # )
        
        # context_precision = ContextualPrecisionMetric(
        #     threshold=0.7,
        #     model=self.model,
        # )
        
        # context_recall = ContextualRecallMetric(
        #     threshold=0.7,
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
                    print(i)
                    await asyncio.gather(
                        faithfulness.a_measure(dataset.test_cases[i]),
                        correctness.a_measure(dataset.test_cases[i]),
                        answer_relevancy.a_measure(dataset.test_cases[i]),
                    )
                    print("Metrics finished!")
                    # context_precision.measure(dataset.test_cases[i])
                    # context_recall.measure(dataset.test_cases[i])
                    
                    correctness_scores.append(correctness.score)
                    answer_relevancy_scores.append(answer_relevancy.score)
                    faithfulness_scores.append(faithfulness.score)
                    # context_precision_scores.append(context_precision.score)
                    # context_recall_scores.append(context_recall.score)
                    with open('50_200_1.pkl', 'wb') as f:
                        pickle.dump([answer_relevancy_scores,correctness_scores,faithfulness_scores], f)
                    # with open('correctness.pkl', 'wb') as f:
                    #      pickle.dump(correctness_scores, f)
                    # with open('faithfulness_scores.pkl', 'wb') as f:
                    #      pickle.dump(faithfulness_scores, f)
                    # with open('context_precision_scores.pkl', 'wb') as f:
                    #      pickle.dump(context_precision_scores, f)
                    # with open('context_recall_scores.pkl', 'wb') as f:
                    #      pickle.dump(context_recall_scores, f)
                    break  # Exit the retry loop if successful
                except Exception as e:
                    # if "'GenerateContentResponse' object has no attribute 'result'" in str(e):
                    #     correctness_scores.append(0)
                    #     with open('correctness_scores.pkl', 'wb') as f:
                    #         pickle.dump(correctness_scores, f)
                    #     break
                    if retries < 6:
                        print(f"Error encountered: {e}. Retrying in 10 seconds... (Attempt {retries}/{6})")
                        time.sleep(10)
                        retries += 1
                    else:
                        print(f"Error encountered: {e}. Max retries reached. Stopping the process.")
                        stop_processing = True
                        break
        self.output["answer_relevancy"] = answer_relevancy_scores
        self.output["faithfulness"] = faithfulness_scores
        # self.output["context_precision"] = context_precision_scores
        # self.output["context_recall"] = context_recall_scores
        self.output["correctness"] = correctness_scores

    def get_evaluate_output(self):
        df = pd.DataFrame.from_dict(self.output)
        df.to_csv("./output/" + self.filename,index=False)
        return df
        
    def get_relevance_score(self):
        average_answer_relevancy = sum(self.output["answer_relevancy"]) / len(self.output["answer_relevancy"])
        average_faithfulness = sum(self.output["faithfulness"]) / len( self.output["faithfulness"])
        # average_context_precision = sum(self.output["context_precision"]) / len(self.output["context_precision"])
        # average_context_recall = sum(self.output["context_recall"]) / len(self.output["context_recall"])
        average_correctness = sum(self.output["correctness"]) / len(self.output["correctness"])

        print("Trung bình của các chỉ số:")
        print("Answer Relevancy:", average_answer_relevancy)
        print("Faithfulness:", average_faithfulness)
        # print("Context Precision:", average_context_precision)
        # print("Context Recall:", average_context_recall)
        print("Correctness:", average_correctness)

class CustomGeminiFlash(DeepEvalBaseLLM):
    def __init__(self):
        self.safe_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.model = genai.GenerativeModel(model_name="models/gemini-1.5-flash",safety_settings=self.safe_settings)
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
    
class CustomModel(DeepEvalBaseLLM):
    def __init__(self,api_key,base_url,model_name):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
    def load_model(self):
        model = OpenAI(api_key=self.api_key,base_url=self.base_url)
        return model

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        resp = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
        )
        return resp.choices[0].message.content
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name
# Init
# gemini_chat = ChatGoogleGenerativeAI(model='gemini-1.5-pro',google_api_key=os.getenv("GOOGLE_API_KEY"),safety_settings=safety_settings,temperature=0.1)
# model = CustomGeminiFlash()
#model = CustomModel(api_key="2062b0f5-e861-4e72-9b78-6a348dba0412",base_url="https://api.arliai.com/v1",model_name="Meta-Llama-3.1-8B-Instruct")
model = CustomModel(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzI4NjU5MDc4LCJpYXQiOjE3MjgwNTQyNzgsInN1YiI6IjcxNTY1N2M2LWE1OTgtNGM4Ny1hNjBhLTJiMjc1MzAwYjE4YiIsImVtYWlsIjoiZ29nb3J1bjIzNUBnbWFpbC5jb20iLCJwaG9uZSI6IiIsImFwcF9tZXRhZGF0YSI6eyJwcm92aWRlciI6ImdlbmVyaWMxIiwicHJvdmlkZXJzIjpbImdlbmVyaWMxIl19LCJ1c2VyX21ldGFkYXRhIjp7ImVtYWlsIjoiZ29nb3J1bjIzNUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaXNzIjoiaHR0cHM6Ly93d3cub3ZoLmNvbS9hdXRoL29hdXRoMi91c2VyIiwicGhvbmVfdmVyaWZpZWQiOmZhbHNlLCJzdWIiOiJuZzEyOTA0OC1vdmgifSwicm9sZSI6ImF1dGhlbnRpY2F0ZWQiLCJhYWwiOiJhYWwxIiwiYW1yIjpbeyJtZXRob2QiOiJvYXV0aCIsInRpbWVzdGFtcCI6MTcyODA1NDI3OH1dLCJzZXNzaW9uX2lkIjoiZGE2MWEzNTEtMzg1YS00ZGJhLTllZjktZDBjM2Y5NDBiMmI4In0.d8BBSxBfg_q9x5P8P3wofX0BNq4tnsXWp8ASgXNzy_w",base_url="https://llama-3-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",model_name="Meta-Llama-3-70B-Instruct")
evaluator = Evaluator(path_to_data="./data/100_400_processed_non_parahrased_mistral.json", model=model,filename="100_400_processed_non_parahrased_mistral_result.csv")

# Evaluation
asyncio.run(evaluator.eval())

# Get Dataframe
result = evaluator.get_evaluate_output()
evaluator.get_relevance_score()

