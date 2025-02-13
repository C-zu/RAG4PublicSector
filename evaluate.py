from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_genai import (
    HarmBlockThreshold,
    HarmCategory
)
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCaseParams
# from deepeval.test_case import LLMTestCase
import time
import pandas as pd
import asyncio
<<<<<<< HEAD
# import os
from pydantic import BaseModel
import instructor
import google.generativeai as genai
from dotenv import load_dotenv
import nest_asyncio
import pickle
from openai import OpenAI
import ast
# import numpy as np
nest_asyncio.apply()
load_dotenv()
# genai.configure(api_key="AIzaSyCijbw4C_WoEUPVPtss6uZ_qe9-GnbGaoY")
=======

from dotenv import load_dotenv
load_dotenv()

>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
class Evaluator():
    def __init__(
            self,
            path_to_data,
            model,
<<<<<<< HEAD
            filename
=======
            filename,
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
    ):
        self.path_to_data = path_to_data
        self.output = {}
        self.model = model
<<<<<<< HEAD
        self.filename = filename
        self.df = None
        self.load_csv()

    def load_csv(self):
        self.df = pd.read_csv(self.path_to_data)
=======
        self.file_name = filename

    def load_json(self):
        with open(self.path_to_data, "r", encoding="utf-8") as infile: 
            data = json.load(infile)
        self.output = data 
        return data
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
    
    def load_dataset(self):
        dataset = EvaluationDataset()
        # data = self.load_csv()
        # if os.path.exists("./output/error_score.csv"):
        #     old_error = pd.read_csv("./output/error_score.csv")
        # data = data[~data['questions'].isin(old_error['questions'])]
        datadict = {}
        datadict["questions"] = self.df['questions'].to_list()
        datadict["answers"] = self.df['answers'].to_list()
        datadict["ground_truths"] = self.df['ground_truths'].to_list()
        datadict["contexts"] = self.df['contexts'].to_list()
        # self.output = datadict
        for i in range(0, len(datadict["questions"])):
            testcase = LLMTestCase(
                input=datadict["questions"][i], 
                actual_output=datadict["answers"][i],
                expected_output=datadict["ground_truths"][i],
                retrieval_context=ast.literal_eval(datadict["contexts"][i]))
            dataset.add_test_case(testcase)
        return dataset
    
    async def eval(self):
        # answer_relevancy = GEval(
        #     name="Answer Relevancy",
        #     evaluation_steps=[
        #         "Check if 'ACTUAL_OUTPUT' answers the 'INPUT' directly.",
        #         "Penalize irrelevant content in the response.",
        #         "Ignore minor details if they don't affect relevancy.",
        #         "Ensure the response addresses the key aspects of 'INPUT'."
        #     ],
        #     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
        #     model=self.model[1],
        # )

        # faithfulness = GEval(
        #     name="Faithfulness",
        #     evaluation_steps=[
        #         "Ensure 'ACTUAL_OUTPUT' facts are supported by 'RETRIEVAL_CONTEXT'.",
        #         "Penalize unsupported facts in 'ACTUAL_OUTPUT'.",
        #         "Allow vague language if it avoids introducing unsupported information."
        #     ],
        #     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        #     model=self.model[2],
        # )

        # context_precision = GEval(
        #     name="Context Precision",
        #     evaluation_steps=[
        #         "Rank the most relevant nodes in 'RETRIEVAL_CONTEXT' higher for the given 'INPUT'.",
        #         "Ensure the most important information directly related to 'INPUT' is prioritized.",
        #         "Penalize cases where irrelevant or less relevant nodes are ranked higher than the most relevant ones."
        #     ],
        #     evaluation_params=[LLMTestCaseParams.RETRIEVAL_CONTEXT, LLMTestCaseParams.INPUT],
        #     model=self.model[1],
        #     verbose_mode=True
        # )

        # context_recall = GEval(
        #     name="Context Recall",
        #     evaluation_steps=[
        #         "Ensure that 'RETRIEVAL_CONTEXT' contains all the critical information required to generate the 'EXPECTED_OUTPUT'.",
        #         "Verify that the majority of relevant nodes (e.g., key facts, entities, or phrases) in 'RETRIEVAL_CONTEXT', which are essential for generating the 'EXPECTED_OUTPUT', are present, accurate, and complete."
        #     ],
        #     evaluation_params=[LLMTestCaseParams.RETRIEVAL_CONTEXT, LLMTestCaseParams.EXPECTED_OUTPUT],
        #     model=self.model[2],
        #     verbose_mode=True
        # )
        correctness = GEval(
            name="Correctness",
            evaluation_steps=[
                "Check whether 'ACTUAL_OUTPUT' contains phrases indicating uncertainty or inability to answer (e.g., 'I don't know how to answer this question.', 'The answer is not in the context.'). If so, stop evaluation, skip all subsequent steps, and assign 0 score.",
                "Check whether 'ACTUAL_OUTPUT' correctly addresses the intent of the 'INPUT' question, regardless of its length compared to 'EXPECTED_OUTPUT'.",
                "If 'ACTUAL_OUTPUT' successfully answers the 'INPUT' question: Evaluate whether 'ACTUAL_OUTPUT' can replace 'EXPECTED_OUTPUT' as a valid and complete answer, ensuring that 'ACTUAL_OUTPUT' is accurate and covers the main points in 'EXPECTED_OUTPUT'.",
                "If 'EXPECTED_OUTPUT' is longer than 'ACTUAL_OUTPUT': Evaluate positively as long as 'ACTUAL_OUTPUT' sufficiently answers the 'INPUT' question, even if it is less detailed.",
                "If 'ACTUAL_OUTPUT' is longer than 'EXPECTED_OUTPUT': Verify that 'ACTUAL_OUTPUT' is coherent, stays on topic, and directly addresses the 'INPUT' question without including unnecessary or irrelevant information.",
                "In all cases, prioritize whether 'ACTUAL_OUTPUT' thoroughly resolves the 'INPUT' question over strict comparisons of length or level of detail with 'EXPECTED_OUTPUT'."
            ],
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.INPUT
            ],
            model=self.model[0],
            
        )
        # answer_relevancy = AnswerRelevancyMetric(
        #     threshold=0.5,
        #     model=self.model[1],
        #     include_reason=False,         
        # )
        
        # faithfulness = FaithfulnessMetric(
        #     threshold=0.5,
        #     model=self.model[2],
        #     include_reason=False,
        # )
        
        context_precision = ContextualPrecisionMetric(
            threshold=0.5,
            model=self.model[3],
            include_reason=False,
        )
        
        context_recall = ContextualRecallMetric(
            threshold=0.5,
            model=self.model[4],
            include_reason=False,
        )
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        # llm1 = ChatOpenAI(
        #     model="Meta-Llama-3_1-70B-Instruct",    
        #     openai_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzMxNDEwMDY2LCJpYXQiOjE3MzA4MDUyNjYsInN1YiI6IjcxNTY1N2M2LWE1OTgtNGM4Ny1hNjBhLTJiMjc1MzAwYjE4YiIsImVtYWlsIjoiZ29nb3J1bjIzNUBnbWFpbC5jb20iLCJwaG9uZSI6IiIsImFwcF9tZXRhZGF0YSI6eyJwcm92aWRlciI6ImdlbmVyaWMxIiwicHJvdmlkZXJzIjpbImdlbmVyaWMxIl19LCJ1c2VyX21ldGFkYXRhIjp7ImVtYWlsIjoiZ29nb3J1bjIzNUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaXNzIjoiaHR0cHM6Ly93d3cub3ZoLmNvbS9hdXRoL29hdXRoMi91c2VyIiwicGhvbmVfdmVyaWZpZWQiOmZhbHNlLCJzdWIiOiJuZzEyOTA0OC1vdmgifSwicm9sZSI6ImF1dGhlbnRpY2F0ZWQiLCJhYWwiOiJhYWwxIiwiYW1yIjpbeyJtZXRob2QiOiJvYXV0aCIsInRpbWVzdGFtcCI6MTcyODA1NDI3OH1dLCJzZXNzaW9uX2lkIjoiZGE2MWEzNTEtMzg1YS00ZGJhLTllZjktZDBjM2Y5NDBiMmI4In0.CyxDLpG7XVB17pLEUCsL5qBW_1w1apYlApuivf4Dd5k",
        #     openai_api_base="https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",
        # )
        answer_relevancy_scores = []
        faithfulness_scores = []
        context_precision_scores = []
        context_recall_scores = []
        correctness_scores = []
        dataset = self.load_dataset()
        # result = evaluate(test_cases=dataset,metrics=[answer_relevancy,faithfulness,correctness,context_precision,context_recall])
        # with open('./result_eval.pkl', 'wb') as f:
        #     pickle.dump(result,f)
            
        stop_processing = False
        for i in range(0, len(dataset.test_cases)):
<<<<<<< HEAD
            if stop_processing:
              break
            retries = 0
            while retries < 10:
                try:
                    print(i)
                    await asyncio.gather(
                        # correctness.a_measure(dataset.test_cases[i]),
                        # faithfulness.a_measure(dataset.test_cases[i]),
                        # answer_relevancy.a_measure(dataset.test_cases[i]),
                        context_precision.a_measure(dataset.test_cases[i]),
                        context_recall.a_measure(dataset.test_cases[i]),
                    )
                    # print(correctness.score)
                    print("Metrics finished!")
                    # time.sleep(3)
                    # correctness_scores.append(correctness.score)
                    # answer_relevancy_scores.append(answer_relevancy.score)
                    # faithfulness_scores.append(faithfulness.score)
                    context_precision_scores.append(context_precision.score)
                    context_recall_scores.append(context_recall.score)
                    with open('50_200_3.pkl', 'wb') as f:
                        # pickle.dump([correctness_scores], f)
                        pickle.dump([answer_relevancy_scores,correctness_scores,faithfulness_scores,context_precision_scores,context_recall_scores], f)
                    break
                except Exception as e:
                    if retries < 10:
                        print(f"Error encountered: {e}. Retrying in 10 seconds... (Attempt {retries}/{10})")
                        time.sleep(10)
                        retries += 1
                    else:
                        print(f"Error encountered: {e}. Max retries reached. Stopping the process.")
                        stop_processing = True
                        break
        # self.df["correctness"] = correctness_scores
        # self.df["answer_relevancy"] = answer_relevancy_scores
        # self.df["faithfulness"] = faithfulness_scores
        self.df["context_precision"] = context_precision_scores
        self.df["context_recall"] = context_recall_scores

    def save_evaluate_output(self):
        # self.df = pd.DataFrame.from_dict(self.output)
        # error_df = self.df[(self.df['correctness'] == 0.0) | (self.df['questions'] == "Tôi không biết trả lời câu hỏi này.")]
        # if os.path.exists("./output/error_score.csv"):
        #     old_error = pd.read_csv("./output/error_score.csv")
        #     error_df = pd.concat([old_error,error_df], ignore_index=True)
        # error_df.to_csv("./output/error_score.csv",index=False)
        # self.df = self.df[~self.df['questions'].isin(error_df['questions'])]
        # self.df["correctness_binary"] = (self.df['correctness'] > 0.5).astype(int)
        # self.df["answer_relevancy_binary"] = (self.df['answer_relevancy'] > 0.5).astype(int)
        # self.df["faithfulness_binary"] = (self.df['faithfulness'] > 0.5).astype(int)
        self.df["context_precision_binary"] = (self.df['context_precision'] > 0.5).astype(int)
        self.df["context_recall_binary"] = (self.df['context_recall'] > 0.5).astype(int)
        self.df.to_csv("./output/" + self.filename,index=False)
        
    def get_score(self):
=======
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
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd

        print("Trung bình của các chỉ số:")
        # print("Answer Relevancy:", self.df.answer_relevancy.mean())
        # print("Faithfulness:", self.df.faithfulness.mean())
        # print("Correctness:", self.df.correctness.mean())
        print("Context Precision:", self.df.context_precision.mean())
        print("Context Recall:", self.df.context_recall.mean())
        # print("Answer Relevancy binary:", self.df.answer_relevancy_binary.mean())
        # print("Faithfulness binary:", self.df.faithfulness_binary.mean())
        # print("Correctness binary:", self.df.correctness_binary.mean())
        print("Context Precision binary:", self.df.context_precision_binary.mean())
        print("Context Recall binary:", self.df.context_recall_binary.mean())

class QuestionVerify():
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
        self.df = None

    def load_csv(self):
        data = pd.read_csv(self.path_to_data)
        return data
    
    def load_dataset(self):
        dataset = EvaluationDataset()
        data = self.load_csv()
        datadict = {}
        datadict["Question"] = data['Question'].to_list()
        datadict["Answers"] = data['Answers'].to_list()
        datadict["Context"] = data['Context'].to_list()
        self.output = datadict
        for i in range(0, len(datadict["Question"])):
            testcase = LLMTestCase(
                input=datadict["Question"][i],
                actual_output=datadict["Answers"][i],
                # context=datadict["Context"][i]
            )
            dataset.add_test_case(testcase)
        return dataset
    
    async def eval(self):
        question_verify = GEval(
            name="Xác minh câu hỏi",
            evaluation_steps=[
                "The input MUST include the name of a specific procedure. If not, ignore the next steps, 0 score for this.",
                "Verify that the input does not contain 'ambiguous contextual references' phrases such as 'thủ tục này', 'trong văn bản', or 'trong bối cảnh này', 'thủ tục' alone is OK",
                "Determine if the input is specific enough to be answered accurately."
            ],
            evaluation_params=[LLMTestCaseParams.INPUT],
            model=self.model,
            verbose_mode=True
        )
        question_verify_scores = []
        dataset = self.load_dataset()
        stop_processing = False
        for i in range(0, len(dataset.test_cases)):
            if stop_processing:
              break
            retries = 0
            while retries < 10:
                try:
                    print(i)
                    await asyncio.gather(
                        question_verify.a_measure(dataset.test_cases[i]),
                    )
                    time.sleep(3)
                    print("Metrics finished!")
                    question_verify_scores.append(question_verify.score)
                    with open('50_200_3.pkl', 'wb') as f:
                        pickle.dump([question_verify_scores], f)
                    break
                except Exception as e:
                    if retries < 10:
                        print(f"Error encountered: {e}. Retrying in 10 seconds... (Attempt {retries}/{10})")
                        time.sleep(10)
                        retries += 1
                    else:
                        print(f"Error encountered: {e}. Max retries reached. Stopping the process.")
                        stop_processing = True
                        break
        
        self.output["question_verify"] = question_verify_scores

    def save_evaluate_output(self):
        self.df = pd.DataFrame.from_dict(self.output)
        self.df = self.df[self.df["question_verify"] != 0]
        self.df.to_csv("./data/" + self.filename,index=False)
        
    def get_score(self):
        average_question_verify = self.df.question_verify.mean()
        print("Trung bình của các chỉ số:")
        print("Question Verify:", average_question_verify)


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
    
<<<<<<< HEAD
class CustomModel(DeepEvalBaseLLM):
    def __init__(self,api_key,base_url,model_name):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
    def load_model(self):
        model = OpenAI(api_key=self.api_key,base_url=self.base_url)
        return model
=======
# Init
gemini_chat = ChatGoogleGenerativeAI(model='gemini-pro',google_api_key="AIzaSyD3NCZLaMXUpG1UvStJMN8eYB1QeleOg6Y",temperature=0.1)
model = CustomModel(gemini_chat)
path_to_evaluate_data = "./data/testset.json"
evaluator = Evaluator(path_to_data=path_to_evaluate_data, model=model)
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        resp = client.chat.completions.create(
            messages=[
                {
            "role": "system",
            "content": """
            You are a judge tasked with evaluating model responses. Your response must follow these rules without exception:
            1. Your response must be valid JSON only. Ensure that the JSON format strictly adheres to the structure provided in the user's prompt.
            2. Do not include any additional text, commentary, or explanation outside the JSON structure.
            """
        },
        {
            "role": "user",
            "content": prompt,
        },
            ],
            model=self.model_name,
            temperature=0,
            # top_p=.0000000000000000000001,
            # seed=0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# Init
# gemini_chat = ChatGoogleGenerativeAI(model='gemini-1.5-pro',google_api_key=os.getenv("GOOGLE_API_KEY"),safety_settings=safety_settings,temperature=0.1)
# model = CustomGeminiFlash()
# model = CustomModel(api_key="jDVi03OCpbFVWaAZAd8gpa9JOL8mjdqU",base_url="https://api.mistral.ai/v1",model_name="mistral-large-latest")
# model = CustomModel(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzMxNDEwMDY2LCJpYXQiOjE3MzA4MDUyNjYsInN1YiI6IjcxNTY1N2M2LWE1OTgtNGM4Ny1hNjBhLTJiMjc1MzAwYjE4YiIsImVtYWlsIjoiZ29nb3J1bjIzNUBnbWFpbC5jb20iLCJwaG9uZSI6IiIsImFwcF9tZXRhZGF0YSI6eyJwcm92aWRlciI6ImdlbmVyaWMxIiwicHJvdmlkZXJzIjpbImdlbmVyaWMxIl19LCJ1c2VyX21ldGFkYXRhIjp7ImVtYWlsIjoiZ29nb3J1bjIzNUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaXNzIjoiaHR0cHM6Ly93d3cub3ZoLmNvbS9hdXRoL29hdXRoMi91c2VyIiwicGhvbmVfdmVyaWZpZWQiOmZhbHNlLCJzdWIiOiJuZzEyOTA0OC1vdmgifSwicm9sZSI6ImF1dGhlbnRpY2F0ZWQiLCJhYWwiOiJhYWwxIiwiYW1yIjpbeyJtZXRob2QiOiJvYXV0aCIsInRpbWVzdGFtcCI6MTcyODA1NDI3OH1dLCJzZXNzaW9uX2lkIjoiZGE2MWEzNTEtMzg1YS00ZGJhLTllZjktZDBjM2Y5NDBiMmI4In0.CyxDLpG7XVB17pLEUCsL5qBW_1w1apYlApuivf4Dd5k",base_url="https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",model_name="Meta-Llama-3_1-70B-Instruct")
model1 = CustomModel(model_name="Meta-Llama-3_1-70B-Instruct",base_url="https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",api_key="eyJhbGciOiJFZERTQSJ9.eyJwcm9qZWN0IjoiYmQwNzJlY2VkYWNmNDQ1NjlmMzdkYmU4OGQwNjU4NTYiLCJhdWQiOiIzNzM4NjExNjY0MDQzMDM0IiwiZXhwIjoxNzM5MTU5NzcwLCJqdGkiOiJmYWY4ZjZiNS0xNjNkLTQ3YzctYTI1Ni05MGE0MGY5OThmNjgiLCJpc3MiOiJ0cmFpbmluZy5haS5jbG91ZC5vdmgubmV0Iiwic3ViIjoibGgxOTk0MDEtb3ZoIiwib3ZoVG9rZW4iOiJkTURWZ09qek1DN0hPT3JNc0pXN3V3bDFyMjQ4OEo3TURNR2tPT2Y0b1VfV0tqTjFLOFRkZlpjWC1QYjJNb1BxdGpyRlo4N0FJTVRHeTZWWEdYbGd5WElfSE5mYUc5MmpZLUpONDVXNWVHeVdnRklCWjdLSmtjTjZDWVZnS2RvZkthRVRnNnlubkVJd21ISHV1U0xld3M4TTViWENXMkhER0FUdDlCUWJnN0pqZDlNeWttY2tpbXV3bnhKSXFjQ2N1YVZtMlZqbERTajRoTEp3ejJhRVd6VVpJZ0FPSVZkNm85cU1VVk9yTE1FTXdzSkNEdnZXYXdGcVdONlBITjctei1OYVVJMk5icFVtY3pvaHNPTFA3RksyWDhzVWhkVnc1b3NWaHpvRjVXeDc3bVdBbl9XMzdzNmM1QlZyc2RDR3lSQXpSWEJYcDBfMmprTV85cERDbDB1NFMwZTJjdFhsTkE0cDNFbFNZMnRBa09ZWUZmejRrdXl3UzJFUFZ6a1VjNGN1T2NGMHdyMGRUb3pqMG5UMkhyNHdUSmllejVnU2Z4Mlp6b0UzUzlFdVZOMlpHOXRaNGYtcVIyWHEtdlcxb01UQTJXTHYyVVEyWmplelJoT05yaG1GM0QwWVliNTg2a2lmaWU2cjNXcUVWSXBDUWdVS0xfN0lNSG1ySzFaaURQLWtvVjZvWEtxTld3eFpsbUozYWpWdzV2TDdoSzdfc3JuY0FEb3BoRFVBYXh1bG5zOUEtZkktU0dGOUxqTHpKZ3B4UUdRaTYyeU9sdFZiSk9mUGhRbVJuc2JIRGJZMEFCNGdOYkVSZ25FU3ZsMVB0dnB2OUtad0ZsT1JlazhBVGM1TlpiSFI3UHFjaXoxSXV4bFk5MVJ5M0RtTXd1djJFV2V5TGgtZ18wQWVWS1UtY1BxZUp4aG92c2hDWVEwVFlvRUtSSklkMXFLd3pmQmpoU25EdGJJRTkxNWtHamFGcFR5RjJBUEJWRlc3WFp4V3Z5UHM5VFVkUExTbDlVM1hJeGtDczdKS2hubnpJX2ZzRTJkTHhqMmlwNUZzcFoxdHNnaTFBM0c4X0tiY1U4MnZnQ2Zjcjl3LS1fb2tET0IxTFFrS0RxLTBCME1XWF9pb19xWkpRS3hqS0FsY2JSSW9kQWNtSGVqQWgwdHY2X0w4TjB4ck9sOUJMYjF1anRGUFA1WkxlcU41eVM0bllaTzJoX3dHcjZFWTM4bnMtbVFSREpjOHAzUW1ST09obW1EVzYta2JvNUd3RVdpZFFfZmEyY0lkSlRUMW4zQzBSNGN6cndTc1BJOUJvYVBwMmxFbmU3eTU3LS14NXZWcSJ9.kRffxJCgM7T8pvJdf2xGfPMEV2dEUn7CpBU6i7p_moIHja2F2ukSCtyhXibAjUPlfJ382yxn0D83sBGeAQLKDQ")
#ndnh2003
model2 = CustomModel(model_name="Meta-Llama-3_1-70B-Instruct",base_url="https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",api_key="eyJhbGciOiJFZERTQSJ9.eyJwcm9qZWN0IjoiOTU0ZDdjMGNhNWI5NDAxMzgwZDhkOGEyZDg3YmJhYmUiLCJhdWQiOiIzNzM4NjExNjY0MDQzMDM0IiwiZXhwIjoxNzcwNjM1ODk0LCJqdGkiOiJmNjkxZGI4ZS05MmYwLTRiNTktYjFjNC00MzI0Nzg0MjBhNTciLCJpc3MiOiJ0cmFpbmluZy5haS5jbG91ZC5vdmgubmV0Iiwic3ViIjoibmcxMjkwNDgtb3ZoIiwib3ZoVG9rZW4iOiJGX1RMUnp0bG5MYmIyanpsZWVremhPVjNQcVA1UFJPaktQVDQzT1pyMG1xZTJUUTdHcE9HaXNPVnBnRFhIU2NUNm96VU95QkcxdlpOUzR0VWpoTVA5UFc0UlhrcVJzeVlrZ0F6Yzh1TE9vRlk5S0Q3dE1rYW1RN0taZXZsUk5nRXBRZjdoVURWeUZMaVlMcTJ6Z3RfWkQwZmVsUHNDOEw5c2JyYUNtSGQzbExKdnA5WWM3X0VpZ0tCeG9fNGhObWpsN3lWUWt2UTJwcHNJUk1BWUpuYmtkamFXN19CdldFU19sSE5xVy1OMTJZNnpXU1VVM0FwcDBWNmcxODRaRG9SS0JmR2ZvbGxWdjRscTIzb043WlA1Y0Zmd0xYUk5LTm44SGJqcGlPVGhpTEtGY05rNzc1UGlnbWdxb1h2ZHhkeC1iQXVfMG1Xa0U0VE1jWG1wZmsyOGlReUEwVWdxQ2RHaHJTOEt4bWlrbFJZQzVJMWljXy1qU2hfSnFZanROWFM5VVlucmlfbHQ4Qy1OVjI5R1RGOFNzRFYtVHY5MXh3aTNsb3p4Y2Vsei12MEtJTXdmVHJscU4ta0ZSd1ZhLVM5M1dXLWJub3hlZjNUMU5YQ0ktRVVIeGJjMk8xZk1TQ0RUUXRZTW5oM2JuUVpUZ3JlTmg0Q0Q2dU5lb2YwRlh5VjhzLUNqdkk5R3RFaTBZSUFnd2RTNzUxNGJyMmozQTlVcDRGTkdFcXVuV0lmUk4zbFRla082clY1UEhLRDczQVEtdkRqVF9haDdyX1BuTjZTd3FraUZhazdtSld1MmRHOXJ4VWxIRmFNalVyeE1qQm1rZG9MMEpDSVlaNkExTC0xOEVDMS1WNU9xeFZFNHY2WFkxREVHTXR1SmVvdmEySmJDVk5uUHZxNm5EZ1I3Mm9HMzkyT1hYaGxXVGlTcU9NX2VDcVQyNEY5V0k3RGpXM1E2VS10Sm9PeUV0WGJERFJLRzVHQUdMVTFlQS1FaDVzSUttem1iRlhkcHNMaWUxaEswblA2VFA2bDZIa3dDME42MklJaVRCeGlNbDItdmtNeHJvbDFRVHp4TzhIbG9raWlkV3RSTjZTckttRnE5ZU8yLTU2RW5uUHVxSnBBNG5FVi1DWUdMLXFCRS1oQjUyNXJ5SU1vSmZCMjZCM3E2MTZxNU42VHZzSzJ2a3p1Tm9kMjZzblRTNzRNemsxT2kzbFcya3lVNzRGaVBRVDEzZVJCdEo2ZU41dHMwSnN5MktseV95WnduSDMwSTFrbW9OM1hPV1ZoaTItbjB4YnhsclV3T3NTbHZuQWpyQ0JOVkxtX1E5cUR3UmNEM2ZKYSJ9.IWuqfMSqSoEOTTEHOxeyXKE7ZyMB_LXbVqh8h-PNJQwQhyeaIPa64Cp9GDaXhvQhZHpfBccLqWWbli--A4HKCg")
model3 = CustomModel(model_name="Meta-Llama-3_1-70B-Instruct",base_url="https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",api_key="eyJhbGciOiJFZERTQSJ9.eyJwcm9qZWN0IjoiNDUyM2NiZDgzOWZiNDAyNGExZDJmOTU4NmNiM2Q2M2MiLCJhdWQiOiIzNzM4NjExNjY0MDQzMDM0IiwiZXhwIjoxNzcwNjM2Mzc1LCJqdGkiOiI1ZTIyYWRhZi03OTRlLTRmMTYtOGVlMS05Y2RlNzIwNTgwMjAiLCJpc3MiOiJ0cmFpbmluZy5haS5jbG91ZC5vdmgubmV0Iiwic3ViIjoiaHc0NTAxMS1vdmgiLCJvdmhUb2tlbiI6InJOOXk3U01uRU9mckNkeEdXVWVFU1pxRDB1ZjE4alB2MjlXNjlQNHFKZnBzUjhWT2oyM3A2bXYtUFZ5d01XaENUWjdTRkdXcVREcnFRTXFwdzY4ZnRWTXdoSFZPeGJsb3hVX2M0NElEYXJpTFBGTmFIdjQ4c0ZtcUdBQnNTOUMzRXY3MndaTWRjX2dzQ2ZlaXJHTVJkNjF1QmdFMEwtcGZOWDFXWW5EZGJwYW9GWWFMNWlIYWdOc3dtV2FKUjAzeE1BWFdWLWxydnJKSXRFRlBjcE9qSGhRcWExTVZpZTBXMXNES3FxcDA2QXlyVjR5U0VSdEJWLWdKVmhRVU9zR1FBTGxJcUNnWjNyVFlBSW5wQ2g1Q3dtVDNMWmYwQjNjdzNRR2U2REt4N05EcDBPLVBiSkFiMHI3WHF2LW9BMW5KU05ZYlhMNkdsZ1ItQS1QVTh3WFE5MmdTUlJidzd0MGY3ZWpQb1Q0MHR4NTI3VVU5ZERLVUplSkxKQ1FOVXJqWXhUZERGRkozZWctU3gwRnNDZ0R4RXB4Ulp5U05aUmhNMFB0eV85dnZidjBvamZKZ2cyZE1OQUtXZVh2UEhROUNsckJJbnY0Z2lSdWZRVExOOUpqLTlBWnNueVVwVlY4MTdIb3RKelBLSzlRbHRLaVFrRUJFYVdkUUh1czdSTDlfR25ZZVBGcG13VEIwSm0zWDE1UnlKM00yOE56ZFItMUpYc2ZiY2tQWTNvMUdPNVBaVy14anJkS2FvMG1JbzV3V0FVNzRpbVhSblZEVHU3T25wN2ZwdmFLNHpfbjlIcG1oVjIwZVZVU2Y3U0kxLWZfd0RWbzVCRkwxQW1SY0VBLVRlbkNSeVhWRlJxV1hybnRIdDRNOUQ4X3gzanphd3p6cmM3bi16Yl9Wc2FmNmFTc0xnYUFFMGtHSENHMGFOYi1sT0JGUk1uTjZoRUVLMnpOaVM5NlF2ZkRibGVBaVBaSnVmX1VyOHJhZ2Vtbzk4akt3UFNiYy1oa3hLdFZ4ejlQRUNpLVJaM0hTelkzbGVMc2NOZDJueEhYYXh0RGhTU1o3c2JsUjRzcV9Rdm85UjZvcHN1MGY2bVRtc1R4RlNqd1UtN2pSV05qWGRDVlJEZmRBbDI3VjhqVVpwTHBwd0tOVk9GblRXRzBzUEJXQVBBMG1Ddm5JbGFDdkNzVTdYYThtWlViSWNQRE5fOG10UGJScUQxcm40N3ZVaktOeVZpS29XZUZZc2dHaGVWeW9sejNqSkdWZmdDUmVCUTU0RUw4TFQ3a08zQ010aHcwNzMyUFR5ZFZHejBObnhBMC1na0U0VzdCN3hVc200R0FXIn0.xCjsUt0r_75yFirgvVUsAq2W36Lje-38K4607hOryCpk98TtAz_TX6FjSjOm4Pgpi8shu8uoP5hvVVYymD1HBA")
model4 = CustomModel(model_name="mistral-large-latest",base_url="https://api.mistral.ai/v1",api_key="JhiTAjoftMAgoiDsjVLagRf8ZIfapTsR")
model5 = CustomModel(model_name="mistral-large-latest",base_url="https://api.mistral.ai/v1",api_key="gg4mKZEWdXLZdMQEEHvq97vbQEOApqcL")
#gogo,ndnh,211,whyandwhile4,???
model = [model1,model2,model3,model4,model5]
# evaluator = Evaluator(path_to_data="./output/error_score.csv", model=model,filename="error_result.csv")
# evaluator = QuestionVerify(path_to_data="./data/600_700.csv", model=model[2],filename="600_700_Qverified.csv")
evaluator = Evaluator(path_to_data="./data/CHUNK_HYBRID.csv", model=model,filename="CHUNK_HYBRID_result.csv")
asyncio.run(evaluator.eval())

# Get Dataframe
<<<<<<< HEAD
result = evaluator.save_evaluate_output()
evaluator.get_score()
=======
result = evaluator.get_evaluate_output()
result.to_csv("./data/eval_class_19_4.csv")
evaluator.get_relevance_score()
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd

