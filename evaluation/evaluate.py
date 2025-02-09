from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_genai import (
    HarmBlockThreshold,
    HarmCategory
)
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
# from deepeval.test_case import LLMTestCase
import time
import pandas as pd
import asyncio
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
        self.df = None
        self.load_csv()

    def load_csv(self):
        self.df = pd.read_csv(self.path_to_data)
    
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
        #     model=self.model[3],
        # )

        # context_recall = GEval(
        #     name="Context Recall",
        #     evaluation_steps=[
        #         "Ensure that 'RETRIEVAL_CONTEXT' contains all the critical information required to generate the 'EXPECTED_OUTPUT'.",
        #         "Verify that the majority of relevant nodes (e.g., key facts, entities, or phrases) in 'RETRIEVAL_CONTEXT', which are essential for generating the 'EXPECTED_OUTPUT', are present, accurate, and complete."
        #     ],
        #     evaluation_params=[LLMTestCaseParams.RETRIEVAL_CONTEXT, LLMTestCaseParams.EXPECTED_OUTPUT],
        #     model=self.model[4],
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
        
        # context_precision = ContextualPrecisionMetric(
        #     threshold=0.5,
        #     model=self.model[3],
        #     include_reason=False,
        # )
        
        # context_recall = ContextualRecallMetric(
        #     threshold=0.5,
        #     model=self.model[4],
        #     include_reason=False,
        # )
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
            if stop_processing:
              break
            retries = 0
            while retries < 10:
                try:
                    print(i)
                    await asyncio.gather(
                        correctness.a_measure(dataset.test_cases[i]),
                        # faithfulness.a_measure(dataset.test_cases[i]),
                        # answer_relevancy.a_measure(dataset.test_cases[i]),
                        # context_precision.a_measure(dataset.test_cases[i]),
                        # context_recall.a_measure(dataset.test_cases[i]),
                    )
                    # print(correctness.score)
                    print("Metrics finished!")
                    time.sleep(3)
                    correctness_scores.append(correctness.score)
                    # answer_relevancy_scores.append(answer_relevancy.score)
                    # faithfulness_scores.append(faithfulness.score)
                    # context_precision_scores.append(context_precision.score)
                    # context_recall_scores.append(context_recall.score)
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
        self.df["correctness"] = correctness_scores
        # self.df["answer_relevancy"] = answer_relevancy_scores
        # self.df["faithfulness"] = faithfulness_scores
        # self.df["context_precision"] = context_precision_scores
        # self.df["context_recall"] = context_recall_scores

    def save_evaluate_output(self):
        # self.df = pd.DataFrame.from_dict(self.output)
        # error_df = self.df[(self.df['correctness'] == 0.0) | (self.df['questions'] == "Tôi không biết trả lời câu hỏi này.")]
        # if os.path.exists("./output/error_score.csv"):
        #     old_error = pd.read_csv("./output/error_score.csv")
        #     error_df = pd.concat([old_error,error_df], ignore_index=True)
        # error_df.to_csv("./output/error_score.csv",index=False)
        # self.df = self.df[~self.df['questions'].isin(error_df['questions'])]
        self.df["correctness_binary"] = (self.df['correctness'] > 0.5).astype(int)
        # self.df["answer_relevancy_binary"] = (self.df['answer_relevancy'] > 0.5).astype(int)
        # self.df["faithfulness_binary"] = (self.df['faithfulness'] > 0.5).astype(int)
        # self.df["context_precision_binary"] = (self.df['context_precision'] > 0.5).astype(int)
        # self.df["context_recall_binary"] = (self.df['context_recall'] > 0.5).astype(int)
        self.df.to_csv("./output/" + self.filename,index=False)
        
    def get_score(self):

        print("Trung bình của các chỉ số:")
        # print("Answer Relevancy:", self.df.answer_relevancy.mean())
        # print("Faithfulness:", self.df.faithfulness.mean())
        print("Correctness:", self.df.correctness.mean())
        # print("Context Precision:", self.df.context_precision.mean())
        # print("Context Recall:", self.df.context_recall.mean())
        # print("Answer Relevancy binary:", self.df.answer_relevancy_binary.mean())
        # print("Faithfulness binary:", self.df.faithfulness_binary.mean())
        print("Correctness binary:", self.df.correctness_binary.mean())
        # print("Context Precision binary:", self.df.context_precision_binary.mean())
        # print("Context Recall binary:", self.df.context_recall_binary.mean())

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
            # response_format={"type": "json_object"},
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
model1 = CustomModel(model_name="Meta-Llama-3.1-70B-Instruct",base_url="https://api.sambanova.ai/v1",api_key="a2de19ae-71e3-4147-a5e3-fa4eec164b8c")
#ndnh2003
model2 = CustomModel(model_name="Meta-Llama-3.1-70B-Instruct",base_url="https://api.sambanova.ai/v1",api_key="a2de19ae-71e3-4147-a5e3-fa4eec164b8c")
model3 = CustomModel(model_name="Meta-Llama-3.1-70B-Instruct",base_url="https://api.sambanova.ai/v1",api_key="a2de19ae-71e3-4147-a5e3-fa4eec164b8c")
model4 = CustomModel(model_name="Meta-Llama-3.1-70B-Instruct",base_url="https://api.sambanova.ai/v1",api_key="a2de19ae-71e3-4147-a5e3-fa4eec164b8c")
model5 = CustomModel(model_name="Meta-Llama-3.1-70B-Instruct",base_url="https://api.sambanova.ai/v1",api_key="a2de19ae-71e3-4147-a5e3-fa4eec164b8c")
#gogo,ndnh,211,whyandwhile4,???
model = [model1,model2,model3,model4,model5]
# evaluator = Evaluator(path_to_data="./output/error_score.csv", model=model,filename="error_result.csv")
# evaluator = QuestionVerify(path_to_data="./data/600_700.csv", model=model[2],filename="600_700_Qverified.csv")
evaluator = Evaluator(path_to_data="./data/0_600_batchRAGver3.csv", model=model,filename="0_600_batchRAGver3_result.csv")
asyncio.run(evaluator.eval())

# Get Dataframe
result = evaluator.save_evaluate_output()
evaluator.get_score()

