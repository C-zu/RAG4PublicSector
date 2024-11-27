import pandas as pd
import ast
from llama_index.llms.gemini import Gemini
import asyncio
from tqdm import tqdm
from openai import OpenAI
import re


sim_prompt = """
Bạn là một trợ lý giúp đánh giá độ tương đồng về mặt ngữ nghĩa của {num_answer} câu trả lời từ một câu hỏi. Hãy đánh giá từng cặp.
Câu trả lời chỉ gồm 3 chữ số, không giải thích bất cứ thứ gì.
Bạn được cung cấp các <instruction> để hướng dẫn bạn cách đánh giá.

<instruction>
- Trả lời theo mẫu gợi ý bên dưới.
- Kết quả phải là của đồng thời 3 câu trả lời, không tách riêng ra thành từng cặp. Chỉ có 1 điểm số duy nhất với 3 con số trong mọi trường hợp.
- Nếu thấy trong câu trả lời nào có lặp lại câu hỏi, ngay lập tức trả về 0 0 0.
- Dựa trên các metric hiện đại nhất, tìm điểm số độ tương đồng về ngữ nghĩa giữa hai câu trả lời.
- Đầu vào là {num_answer} câu trả lời, đầu ra là điểm số đánh giá từ 0 hoặc 1 của từng cặp lần lượt. Ví dụ 3 câu trả lời thì điểm của 1 với 2, 1 với 3, 2 với 3.
- Nếu 1 trong hai câu trả lời đang so sánh được tóm tắt hơn nhưng vẫn tương đồng về mặt ngữ nghĩa thì hãy cho điểm số 1.
- Chỉ trả lời điểm số ví dụ 3 câu trả lời thì đầu ra là "1 1 1" và không trả lời gì hơn.
<instruction/>

Câu hỏi:
{question}
Các câu trả lời:
{answers}
Điểm số:
"""
gemini_api_key = "AIzaSyBJCmUGCqboT3AWVHp-7qUGejIrmfdSfCw" #nghiadeptrai1804

class SimilarityCheck:
    def __init__(
        self,
        input_dataframe: pd.DataFrame,
        llm: str,
        output_path: str = 'output.csv',
        # prompt: str = 'default prompt',
        batch_size: int = None,
        api_key_dictionary: dict = None,
    ):
        if input_dataframe is None:
            raise ValueError("input_dataframe is required")
        
        if llm is None:
            self.llm_name = "gemini-1.5-flash"
        else:
            self.llm_name = llm
        # if prompt is None:
        #     raise ValueError("prompt is required")

        self.api_key_dictionary = api_key_dictionary
        self.input_dataframe = input_dataframe

        if api_key_dictionary:
            if self.llm_name == "pixtral-12b-2409" or self.llm_name =='mistral-large-latest' or self.llm_name == 'mistral-small-latest':
                self.llm = OpenAI(api_key=self.api_key_dictionary[llm],base_url="https://api.mistral.ai/v1")
        
            if self.llm_name == 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo' or self.llm_name == 'Qwen/Qwen2-72B-Instruct':
                self.llm = OpenAI(api_key=self.api_key_dictionary[llm],base_url="https://api.together.ai/v1")

            if self.llm_name == 'gemma2-9b-it':
                self.llm = OpenAI(api_key=self.api_key_dictionary[llm],base_url="https://api.groq.com/openai/v1")

            if self.llm_name == 'qwen/qwen-2.5-72b-instruct' or self.llm_name == 'qwen/qwen-2-vl-72b-instruct' or self.llm_name == "open-mistral-nemo-2407" or self.llm_name == "google/gemma-2-9b-it:free" or self.llm_name == "google/gemini-flash-8b-1.5-exp":
                self.llm = OpenAI(api_key=self.api_key_dictionary[llm],base_url="https://openrouter.ai/api/v1")
            
            # Using different API key instead
            if self.llm_name == 'gemini-1.5-flash-8b-exp-0924':
                # self.llm = Gemini(model='models/'+self.llm_name,api_key=api_key_dictionary[self.llm_name],temperature=0.1)
                self.llm = Gemini(model='models/'+self.llm_name,api_key=gemini_api_key,temperature=0.1)
            
            if self.llm_name == 'gemini-1.5-flash':
                # self.llm = Gemini(model='models/'+self.llm_name,api_key=api_key_dictionary[self.llm_name],temperature=0.1)
                self.llm = Gemini(model='models/'+self.llm_name,api_key=gemini_api_key,temperature=0.1)
        else:
            self.llm = Gemini(model='models/'+self.llm_name,api_key=gemini_api_key,temperature=0)
    
        self.result_dataframe = pd.DataFrame()
        self.output_path = output_path
        self.prompt = sim_prompt
        self.batch_size = batch_size

    def __str__(self):
        return f"SimilarityCheck(input_dataframe={self.input_dataframe}, model_name={self.model_name})"

    async def check_similarity(self):
        total_batches = len(self.input_dataframe/self.batch_size)
        pbar = tqdm(total=total_batches, ncols=100, desc="Processing", position=0)

        # Initialize the failed_dataframe and failed_dataframe to store skipped and failed rows
        failed_dataframe = pd.DataFrame()

        for index, row in self.input_dataframe.iterrows():
            answers = ast.literal_eval(row['Answers'])
            question = row['Question']

            if len(answers) < 2:
                pbar.write(f"Skipping row {index} as it does not contain enough answers (minimum 2 required).")
                
                # Append the skipped row to failed_dataframe
                failed_dataframe = pd.concat([failed_dataframe, pd.DataFrame([row])], ignore_index=True)
                failed_dataframe.to_csv(self.output_path.replace('.csv', '_failed_rows.csv'), index=False)

                pbar.update(1)
                continue

            # Calculate average similarity for rows that are not skipped
            average_similarity = await self.calculate_similarity(answers, question)

            if average_similarity == -1:
                # If calculation failed, log the row in the failed dataframe
                failed_dataframe = pd.concat([failed_dataframe, pd.DataFrame([row])], ignore_index=True)
                failed_dataframe.to_csv(self.output_path.replace('.csv', '_failed_rows.csv'), index=False)
                continue

            pbar.set_postfix({"Avg Similarity": average_similarity})
            pbar.update(1)

            if average_similarity == 1:
                new_row = row.copy()
                new_row['avg_similarity'] = average_similarity
                self.result_dataframe = pd.concat([self.result_dataframe, pd.DataFrame([new_row])], ignore_index=True)
                self.result_dataframe.to_csv(self.output_path, index=False)

        pbar.close()
        failed_path = self.output_path.replace('.csv', '_failed_rows.csv')
        print(f"Failed verified answers dataframe is saved to: {failed_path}/n")
        print(f"Verified answers dataframe is saved to: {self.output_path}/n")
        
        return self.result_dataframe
    
    async def calculate_similarity(self, answers, question):
        retries = 4
        delay = 60
        attempt = 0
        while attempt < retries:
            try:
                answer_replacements = "/n".join([f"Câu {i+1}: {answers[i]}" for i in range(len(answers))]) + "/n"

                if self.llm_name == 'gemini-1.5-flash-8b-exp-0924' or self.llm_name == 'gemini-1.5-flash':
                    output = await self.llm.acomplete(self.prompt.replace("{num_answer}", str(len(answers))).replace("{question}", question).replace("{answers}", answer_replacements))
                    scores = list(map(float, output.text.split()))
                else:
                    output = self.llm.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": self.prompt.replace("{num_answer}", str(len(answers))).replace("{question}", question).replace("{answers}", answer_replacements),
                            }
                        ],
                        model=self.llm_name,
                        temperature=0   
                    )
                
                # print(scores)
                if len(scores) == 0:
                    average_similarity = 0
                else:
                    average_similarity = sum(scores) / len(scores)
                # print(average_similarity)
                return average_similarity
            
            except Exception as e:
                if "429" in str(e) or getattr(e, 'code', None) == 429:
                    # tqdm.write("Error 429 detected. Retrying without counting this attempt.")
                    await asyncio.sleep(delay)
                    continue
                else:
                    tqdm.write(f"Attempt {attempt + 1}/{retries} failed with error: {e}")
                
                if attempt < retries - 1:
                    tqdm(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    attempt += 1 
                else:
                    tqdm.write("Max retries reached. Logging failed case.")
                    return -1

    async def batch_process(self, batch, failed_dataframe):
        tasks = []
        for row in batch:
            answers = row['Answers']
            question = row['Question']

            if len(answers) < 2:
                # If there are not enough answers, skip the row and log it in failed_dataframe
                failed_dataframe = pd.concat([failed_dataframe, pd.DataFrame([row])], ignore_index=True)
                continue

            # Create task for calculating similarity
            tasks.append(self.calculate_similarity(answers, question))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results, failed_dataframe

    async def pipeline_check_similarity(self):
        batch_size = self.batch_size
        total_batches = (len(self.input_dataframe) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, ncols=100, desc="Processing")

        failed_dataframe = pd.DataFrame()
        batch = []

        for index, row in self.input_dataframe.iterrows():
            batch.append(row)

            if len(batch) == batch_size or index == total_batches - 1:  # Process in batches of 5 or at the last batch
                results, failed_dataframe = await self.batch_process(batch, failed_dataframe)
                
                for row, result in zip(batch, results):
                    if isinstance(result, Exception) or result == -1:
                        failed_dataframe = pd.concat([failed_dataframe, pd.DataFrame([row])], ignore_index=True)
                    else:
                        average_similarity = result
                        pbar.set_postfix({"Avg Similarity": average_similarity})
                        
                        if average_similarity == 1:
                            new_row = row.copy()
                            new_row['avg_similarity'] = average_similarity
                            self.result_dataframe = pd.concat([self.result_dataframe, pd.DataFrame([new_row])], ignore_index=True)
                            await asyncio.to_thread(self.result_dataframe.to_csv, self.output_path, index=False)

                batch = []

                pbar.update(1)  

        pbar.close()

        # Save the failed rows
        failed_path = self.output_path.replace('.csv', '_failed_rows.csv')
        await asyncio.to_thread(failed_dataframe.to_csv, failed_path, index=False)

        print(f"Failed verified answers dataframe is saved to: {failed_path}\n")
        print(f"Verified answers dataframe is saved to: {self.output_path}\n")

        return self.result_dataframe
    
# together_api_key = "b2364e8538f36185c3c7551f17ca2f730e3d0b495b5850faa4888a7043e0fc11" #trongnghiakazuhatran@gmail.com
# gemini_api_key = "AIzaSyAokJTNFmApBzp6tHEaX9P_cbvkkxkj6r8"
# mistral_api_key = "jDVi03OCpbFVWaAZAd8gpa9JOL8mjdqU"
# groq_api_key = "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"
# open_router_api_key = "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"

# api_key_dictionary = {
#     'pixtral-12b-2409': mistral_api_key,
#     'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo': together_api_key,
#     'Qwen/Qwen2-72B-Instruct': together_api_key,
#     'gemini': gemini_api_key,
#     'gemma2-9b-it': groq_api_key,
#     'qwen/qwen-2.5-72b-instruct': open_router_api_key,
#     'qwen/qwen-2-vl-72b-instruct': open_router_api_key,
#     'open-mistral-nemo-2407': open_router_api_key,
#     'google/gemma-2-9b-it:free':open_router_api_key,
#     'google/gemini-flash-8b-1.5-exp': open_router_api_key,
#     'gemini-1.5-flash-8b-exp-0924': gemini_api_key,
#     'mistral-large-latest':mistral_api_key,
#     'mistral-small-latest': mistral_api_key
# }

# df = pd.read_csv("E:/thesis/RAG4PublicSector/data/data_gen_from_pipeline/pipeline_index_400_500/mistral-large-latest_verified_question_dataframe.csv")
# sim_checker = SimilarityCheck(df, llm = 'gemini-1.5-flash-8b-exp-0924', output_path='E:/thesis/RAG4PublicSector/data/mistral-large-latest_verified_answer_dataframe.csv', batch_size=8, api_key_dictionary=api_key_dictionary)
# asyncio.run(sim_checker.check_similarity())