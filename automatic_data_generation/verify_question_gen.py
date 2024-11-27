from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from llama_index.llms.gemini import Gemini
import csv

# import os
# import sys
# original_directory = os.getcwd()

# try:
#     os.chdir('E:/thesis/RAG4PublicSector')  # Adjust this path
#     sys.path.append(os.getcwd())
#     from rag_pipeline.rag_init import verified_prompt
# finally:
#     os.chdir(original_directory)

# from rag_pipeline.rag_init import verified_prompt
from typing import List
import pandas as pd
import asyncio
verified_prompt = """Bạn sẽ được cung cấp một đoạn văn bản và 5 câu hỏi. Hãy xác nhận liệu các câu hỏi có liên quan đến đoạn văn bản này không. Nếu câu hỏi có liên quan và hợp lệ, hãy cung cấp câu trả lời dựa trên thông tin trong context, tuân theo các <answer_requirements> được cung cấp. Nếu câu hỏi không liên quan, phải trả lời "Không".
Bạn được cung cấp một <example> để có thể hiểu hơn.
Hướng dẫn:

Các câu hỏi sai quy định: (Hãy trả lời "Không" nếu gặp những câu hỏi sai qui định này)
1. Câu hỏi mơ hồ, không có tên thủ tục.
2. Nếu câu hỏi không chứa tên thủ tục mà thay vào đó là chữ "này".

<answer_requirements>
1. Sau mỗi câu trả lời đều phải có thêm kí tự '[END]'.
2. Câu trả lời phải có chủ ngữ và phải trả lời đầy đủ.
3. Trực tiếp đưa ra câu trả lời, không dược đưa bất kỳ câu hỏi nào vào câu trả lời.
4. Nếu câu hỏi không liên quan, hãy trả lời "Không".
5. Nếu không biết trả lời, hãy trả lời "Không".
6. Số lượng câu trả lời luôn luôn phải là 5.
7. Nếu ngữ cảnh cung cấp tài liệu liên quan có dạng "<a href="url">link text</a>", hãy thay nó sang dạng markdown [link text](url) và in ra đầy đủ.
<answer_requirements>

<example>
Đoạn văn bản:
Chi tiết thủ tục hành chính:
Mã thủ tục:
1.000105
Số quyết định:
1560/QĐ-LĐTBXH
Tên thủ tục:
Báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài
Cấp thực hiện:
Cấp Tỉnh
Loại thủ tục:
TTHC được luật giao quy định chi tiết
Lĩnh vực:
Việc làm
...

5 câu hỏi:
Thủ tục này thuộc loại thủ tục gì?
Mã thủ tục báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài là gì?
Thủ tục trên được thực hiện ở đâu?
Văn bản trên có mã thủ tục là gì?
Lĩnh vực thực hiện của báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài là gì?

5 câu trả lời:
Không [END]
Mã thủ tục của thủ tục báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài là 1.000105. [END]
Không [END]
Không [END]
Lĩnh vực thực hiện của báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài là Việc làm. [END]

Bạn sẽ được cung cấp <explaination> cho việc đánh giá này để có thể hiểu hơn lí do đánh giá như vậy.
<example/>

<explaination>
Giải thích lí do:
Câu hỏi "Thủ tục này thuộc loại thủ tục gì?" không hợp lệ vì không có tên thủ tục.
Câu hỏi "Mã thủ tục báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài là gì?" có liên quan nên trả lời.
Câu hỏi "Thủ tục trên được thực hiện ở đâu?" không hợp lệ vì không có tên thủ tục.
Câu hỏi "Văn bản trên có mã thủ tục là gì?" không hợp lệ vì không có mã thủ tục.
Câu hỏi "Lĩnh vực thực hiện của báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài là gì?" có liên quan nên trả lời.
<explaination/>

Đoạn văn bản:
{context}
5 câu hỏi:
{questions}
5 câu trả lời:
"""

class CrossCheckingLLMs:
    def __init__(
        self,
        input_dataframe: pd.DataFrame,
        qa_dataframe: pd.DataFrame,
        generator_LLM: str,
        checker_LLM: List[str], 
        output_path: str,
        api_key_dictionary
    ):
        if input_dataframe is None:
            raise ValueError("input_dataframe is required")
        if qa_dataframe is None:
            raise ValueError("qa_dataframe is required")
        if checker_LLM is None:
            raise ValueError("checker_LLM is required")

        self.api_key_dictionary = api_key_dictionary
        self.input_dataframe = input_dataframe
        self.qa_dataframe = qa_dataframe
        self.generator_LLM = generator_LLM
        self.checker_LLM = checker_LLM
        self.init_prompt = verified_prompt  
        self.output_path = output_path
        self.result_dataframe = pd.DataFrame()
        

    def __str__(self):
        return f"CrossCheckingLLMs(input_dataframe={self.input_dataframe}, generator_LLM={self.generator_LLM}, checker_LLM={self.checker_LLM}), qa_dataframe={self.qa_dataframe}"

    def get_model_OpenAI(self, llm_name):
        if llm_name == "pixtral-12b-2409" or llm_name == "open-mixtral-8x7b" or llm_name=="mistral-large-latest" or llm_name=='mistral-small-latest':
            base_url="https://api.mistral.ai/v1"
        if llm_name == 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo' or llm_name == 'Qwen/Qwen2-72B-Instruct':
            base_url="https://api.together.ai/v1"
        if llm_name == 'gemma2-9b-it':
            base_url="https://api.groq.com/openai/v1"
        if llm_name == "qwen/qwen-2.5-72b-instruct" or llm_name == "qwen/qwen-2-vl-72b-instruct" or llm_name== "open-mistral-nemo-2407" or llm_name == "google/gemma-2-9b-it:free" or llm_name == "google/gemini-flash-8b-1.5-exp":
            base_url="https://openrouter.ai/api/v1"
            
        return OpenAI(api_key=self.api_key_dictionary[llm_name],base_url=base_url)
        
    async def check_relevent(self, context, question_list, llm):
        if llm =="gemini-1.5-flash-8b-exp-0924":
            llm_interface = Gemini(model='models/'+llm,api_key=self.api_key_dictionary[llm],temperature=0.1)
        else:
            llm_interface = self.get_model_OpenAI(llm)
        time_delay = 2

        while True:
            try:
                if llm =="gemini-1.5-flash-8b-exp-0924":
                    result = await llm_interface.acomplete(self.init_prompt.replace("{context}", context).replace("{questions}", "/n".join(question_list)))
                    result = result.text.split('[END]')[:-1]
                else:
                    result = llm_interface.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": self.init_prompt.replace("{context}", context).replace("{questions}", "/n".join(question_list)),
                            }
                        ],
                        #llm day la cai string nha may
                        model=llm,
                    )
                    result = result.choices[0].message.content.split('[END]')[:-1]
                    result = [s.strip() for s in result]    
                return result
            except Exception as e:
                tqdm.write(f"Error in check_relevent: {e}")
                tqdm.write(f"Retry after {time_delay}s")
                await asyncio.sleep(time_delay)

    async def cross_check(self):
        total_batches = len(range(0, len(self.input_dataframe)))
        pbar = tqdm(total=total_batches, ncols=100, desc="Processing")
        # tqdm.write("/n")  # Using tqdm.write to avoid conflict with the progress bar

        failed_cases = pd.DataFrame()  # DataFrame to store failed cases
        index_outside = 0
        for index, row in self.input_dataframe.iterrows():
            qa_question_list = self.qa_dataframe['query'][index_outside * 5:(index_outside + 1) * 5].to_list()
            qa_answer_list = self.qa_dataframe['generation_gt'][index_outside * 5:(index_outside + 1) * 5].to_list()

            retry_count = 0  # Initialize the retry counter
            max_retries = 5  # Set the maximum number of retries

            index_outside += 1
            while retry_count <= max_retries:
                try:
                    tasks = []
                    for llm in self.checker_LLM:
                        task = self.check_relevent(row['contents'], qa_question_list, llm)
                        tasks.append(task)

                    results = await asyncio.gather(*tasks)

                    if any(len(result) < len(qa_question_list) for result in results):
                        raise IndexError("One of the results lists is shorter than expected.")
                    elif len(qa_question_list) == 0:
                        tqdm.write(f"No question found: {qa_question_list}")
                        raise IndexError("There is no question!")

                    index_l = 0
                    while index_l < len(results[0]):
                        flag = 0
                        for slice_list in results:
                            if slice_list[index_l] == 'Không' or slice_list[index_l] == 'Không.':
                                flag = 1
                                break

                        if flag == 0:
                            new_row = {
                                "Context": row['contents'],
                                "Question": qa_question_list[index_l],
                                "Answers": [qa_answer_list[index_l][0]] + [slice_list[index_l] for slice_list in results]
                            }


                            new_row_df = pd.DataFrame([new_row])
                            self.result_dataframe = pd.concat([self.result_dataframe, new_row_df], ignore_index=True)
                            self.result_dataframe.to_csv(self.output_path, index=False)

                        index_l += 1

                    pbar.update(1)
                    break  

                except IndexError as e:
                    retry_count += 1
                    tqdm.write(f"Retrying due to error: {e}, attempt {retry_count} of {max_retries}")
                except Exception as e:
                    retry_count += 1
                    tqdm.write(f"Retrying due to an unexpected error: {e}")
                

            if retry_count > max_retries:
                tqdm.write(f"Skipping row {index} after {max_retries} failed attempts")
                failed_case = {
                    "Context": row['contents'],
                    "Question": qa_question_list,
                    "Answers": qa_answer_list
                }
                failed_case_df = pd.DataFrame([failed_case])
                failed_cases = pd.concat([failed_cases, failed_case_df], ignore_index=True)
                pbar.update(1)

        # Store failed cases in the separate output path
        failed_cases.to_csv(self.output_path + "_failed_cases.csv", index=False)
        pbar.close()

        tqdm.write(f"Verified question dataframe is saved to: {self.output_path}")
        tqdm.write(f"Failed cases of verified question dataframe are saved to: {self.output_path}_failed_cases.csv")

        return self.result_dataframe



    def get_result(self):
        return self.result_dataframe


# corpus_df = pd.read_parquet('data/corpus.parquet')
# check_df = pd.read_csv('data/data_gen_from_pipeline/pipeline_index_0_5/mistral-large-latest.parquet')
# checker = CrossCheckingLLMs(corpus_df[0:100], check_df, llm_mistral, [llm_qwen, llm_llama3])
# await checker.cross_check()
# checker.cross_check()