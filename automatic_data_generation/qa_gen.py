# -*- coding: utf-8 -*-
import os
import pandas as pd
from typing import Callable, Optional, List, Dict, Any
from openai import OpenAI
import uuid
import asyncio
from tqdm import tqdm
import asyncio
from time import sleep
from llama_index.llms.gemini import Gemini


# 1. Câu hỏi phải luôn có tên thủ tục.

qa_prompt = """
Bạn là một AI được giao nhiệm tạo ra bộ câu hỏi và trả lời TIẾNG VIỆT để huấn luyện và đánh giá model từ các văn bản về thủ tục dịch vụ công được cung cấp trước.
Nguồn thông tin: Cả câu hỏi và câu trả lời phải được trích xuất trực tiếp từ văn bản đã cho. Không suy đoán hoặc thêm thông tin ngoài những gì có trong văn bản.

Hướng dẫn:
Yêu cầu về câu hỏi:
1. Khi tạo câu hỏi, phải luôn kèm theo tên thủ tục trong văn bản trích xuất, vì văn bản đó sẽ bị loại bỏ khi đánh giá câu hỏi này.
2. Câu hỏi phải được tạo ra từ thông tin đã được cung cấp, không hỏi những câu không rõ ràng hoặc không liên quan.
3. Các câu hỏi luôn phải khác nhau.
4. Ưu tiên hỏi về các thành phần phức tạp trong văn bản cho trước.
5. Câu hỏi được tạo ra phải khó (câu trả lời cho câu hỏi phải dài).
6. Câu hỏi không được có chữ "này", nếu có, hãy thay chữ "này" thành tên của thủ tục được cung cấp.

Yêu cầu về câu trả lời:
1. Trả lời trực tiếp, không nhắc lại câu hỏi.
2. Câu trả lời phải đầy đủ, chính xác và được trích dẫn trực tiếp từ văn bản.
3. Trả lời một cách rõ ràng và không mơ hồ, không sử dụng các đại từ chung chung như "nó", "điều này", v.v.
4. Câu hỏi phải có tính đa dạng về cách diễn đạt nhưng vẫn giữ được sự nhất quán về nội dung thông tin. Dùng các từ đồng nghĩa hoặc cách diễn đạt khác nhau để hỏi về các khía cạnh khác nhau của thủ tục.
5. Số lượng câu hỏi: Kết quả phải bao gồm số lượng câu hỏi và câu trả lời được yêu cầu.

Ví dụ về câu hỏi sai:
Tên của thủ tục này là gì?
Cấp thực hiện của thủ tục này là gì?
Ai là người thực hiện thủ tục này?

Giải thích lí do các câu hỏi trên không hợp lệ:
Câu hỏi "Tên của thủ tục này là gì?" không hợp lệ vì không có tên thủ tục. Phải sửa chữ "này" thành tên của thủ tục được cung cấp.
Câu hỏi "Cấp thực hiện của thủ tục này là gì?" không hợp lệ vì không có tên thủ tục. Phải sửa chữ "này" thành tên của thủ tục được cung cấp.
Câu hỏi "Ai là người thực hiện thủ tục này?" không hợp lệ vì không có tên thủ tục, phải thêm tên thủ tục vào. Lí do là vì văn bản được sử dụng để trích xuất sẽ bị loại bỏ.

Kết quả với 2 câu hỏi và trả lời:
[Q]: Câu hỏi nào đó
[A]: Câu trả lời cho câu hỏi trên
[Q]: Câu hỏi nào đó
[A]: Câu trả lời cho câu hỏi trên

Văn bản:

{{text}}

Kết quả với {{num_questions}} câu hỏi và trả lời:

"""

class Question_Generation:
    def __init__(
        self,
        context_df: pd.DataFrame,
        llm: str,
        output_path: str,
        batch_size: int,
        api_key_dictionary
    ) -> pd.DataFrame:
        if context_df is None:
            raise ValueError('context_df is required')
        if llm is None:
            raise ValueError('llm is required')
        self.api_key_dictionary = api_key_dictionary
        self.corpus_df = context_df
        self.output_dataframe = None
        self.context_size = len(context_df)
        self.llm_name = llm

        # cases LLM 
        if self.llm_name == "pixtral-12b-2409" or self.llm_name =='mistral-large-latest' or self.llm_name == 'mistral-small-latest':
            self.llm = OpenAI(api_key=self.api_key_dictionary[llm],base_url="https://api.mistral.ai/v1")
        
        if self.llm_name == 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo' or self.llm_name == 'Qwen/Qwen2-72B-Instruct':
            self.llm = OpenAI(api_key=self.api_key_dictionary[llm],base_url="https://api.together.ai/v1")

        if self.llm_name == 'gemma2-9b-it':
            self.llm = OpenAI(api_key=self.api_key_dictionary[llm],base_url="https://api.groq.com/openai/v1")

        if self.llm_name == 'qwen/qwen-2.5-72b-instruct' or self.llm_name == 'qwen/qwen-2-vl-72b-instruct' or self.llm_name == "open-mistral-nemo-2407" or self.llm_name == "google/gemma-2-9b-it:free" or self.llm_name == "google/gemini-flash-8b-1.5-exp":
            self.llm = OpenAI(api_key=self.api_key_dictionary[llm],base_url="https://openrouter.ai/api/v1")
        
        if self.llm_name == 'gemini-1.5-flash-8b-exp-0924':
            self.llm = Gemini(model='models/'+self.llm_name,api_key=api_key_dictionary[self.llm_name],temperature=0.1)

        self.output_path = output_path
        self.batch_size = batch_size
        
    def parse_output(self, result: str) -> List[Dict]:
        result = result.strip()
        result = result.split("[Q]:")
        final_result = list()
        for res in result:
            res = res.strip()
            if res and "\n[A]:" in res:
                qa = res.split("\n[A]:")
                final_result.append({
                    'query': qa[0].strip(),
                    'generation_gt': qa[1].strip()
                })
        return final_result
    
    async def process_batch(self, tasks, batch) -> List[Any]:
        results = []

        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + batch]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)

        return results

    def save_parquet_safe(self, df: pd.DataFrame, filepath: str, upsert: bool = False): 
        output_file_dir = os.path.dirname(filepath)
        if not os.path.isdir(output_file_dir):
            raise NotADirectoryError(f"directory {output_file_dir} not found.")
        if not filepath.endswith("parquet"):
            raise NameError(f'file path: {filepath}  filename extension need to be ".parquet"')
        if os.path.exists(filepath) and not upsert:
            raise FileExistsError(f"file {filepath} already exists."
                                "Set upsert True if you want to overwrite the file.")
        df.to_parquet(filepath, index=False)

    def validate_llama_index_prompt(self, prompt: str) -> bool:
        if "{{text}}" not in prompt:
            raise ValueError("The prompt must include the placeholder {{text}}.")
        if "{{num_questions}}" not in prompt:
            raise ValueError("The prompt must include the placeholder {{num_questions}}.")
        return True
    
    async def async_qa_gen_llama_index(
        self,
        content: str,
        llm,
        prompt: str,
        question_num,
    ):
        self.validate_llama_index_prompt(prompt)

        async def generate(content: str, llm):
            while True:
                try:
                    if self.llm_name == 'gemini-1.5-flash-8b-exp-0924':
                        output = await self.llm.acomplete(prompt.replace("{{text}}", content).replace("{{num_questions}}", str(question_num)))
                        result = self.parse_output(output.text)
                    else:
                        output = llm.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt.replace("{{text}}", content).replace("{{num_questions}}", str(question_num)),
                                }
                            ],
                            model=self.llm_name
                        )
                        result = self.parse_output(output.choices[0].message.content)

                    if len(result) == question_num:
                        return result

                except Exception as e:
                    tqdm.write(f"Error in async_qa: {str(e)}")
                    await asyncio.sleep(2)
                    continue
                # raise InterruptedError(f"Failed to generate output of length {question_num} after {max_retries} retries.")
        return await generate(content, llm)
    
    async def generate_qa_llama_index(
        self,
        llm,
        contents: List[str],
        prompt: Optional[str] = None,
        question_num_per_content: int = 5,
        batch: int = 4,
    ) -> List[List[Dict]]:
        
        tasks = [
            self.async_qa_gen_llama_index(
                content, llm, prompt, question_num_per_content
            )
            for content in contents
        ]

        # Use await instead of run_until_complete
        results = await self.process_batch(tasks, self.batch_size)
        return results
    
    async def make_query_generation_gt(self, row):
        return row['qa']['query'], row['qa']['generation_gt']

    async def run(self, upsert: bool = True) -> pd.DataFrame:
        cache_batch = self.batch_size
        corpus_df = self.corpus_df
        output_filepath = self.output_path
        content_size = self.context_size

        assert content_size > 0, "content_size must be greater than 0."
        if content_size > len(corpus_df):
            raise ValueError(f"content_size {content_size} is larger than the corpus size {len(corpus_df)}. "
                            "Setting content_size to the corpus size.")
        content_size = len(corpus_df)

        qa_data = pd.DataFrame()

        total_batches = len(range(0, len(corpus_df), cache_batch))
        pbar = tqdm(total=total_batches, ncols=100, desc="Processing")
        # tqdm.write('\n')  # Using tqdm.write to avoid duplication with progress bar

        for idx, i in enumerate(range(0, len(corpus_df), cache_batch)):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    qa = await self.generate_qa_llama_index(
                        contents=corpus_df['contents'].tolist()[i:i + cache_batch],
                        llm=self.llm,
                        prompt=qa_prompt
                    )

                    temp_qa_data = pd.DataFrame({
                        'qa': qa,
                        'retrieval_gt': corpus_df['doc_id'].tolist()[i:i + cache_batch],
                    })
                    temp_qa_data = temp_qa_data.explode('qa', ignore_index=True)
                    temp_qa_data['qid'] = [str(uuid.uuid4()) for _ in range(len(temp_qa_data))]

                    temp_qa_data[['query', 'generation_gt']] = await asyncio.gather(
                        *[self.make_query_generation_gt(row) for _, row in temp_qa_data.iterrows()]
                    )
                    
                    temp_qa_data = temp_qa_data.drop(columns=['qa'])

                    temp_qa_data['retrieval_gt'] = temp_qa_data['retrieval_gt'].apply(lambda x: [[x]])
                    temp_qa_data['generation_gt'] = temp_qa_data['generation_gt'].apply(lambda x: [x])

                    if idx == 0:
                        qa_data = temp_qa_data
                    else:
                        qa_data = pd.concat([qa_data, temp_qa_data], ignore_index=True)

                    if output_filepath is not None:
                        self.save_parquet_safe(qa_data, output_filepath, upsert=upsert)

                    pbar.update(1)
                    break 

                except Exception as e:
                    tqdm.write(f"Retry {attempt + 1}/{max_retries} due to error: {e}")
                    if attempt == max_retries - 1:
                        tqdm.write(f"Skip index {idx} due to {e}")
                    await asyncio.sleep(10)
                    continue
        pbar.close()
        if output_filepath is not None:
            tqdm.write(f"QA DataFrame has been saved to: {output_filepath}\n")

        return qa_data


# corpus_df = pd.read_parquet('data/corpus.parquet')
# qa_generator = Question_Generation(corpus_df[0:10], 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',output_path='data/test_pipeline/test.parquet', batch_size=4, api_key='c4a64dc6084719dc34057f3e3c1fb8bc825ede923032fe0292a3f2f21e39c822')
# test_df = asyncio.run(qa_generator.run())