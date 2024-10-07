import pandas as pd
import ast
from llama_index.llms.gemini import Gemini
import asyncio
from tqdm import tqdm

sim_prompt = """
Bạn là một trợ lý giúp paraphrasing 1 câu hỏi thành {num_question} câu hỏi khác nhau.

Tiêu chí tạo ra câu hỏi mới:
1. Chỉ cần câu hỏi, không cần số thứ tự câu hỏi ở trước.
2. các câu hỏi được paraphrase phải tương đồng với câu hỏi ban đầu.
3. ưu tiên sự sáng tạo trong paraphrase câu hỏi, nhằm giúp đánh giá một quy trình trả lời tự động.
4. câu hỏi được paraphrase không được tự ý thêm bất kỳ thứ gì không liên quan đến câu hỏi ban đầu.
5. Không cần có \n mà chỉ cần sau mỗi câu hỏi hãy thêm kí tự [END].

Câu hỏi ban đầu: {question}

{num_question} câu hỏi khác được paraphrase: 
"""
gemini_api_key = "AIzaSyAokJTNFmApBzp6tHEaX9P_cbvkkxkj6r8"

class ParaphrasingQuestion:
    def __init__(
        self,
        input_dataframe: pd.DataFrame,
        model_name: str = None,
        output_path: str = 'output.csv',
        # prompt: str = 'default prompt',
        batch_size: int = 4,
        api_key_dictionary: dict = None,
    ):
        if input_dataframe is None:
            raise ValueError("input_dataframe is required")
        if model_name is None:
            self.model_name = "models/gemini-1.5-flash"
        else:
            self.model_name = model_name
        # if prompt is None:
        #     raise ValueError("prompt is required")

        self.input_dataframe = input_dataframe
        if api_key_dictionary:
            self.llm = Gemini(model=self.model_name,api_key=api_key_dictionary['gemini'],temperature=0)
        else:
            self.llm = Gemini(model=self.model_name,api_key=gemini_api_key,temperature=0)

        self.result_dataframe = pd.DataFrame()
        self.output_path = output_path
        self.prompt = sim_prompt
        self.batch_size = batch_size

    def __str__(self):
        return f"SimilarityCheck(input_dataframe={self.input_dataframe}, model_name={self.model_name})"
    
    def convert_to_list(self, answer_str):
        try:
            return ast.literal_eval(answer_str)
        except (ValueError, SyntaxError):
            return []
        
    async def paraphrasing_run(self):
        total_batches = len(self.input_dataframe)
        pbar = tqdm(total=total_batches, ncols=100, desc="Processing")

        # Initialize the result dataframe
        if not hasattr(self, 'result_dataframe'):
            self.result_dataframe = pd.DataFrame()

        failed_dataframe = pd.DataFrame()
        temp_df = self.input_dataframe.copy()

        # Convert 'Answers' to lists and count the number of answers
        temp_df['Answers'] = temp_df['Answers'].apply(self.convert_to_list)
        temp_df['num_answers'] = temp_df['Answers'].apply(len)

        for index, row in temp_df.iterrows():
            question = row['Question']
            num_question = str(row['num_answers']-1)

            # Paraphrase the question
            list_of_question = await self.paraphrase(question, num_question)

            # Handle failed paraphrasing cases
            if list_of_question == -1:
                failed_dataframe = pd.concat([failed_dataframe, pd.DataFrame([row])], ignore_index=True)
                failed_dataframe.to_csv(self.output_path.replace('.csv', '_failed_rows.csv'), index=False)
                continue

            # Update progress bar
            pbar.update(1)

            # Ensure both list_of_question and Answers are non-empty
            if list_of_question and row['Answers']:
                # Split the lists into individual rows
                for q, a in zip(list_of_question, row['Answers']):
                    new_row = {
                        'Context': row['Context'],  # Assuming the 'Context' column is present
                        'Question': q.replace('\n', ''),
                        'Answer': a.replace('\n', '')
                    }

                    # Append new row to the result dataframe
                    self.result_dataframe = pd.concat([self.result_dataframe, pd.DataFrame([new_row])], ignore_index=True)

        # Close the progress bar
        pbar.close()

        # Save the result dataframe containing only 'Context', 'Question', and 'Answer'
        self.result_dataframe.to_csv(self.output_path, columns=['Context', 'Question', 'Answer'], index=False)

        return self.result_dataframe

    async def batch_process_paraphrasing(self, batch, failed_dataframe):
        tasks = []
        for row in batch:
            question = row['Question']
            num_question = str(row['num_answers'] - 1)

            # Create a task for paraphrasing
            tasks.append(self.paraphrase(question, num_question))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results
        for row, list_of_question in zip(batch, results):
            if isinstance(list_of_question, Exception) or list_of_question == -1:
                # If paraphrasing fails, log the row in failed_dataframe
                failed_dataframe = pd.concat([failed_dataframe, pd.DataFrame([row])], ignore_index=True)
            else:
                # Ensure both list_of_question and Answers are non-empty
                if list_of_question and row['Answers']:
                    # Split the lists into individual rows
                    for q, a in zip(list_of_question, row['Answers']):
                        new_row = {
                            'Context': row['Context'],  # Assuming 'Context' column is present
                            'Question': q.replace('\n', ''),
                            'Answer': a.replace('\n', '')
                        }

                        # Append new row to result dataframe
                        self.result_dataframe = pd.concat([self.result_dataframe, pd.DataFrame([new_row])], ignore_index=True)

        return failed_dataframe


    async def pipeline_paraphrasing_run(self):
        batch_size = self.batch_size
        total_batches = len(range(0, len(self.input_dataframe), batch_size))
        pbar = tqdm(total=total_batches, ncols=100, desc="Processing")

        # Initialize the result dataframe if not already initialized
        if not hasattr(self, 'result_dataframe'):
            self.result_dataframe = pd.DataFrame()

        failed_dataframe = pd.DataFrame()
        temp_df = self.input_dataframe.copy()

        # Convert 'Answers' to lists and count the number of answers
        temp_df['num_answers'] = temp_df['Answers'].apply(len)

        batch = []

        for index, row in temp_df.iterrows():
            batch.append(row)

            if len(batch) == batch_size or index == total_batches - 1:  # Process in batches of 4 or at the last batch
                failed_dataframe = await self.batch_process_paraphrasing(batch, failed_dataframe)

                # Reset the batch after processing
                batch = []

            # Update progress bar
            pbar.update(len(batch))

        pbar.close()

        # Save the result dataframe containing only 'Context', 'Question', and 'Answer'
        self.result_dataframe.to_csv(self.output_path.replace('.csv', "_paraphrased_final_dataframe.csv"), columns=['Context', 'Question', 'Answer'], index=False)

        # Save the failed rows
        failed_path = self.output_path.replace('.csv', '_failed_rows.csv')
        await asyncio.to_thread(failed_dataframe.to_csv, failed_path, index=False)

        print(f"Failed paraphrased rows are saved to: {failed_path}\n")
        print(f"Paraphrased dataframe is saved to: {self.output_path.replace('.csv', '_paraphrased_final_dataframe.csv')}\n")

        return self.result_dataframe


    async def paraphrase(self, question, num_question):
        retries = 3
        delay = 20

        for attempt in range(retries):
            try:
                prompt = sim_prompt.replace("{num_question}", num_question).replace("{question}", question)
                output = await self.llm.acomplete(sim_prompt.replace("{num_question}", num_question).replace("{question}", question))
                return [question] + output.text.split('[END]')[:-1]

            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")

                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print("Max retries reached. Logging failed case.")
                    return -1
