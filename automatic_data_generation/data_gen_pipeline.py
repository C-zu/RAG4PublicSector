import argparse
from aggregation import AggregationDataframe
from verify_answer_gen import SimilarityCheck, sim_prompt
from verify_question_gen import CrossCheckingLLMs
from qa_gen import Question_Generation
from paraphase_question import ParaphrasingQuestion
import os
import pandas as pd
from openai import OpenAI
import asyncio
import shutil
import re
from tqdm import tqdm
import ast

together_api_key = "b2364e8538f36185c3c7551f17ca2f730e3d0b495b5850faa4888a7043e0fc11" #trongnghiakazuhatran@gmail.com
gemini_api_key = "AIzaSyD8zWm_BwywzU5uvjsqeMhLSKk6XmoWW40" #bapngan
mistral_api_key = "jDVi03OCpbFVWaAZAd8gpa9JOL8mjdqU"
groq_api_key = "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"
open_router_api_key = "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"

api_key_dictionary = {
    'pixtral-12b-2409': mistral_api_key,
    'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo': together_api_key,
    'Qwen/Qwen2-72B-Instruct': together_api_key,
    'gemini': gemini_api_key,
    'gemma2-9b-it': groq_api_key,
    'qwen/qwen-2.5-72b-instruct': open_router_api_key,
    'qwen/qwen-2-vl-72b-instruct': open_router_api_key,
    'open-mistral-nemo-2407': open_router_api_key,
    'google/gemma-2-9b-it:free':open_router_api_key,
    'google/gemini-flash-8b-1.5-exp': open_router_api_key,
    'gemini-1.5-flash-8b-exp-0924': gemini_api_key,
    'mistral-large-latest':mistral_api_key,
    'mistral-small-latest': mistral_api_key
}



class DataPipeline:
    def __init__(self, context, LLM_list, start_index, end_index, output_path, api_key_dictionary):
        """
        Automatically create a dataframe for evaluating phase.

        Args:
            context (dataframe): A dataframe that contains all of the public sector data.
            
            LLM_list (List): a list that contains llm interface which are used in voting phase.
            Model for LLM_list:
                - mistralai/Mistral-7B-Instruct-v0.3
                - meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
                - Qwen/Qwen2-72B-Instruct

        Output: 1 dataframe contains all the verified data.
        """
        if context is None:
            raise ValueError("context is required")
        if LLM_list is None:
            raise ValueError("LLM_list is required")
        
        available_LLM_list = [
            'mistral-small-latest', 
            'mistral-large-latest',
            'gemini-1.5-flash-8b-exp-0924',
            'google/gemini-flash-8b-1.5-exp',
            'google/gemma-2-9b-it:free',
            'open-mistral-nemo-2407',
            'open-mixtral-8x7b', 
            'qwen/qwen-2-vl-72b-instruct',
            'qwen/qwen-2.5-72b-instruct',
            'gemma2-9b-it',
            'pixtral-12b-2409',
            'databricks/dbrx-instruct',
            'mistralai/Mistral-7B-Instruct-v0.1',  
            'microsoft/WizardLM-2-8x22B', 
            'mistralai/Mistral-7B-Instruct-v0.3', 
            'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', 
            'Qwen/Qwen2-72B-Instruct',
            'google/gemma-2-9b-it']
        
        for llm in LLM_list:
            if llm not in available_LLM_list:
                raise ValueError(llm + "is not supported")
            self.LLM_list = LLM_list

        # return valuabl
        self.api_key_dictionary = api_key_dictionary
        self.context = context
        self.QA_dataframe = []
        self.question_verification_dataframe = []
        self.answer_verification_dataframe = []
        self.final_dataframe = None
        self.paraphrased_dataframe = None
        self.start_index = start_index
        self.end_index = end_index

        recent_path = os.getcwd() + '/'
        self.output_path = recent_path + output_path + f"/pipeline_index_{start_index}_{end_index}"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        else:
            for filename in os.listdir(self.output_path):
                file_path = os.path.join(self.output_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path) 
                except Exception as e:
                    tqdm.write(f"Failed to delete {file_path}. Reason: {e}")
        
    async def qa_generation(self):
        tasks = []

        for index, llm_name in enumerate(self.LLM_list):
            tqdm.write(f"Generating QA data using model: {llm_name}")

            qa_generator = Question_Generation(
                self.context,
                llm_name,
                output_path=self.output_path + f'/{llm_name.split("/")[0]}.parquet',
                batch_size=2,
                # prompt=qa_prompt,
                api_key_dictionary=self.api_key_dictionary
            )

            tasks.append(self.generate_question(qa_generator, llm_name))

        await asyncio.gather(*tasks)

    async def generate_question(self, qa_generator, llm_name):
        qa_dataframe = await qa_generator.run()
        self.QA_dataframe.append(qa_dataframe)

    async def async_question_verification(self):
        q_tasks = []
        for index, df in enumerate(self.QA_dataframe):
            list_llm_checkers = self.LLM_list[:index] + self.LLM_list[index + 1:]
            df_name = self.LLM_list[index].split('/')[0]

            checker = CrossCheckingLLMs(
                self.context,
                df,
                None,
                list_llm_checkers,
                output_path=self.output_path + f'/{df_name}_verified_question_dataframe.csv',
                api_key_dictionary=self.api_key_dictionary,
            )

            q_tasks.append(checker.cross_check())

        # Thực thi tất cả các nhiệm vụ cùng lúc
        local_results = await asyncio.gather(*q_tasks)

        # Lưu kết quả vào `self.question_verification_dataframe`
        self.question_verification_dataframe.extend(local_results)
        for index, df in enumerate(self.question_verification_dataframe):
            df.to_csv(f"{self.output_path}/{index}.csv")
        return None

        

    # async def answer_verification(self, list_verified_question):
    #         print(f"list_verified_question length: {len(list_verified_question)}")
    #         print(f"self.LLM_list length: {len(self.LLM_list)}") 
    #         df1 = pd.read_csv(f"{self.output_path}/gemini-1.5-flash-8b-exp-0924_verified_question_dataframe.csv")
    #         df2 = pd.read_csv(f"{self.output_path}/meta-llama_verified_question_dataframe.csv")
    #         df3 = pd.read_csv(f"{self.output_path}/mistral-large-latest_verified_question_dataframe.csv")
    #         list_verified_question = [df1,df2,df3]
            
    #         for index, df in enumerate(list_verified_question):
    #             df_name = self.LLM_list[index].split('/')[0]
    #             sim_checker = SimilarityCheck(
    #                 df,
    #                 batch_size=8,
    #                 output_path=self.output_path + f'/{df_name}_verified_answer_dataframe.csv',
    #                 llm='gemini-1.5-flash',
    #                 # prompt=sim_prompt,  # Uncomment if needed
    #                 api_key_dictionary=self.api_key_dictionary
    #             )
    #             verified_answers_df = await sim_checker.pipeline_check_similarity()  
    #             self.answer_verification_dataframe.append(verified_answers_df)
    #         return None
    
    async def answer_verification(self, list_verified_question):
        print(f"list_verified_question length: {len(list_verified_question)}")
        print(f"self.LLM_list length: {len(self.LLM_list)}") 

        try:
            for index, df in enumerate(list_verified_question):
                df_name = self.LLM_list[index].split('/')[0]
                sim_checker = SimilarityCheck(
                    df,
                    batch_size=8,
                    output_path=self.output_path + f'/{df_name}_verified_answer_dataframe.csv',
                    llm='gemini-1.5-flash',
                    # prompt=sim_prompt,  # Uncomment if needed
                    api_key_dictionary=self.api_key_dictionary
                )
                verified_answers_df = await sim_checker.pipeline_check_similarity()  
                self.answer_verification_dataframe.append(verified_answers_df)
        except Exception as e:
            print(f"Error processing DataFrame for index {index}: {e}")
            # Read all DataFrames from files
            try:
                df1 = pd.read_csv(f"{self.output_path}/gemini-1.5-flash-8b-exp-0924_verified_question_dataframe.csv")
                df2 = pd.read_csv(f"{self.output_path}/meta-llama_verified_question_dataframe.csv")
                df3 = pd.read_csv(f"{self.output_path}/mistral-large-latest_verified_question_dataframe.csv")
                list_verified_question = [df1, df2, df3]
                print("Successfully reloaded all DataFrames from files.")
                for index, df in enumerate(list_verified_question):
                    df_name = self.LLM_list[index].split('/')[0]
                    sim_checker = SimilarityCheck(
                        df,
                        batch_size=8,
                        output_path=self.output_path + f'/{df_name}_verified_answer_dataframe.csv',
                        llm='gemini-1.5-flash',
                        # prompt=sim_prompt,  # Uncomment if needed
                        api_key_dictionary=self.api_key_dictionary
                    )
                    verified_answers_df = await sim_checker.pipeline_check_similarity()  
                    self.answer_verification_dataframe.append(verified_answers_df)
            except Exception as reload_error:
                print(f"Failed to reload DataFrames from files: {reload_error}")
                return None  # Exit the function if reloading fails
        return None
        
    def aggregation_dataframe(self):
        try:
            combination_interface = AggregationDataframe(
                self.answer_verification_dataframe, 
                output_path=self.output_path + f'/final_dataframe_{self.start_index}_{self.end_index}.csv'
            )
            self.final_dataframe = combination_interface.pipeline_run()
        except Exception as e:
            tqdm.write(f"Error occurred in pipeline run: {e}")
            
            try:
                df1 = pd.read_csv(f"{self.output_path}/gemini-1.5-flash-8b-exp-0924_verified_answer_dataframe.csv")
                df2 = pd.read_csv(f"{self.output_path}/meta-llama_verified_answer_dataframe.csv")
                df3 = pd.read_csv(f"{self.output_path}/mistral-large-latest_verified_answer_dataframe.csv")
                dataframes = [df1, df2, df3]
                aggregator = AggregationDataframe(
                    dataframes, 
                    f"{self.output_path}/final_non_processed_3_answer_dataframe.csv"
                )
                self.final_dataframe = aggregator.run()
                tqdm.write("Successfully ran aggregation from fallback method.")
            except Exception as fallback_error:
                tqdm.write(f"Error occurred in fallback aggregation: {fallback_error}")
    
        # async def paraphrase_phase(self):
        #     pq = ParaphrasingQuestion(
        #         self.final_dataframe,
        #         output_path=self.output_path+f'_paraphased_final_dataframe.csv',
        #         api_key_dictionary=self.api_key_dictionary,
        #         batch_size=8
        #     )
        #     self.paraphrased_dataframe = await pq.pipeline_paraphrasing_run()
                
    def clean_data(self, line):
        line = line.replace("*","")
        line = line.replace("-","")
        return line

    def process_final_data(self):
        processed_data = self.final_dataframe.copy()
        # processed_data = pd.read_csv("E:/thesis/RAG4PublicSector/data/data_gen_from_pipeline/pipeline_index_500_502/final_non_processed_3_answer_dataframe_400_500.csv")

        processed_data ['Answers'] = processed_data ['Answers'].apply(lambda x: max(x, key=len) if x else x[0])
        processed_data['Answers'] = processed_data['Answers'].apply(self.clean_data)
        processed_data['Question'] = processed_data['Question'].apply(self.clean_data)

        for index, row in processed_data.iterrows():
            if "thủ tục này" in row['Question']:
                result = re.search(r"Tên thủ tục:\r\n(.*?)\r\nCấp thực hiện", row['Context'], re.DOTALL)

                if result:
                    extracted_string = result.group(1).strip()
                    row['Question'] = row['Question'].replace("thủ tục này", extracted_string)
        
        for index, row in processed_data.iterrows():
            result = re.search(r'\d+\..+?\?', row['Answers'], re.DOTALL)

            if result:
                pattern = r'\d+\..+?\?'
                cleaned_answer = re.sub(pattern, '', row['Answers']).strip()
                cleaned_answer = cleaned_answer.replace('-', '').strip()
                processed_data.at[index, 'Answers'] = cleaned_answer
            
        for index, row in processed_data.iterrows():
            result = re.search(r'^\d+\.\s*', row['Answers'])

            if result:
                pattern = r'^\d+\.\s*'
                cleaned_answer = re.sub(pattern, '', row['Answers']).strip()
                processed_data.at[index, 'Answers'] = cleaned_answer

        if processed_data is not None:
            tqdm.write("Processed data has been saved!")
            output_path=self.output_path+f'/final_processed_dataframe_{self.start_index}_{self.end_index}.csv'
            processed_data.to_csv(output_path)

    def run(self):
        print("=======================================")
        print("Starting QA Generation phase")
        asyncio.run(self.qa_generation())
        print("=======================================\n")

        print("Starting Verifying Question dataframes")
        print("=======================================")
        asyncio.run(self.async_question_verification())

        print("Starting Verifying Answer dataframes")
        print("=======================================\n")
        asyncio.run(self.answer_verification(self.question_verification_dataframe))

        print("Starting Aggregation all dataframes")
        print("=======================================")
        self.aggregation_dataframe()
        print("=======================================\n")

        print("Starting Cleaning all dataframes")
        print("=======================================")
        self.process_final_data()
        print("=======================================\n")
        # print("Starting Paraphrase final dataframes")
        # print("=======================================")
        # asyncio.run(self.paraphrase_phase())

    
def main():
    parser = argparse.ArgumentParser(description="Run DataPipeline with corpus size as input.")
    parser.add_argument("corpus_start", type=int, help="Number of the start row to process from the corpus")
    parser.add_argument("corpus_end", type=int, help="Number of the end row to process from the corpus")

    args = parser.parse_args()

    corpus_df = pd.read_parquet('data/corpus.parquet')

    if (args.corpus_start > len(corpus_df)) or (args.corpus_end > len(corpus_df)):
        raise ValueError("Input corpus index exceeds the total number of rows in the corpus")
    
    if args.corpus_start > args.corpus_end:
        raise ValueError("The start of row should be lower than the end row")

    pipepline = DataPipeline(corpus_df[args.corpus_start:args.corpus_end], 
                       ['mistral-large-latest', 'gemini-1.5-flash-8b-exp-0924', 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'], 
                       api_key_dictionary=api_key_dictionary,
                       start_index=args.corpus_start,
                       end_index=args.corpus_end,
                       output_path='/data/data_gen_from_pipeline')
    
    # asyncio.run(pipepline.answer_verification(1))
    pipepline.run()

if __name__ == "__main__":
    main()


