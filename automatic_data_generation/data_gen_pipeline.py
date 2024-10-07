import argparse
from aggregation import AggregationDataframe
from verify_answer_gen import SimilarityCheck, sim_prompt
from verify_question_gen import CrossCheckingLLMs
from qa_gen import Question_Generation
from paraphase_question import ParaphrasingQuestion
import os
import pandas as pd
import pandas as pd
from openai import OpenAI
import asyncio
import shutil

together_api_key = "b2364e8538f36185c3c7551f17ca2f730e3d0b495b5850faa4888a7043e0fc11" #trongnghiakazuhatran@gmail.com
gemini_api_key = "AIzaSyAokJTNFmApBzp6tHEaX9P_cbvkkxkj6r8"
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

        # return valuable
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
                    print(f"Failed to delete {file_path}. Reason: {e}")
        
    async def qa_generation(self):
        tasks = []

        for llm_name in self.LLM_list:
            print(f"Generating QA data using model: {llm_name}")

            qa_generator = Question_Generation(
                self.context,
                llm_name,
                output_path=self.output_path + f'/{llm_name.split("/")[0]}.parquet',
                batch_size=8,
                # prompt=qa_prompt, 
                api_key_dictionary=self.api_key_dictionary
            )

            tasks.append(self.generate_and_verify(qa_generator, llm_name))

        await asyncio.gather(*tasks)

    async def generate_and_verify(self, qa_generator, llm_name):
        llm_csv_name = llm_name.split('/')[0]

        qa_dataframe = await qa_generator.run()
        self.QA_dataframe.append(qa_dataframe)

        await self.question_and_answer_verification(qa_dataframe, self.LLM_list.index(llm_name))

        # Optionally return the dataframe if needed
        return qa_dataframe


    async def question_and_answer_verification(self, df, index):
        # Get the name of the LLM and create list of checkers
        list_llm_checkers = self.LLM_list[:index] + self.LLM_list[index + 1:]
        df_name = self.LLM_list[index].split('/')[0]
        
        print(f"Verifying questions from {df_name} QA dataframe...")

        # Create an instance of CrossCheckingLLMs for question verification
        checker = CrossCheckingLLMs(
            self.context,
            df,
            None,
            list_llm_checkers,
            output_path=self.output_path + f'/{df_name}_verified_question_dataframe.csv',
            api_key_dictionary=self.api_key_dictionary,
        )

        # Perform question verification and await the result
        verified_questions_df = await checker.cross_check()

        # Store the result in the question verification dataframe list
        self.question_verification_dataframe.append(verified_questions_df)

        # After question verification, proceed to answer verification
        print(f"Verifying answers from {df_name} Verified questions dataframe...")

        # Create an instance of SimilarityCheck for answer verification
        sim_checker = SimilarityCheck(
            verified_questions_df,
            batch_size=8,
            output_path=self.output_path + f'/{df_name}_verified_answer_dataframe.csv',
            llm = 'gemini-1.5-flash-8b-exp-0924',
            # prompt=sim_prompt,  # Uncomment if needed
            api_key_dictionary=self.api_key_dictionary
        )

        # Perform answer verification and await the result
        verified_answers_df = await sim_checker.pipeline_check_similarity()

        # Store the result in the answer verification dataframe list
        self.answer_verification_dataframe.append(verified_answers_df)

    async def question_and_answer_verification_run(self):
        # Create tasks for each dataframe to perform question and answer verification in parallel
        tasks = [
            self.question_and_answer_verification(df, index)
            for index, df in enumerate(self.QA_dataframe)
        ]

        # Use asyncio.gather to run all tasks in parallel
        await asyncio.gather(*tasks)

        # After all tasks are complete, return the final answer verification dataframe
        return self.answer_verification_dataframe

        
    def aggregation_dataframe(self):
        combination_interface = AggregationDataframe(self.answer_verification_dataframe, output_path=self.output_path+f'/final_dataframe_{self.start_index}_{self.end_index}.csv')
        self.final_dataframe = combination_interface.pipeline_run()

    async def paraphrase_phase(self):
        pq = ParaphrasingQuestion(
            self.final_dataframe,
            output_path=self.output_path+f'_paraphased_final_dataframe.csv',
            api_key_dictionary=self.api_key_dictionary,
            batch_size=8
        )
        self.paraphrased_dataframe = await pq.pipeline_paraphrasing_run()
            

    def run(self):
        print("Starting QA Generation phase")
        asyncio.run(self.qa_generation())
        print("=======================================")

        print("Starting Aggregation all dataframes")
        print("=======================================")
        self.aggregation_dataframe()
        print("=======================================")

        print("Starting Paraphrase final dataframes")
        print("=======================================")
        asyncio.run(self.paraphrase_phase())

    
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
    
    pipepline.run()

if __name__ == "__main__":
    main()


