# import pandas as pd
# import ast

# # Converting
# def convert_to_list(answer_str):
#     try:
#         return ast.literal_eval(answer_str)
#     except (ValueError, SyntaxError):
#         return []

# def aggregate_answers(list_of_lists):
#     combined_list = []
#     seen_items = set()
#     for sublist in list_of_lists:
#         for item in sublist:
#             if item not in seen_items:
#                 seen_items.add(item)
#                 combined_list.append(item)
#     return combined_list

# verifed_answer_mistral = pd.read_csv("mistral_sim_dataframe.csv")
# verifed_answer_llama3  = pd.read_csv("llama3_sim_dataframe.csv")
# verifed_answer_qwen = pd.read_csv("qwen_sim_dataframe.csv")

# intersection_df = pd.merge(verifed_answer_mistral, verifed_answer_llama3, on='Context', how='inner')
# intersection_df = pd.merge(intersection_df, verifed_answer_qwen, on='Context', how='inner')
# grouped_df = intersection_df.groupby('Context')
# grouped_result = grouped_df.size().reset_index(name='Counts')

# common_contexts = intersection_df['Context']

# # Filter
# filtered_mistral = verifed_answer_mistral[verifed_answer_mistral['Context'].isin(common_contexts)]
# filtered_llama3 = verifed_answer_llama3[verifed_answer_llama3['Context'].isin(common_contexts)]
# filtered_qwen = verifed_answer_qwen[verifed_answer_qwen['Context'].isin(common_contexts)]

# # Combine
# combined_df = pd.concat([filtered_mistral, filtered_llama3, filtered_qwen])
  
# # Aggregation
# combined_df['Answers'] = combined_df['Answers'].apply(convert_to_list)
# grouped_df = combined_df.groupby(['Context', 'Question'], as_index=False)
# grouped_df = grouped_df.agg({'Answers': aggregate_answers})

import pandas as pd
import ast

class AggregationDataframe:
    def __init__(self, dataframes, output_path):
        """
        Initializes the class with a list of DataFrames.

        Args:
            dataframes (list): A list of pandas DataFrames.
        """
        self.dataframes = dataframes
        self.output_path = output_path
        self.intersection_df = None
        self.combined_df = None
        self.grouped_df = None

    # Converts answer strings into lists
    def convert_to_list(self, answer_str):
        try:
            return ast.literal_eval(answer_str)
        except (ValueError, SyntaxError):
            return []

    # Aggregates answers, removing duplicates
    def aggregate_answers(self, list_of_lists):
        combined_list = []
        seen_items = set()
        for sublist in list_of_lists:
            for item in sublist:
                if item not in seen_items:
                    seen_items.add(item)
                    combined_list.append(item)
        return combined_list

    # Merge DataFrames based on 'Context'
    def merge_dataframes(self):
        """
        Merges all DataFrames in the list on the 'Context' column using an inner join.
        """
        self.intersection_df = self.dataframes[0]
        for df in self.dataframes[1:]:
            self.intersection_df = pd.merge(self.intersection_df, df, on='Context', how='inner')

    # Filter the DataFrames to keep only the common 'Context' values
    def filter_common_contexts(self):
        """
        Filters all DataFrames to only keep rows with 'Context' values present in all DataFrames.
        """
        common_contexts = self.intersection_df['Context']
        self.dataframes = [df[df['Context'].isin(common_contexts)] for df in self.dataframes]

    # Combine all filtered DataFrames
    def combine_dataframes(self):
        """
        Concatenates all filtered DataFrames into one combined DataFrame.
        """
        self.combined_df = pd.concat(self.dataframes)

    # Apply aggregation by grouping 'Context' and 'Question', and aggregating 'Answers'
    def aggregate(self):
        """
        Aggregates the 'Answers' column by converting strings to lists and removing duplicates.
        """
        self.combined_df['Answers'] = self.combined_df['Answers'].apply(self.convert_to_list)
        self.grouped_df = self.combined_df.groupby(['Context', 'Question'], as_index=False)
        self.grouped_df = self.grouped_df.agg({'Answers': self.aggregate_answers})

    def pipeline_aggregate(self):
        """
        Aggregates the 'Answers' column by converting strings to lists and removing duplicates.
        """
        # self.combined_df['Answers'] = self.combined_df['Answers'].apply(self.convert_to_list)
        self.grouped_df = self.combined_df.groupby(['Context', 'Question'], as_index=False)
        self.grouped_df = self.grouped_df.agg({'Answers': self.aggregate_answers})

    # Run the full pipeline
    def run(self):
        """
        Runs the full pipeline: merge, filter, combine, and aggregate.
        """
        self.merge_dataframes()
        self.filter_common_contexts()
        self.combine_dataframes()
        self.aggregate()
        if (self.output_path is not None):
            print(f"The aggregation data has been saved to {self.output_path}")
            self.grouped_df.to_csv(self.output_path)
        return self.grouped_df

    def pipeline_run(self):
        """
        Runs the full pipeline: merge, filter, combine, and aggregate.
        """
        self.merge_dataframes()
        self.filter_common_contexts()
        self.combine_dataframes()
        self.pipeline_aggregate()
        if (self.output_path is not None):
            print(f"The aggregation data has been saved to {self.output_path}")
            self.grouped_df.to_csv(self.output_path)
        return self.grouped_df

# df1 = pd.read_csv("E:/thesis/RAG4PublicSector/data/data_gen_from_pipeline/pipeline_index_400_500/gemini-1.5-flash-8b-exp-0924_verified_answer_dataframe.csv")
# df2 = pd.read_csv("E:/thesis/RAG4PublicSector/data/data_gen_from_pipeline/pipeline_index_400_500/meta-llama_verified_answer_dataframe.csv")
# df3 = pd.read_csv("E:/thesis/RAG4PublicSector/data/data_gen_from_pipeline/pipeline_index_400_500/mistral-large-latest_verified_answer_dataframe.csv")
# dataframes = [df1, df2, df3]
# aggregator = AggregationDataframe(dataframes, "E:/thesis/RAG4PublicSector/data/data_gen_from_pipeline/pipeline_index_400_500/final_non_processed_3_answer_dataframe_400_500.csv")
# result_df = aggregator.run()
