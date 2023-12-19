
"""
The class uses the Langchains Q and Answering library to generate questions and answers.
It first generates the Qusertion and Answers based on the chunks that it is supplied.
Then it runs a evaluation chain to grade the output from the Question and Answering again the real LLM
invocation to get the answer.

"""
from langchain.llms import Cohere
from langchain.chat_models import ChatOpenAI


from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import QAGenerationChain
from langchain.chains import LLMChain
from langchain.evaluation.qa import QAEvalChain
import pandas as pd

#Make sure you have your OPEN AI or COHERE keys in key.env
load_dotenv('key.env')

def load_model(type:str):
    if type == 'COHERE':
        llm = Cohere(model="command", temperature=0)
    elif type == 'OPEN_AI':
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    return llm

"""
https://api.python.langchain.com/en/latest/chains/langchain.chains.qa_generation.base.QAGenerationChain.html
https://api.python.langchain.com/en/latest/evaluation/langchain.evaluation.qa.eval_chain.QAEvalChain.html
"""
def generate_questions_and_answers_and_evalaute (filename:str):

    loader = PyPDFLoader(filename)
    pages = loader.load()
    llm = load_model('OPEN_AI')
    chain = QAGenerationChain.from_llm(llm)

    examples = []
    for page in pages:
        try:
            examples.append(chain.run(page.page_content.replace(":", " ").replace('"', ''))[0])
        except Exception as e:
            print(e)

    if len(examples) ==0:
        print('Do not have any chunks to proceed!!!!!')
        return
    #Generate the Questions and Answers in a specifc format
    prompt = PromptTemplate(template="Question : {question}\n Answer:", input_variables=["question"])
    chain = LLMChain(llm=llm, prompt=prompt)
    predictions = chain.apply(examples)

    #Evalaute against the prdictions
    eval_chain = QAEvalChain.from_llm(llm)
    graded_output = eval_chain.evaluate(examples, predictions, question_key="question", prediction_key="text")

    questions = []
    answers = []
    real_answers = []
    graded_answers = []
    for i, example in enumerate(examples):
        print(f"Example {i}")
        print(f"Question: " + example["question"])
        questions.append(example["question"])
        print(f"Answer: " + example["answer"])
        answers.append(example["answer"])
        print(f"Real Answer: " + predictions[i]["text"])
        real_answers.append(predictions[i]["text"])
        print(f"Graded: " + graded_output[i]["text"])
        graded_answers.append(graded_output[i]["text"])
    write_to_excel('eval_test.xlsx', questions, answers, real_answers, graded_answers)


def write_to_excel(file_name:str, questions,ans_from_qa_gen_langchain, real_ans_from_llm_langchain,real_graded_langchain):
    columns = ['Question', 'Answer for QAGeneration Chain', 'Real Answer from LLM', 'GRADED']
    df1 = pd.DataFrame(list(zip(questions,ans_from_qa_gen_langchain,real_ans_from_llm_langchain,real_graded_langchain)), columns=columns)
    df1.to_excel(file_name,  sheet_name='Eval')

if __name__ == "__main__":
    generate_questions_and_answers_and_evalaute('Test.pdf')

