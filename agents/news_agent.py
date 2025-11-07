from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, MODEL_NAME

llm = ChatGroq(api_key="GROQ_API_KEY", model_name=MODEL_NAME)
prompt = PromptTemplate(
    input_variables=["news"],
    template="""
Summarize the following financial news:
{news}
Provide a concise overview and insights.
"""
)
chain = LLMChain(llm=llm, prompt=prompt)

def summarize_news(news):
    return chain.run({"news": news})
