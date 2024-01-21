from langchain.llms import OpenAI

# secure the key later
api_key = "xxx"
llm = OpenAI(
    openai_api_key=api_key
)

result = llm("write a very very short poem")
print(result)