from api.openai.chat import chat

def gpt4o_chat(prompt_list: list[str]) -> list[str]:
    return chat(prompt_list, model_name='gpt-4o', seed=42)

if __name__ == '__main__':
    prompt_list = ['What is your favorite food?', 'What is your favorite movie?']
    response_list = gpt4o_chat(prompt_list)
    for response in response_list:
        print(response)