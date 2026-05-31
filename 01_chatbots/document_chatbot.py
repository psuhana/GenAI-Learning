from openai import OpenAI
import os
import json

api_key = os.getenv('OPEN_API_KEY')
if api_key == None:
    print("Empty Key")
else:
    client = OpenAI(api_key= api_key)

isActive = True

if os.path.exists('memory_new.json'):
    with open('memory_new.json','r') as file:
        messages = json.load(file)
else:
    if os.path.exists('notes.txt'):
        with open('notes.txt','r') as file:
            content = file.read()

            pieces = content.split('.')

            messages=[
                    {"role": "system", "content":f"""You are an AI assistant.
                     Use this document: {content}"""}
                ]
    else:
        messages=[
                    {"role": "system", "content":f"""You are an AI assistant."""}
                ]

while(isActive):
    userInput = input("Choose option: \n1.Ask \n2.Stop \n3.Reset Memory ")
    
    match userInput:
        case "1":
            while isActive:
                userQues = input("Ask your question: To stop write STOP ")

                if userQues.lower() == 'stop':
                    break

                messages.append(
                    {'role':'user', 'content':userQues}
                )

                response = client.chat.completions.create(
                    model = 'gpt-4.1-mini',
                    messages= messages
                )

                reply = response.choices[0].message.content
                print(reply)
                messages.append(
                    {'role':'assistant', 'content':reply}
                )

                with open('memory_new.json', 'w') as file:
                    json.dump(messages, file, indent=4)

        case "2":
            isActive = False
            continue
        case "3":
            if os.path.exists('memory_new.json'):
                os.remove('memory_new.json')
                if os.path.exists('notes.txt'):
                    with open('notes.txt','r') as file:
                        content = file.read()
                else:
                    content =""
                messages =[
                    {"role": "system", "content":f"""You are an AI assistant.
                     Use this document: {content}"""}
                ]
                print("Memory RESET successful")
            else:
                print("Nothing to RESET")
        case _:
            print("Invalid choice") 