def general_agent(client, step):
    response = client.chat.completions.create(
        model = 'gpt-4.1-mini',
        messages = [
            {
                'role':'system',
                'content': 'You are an AI Assistant'
            },
            {
                'role':'user',
                'content': step
            }
        ]
    )
    reply = response.choices[0].message.content
    return reply