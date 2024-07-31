def chatbot_response(user_input):
    user_input = user_input.lower()
    
    # Greetings
    if 'hello' in user_input or 'hi' in user_input:
        return "Hello! How can I help you today?"
    # Farewells
    elif 'bye' in user_input or 'goodbye' in user_input:
        return "Goodbye! Have a great day!"
    # Asking for name
    elif 'your name' in user_input:
        return "I'm a chatbot created to assist you with your queries."
    # Asking for help
    elif 'help' in user_input or 'assist' in user_input:
        return "Sure, I'm here to help. What do you need assistance with?"
    # Weather query
    elif 'weather' in user_input:
        return "I can't provide real-time weather updates, but you can check your local weather website or app."
    # Default response
    else:
        return "I'm not sure how to respond to that. Can you please rephrase your question?"

# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Chatbot: Goodbye! Have a great day!")
        break
    print("Chatbot:", chatbot_response(user_input))
