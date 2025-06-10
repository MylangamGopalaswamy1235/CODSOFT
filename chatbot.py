def chatbot():
    print("Chatbot: Hello! I'm your Python Chatbot. Ask me anything. Type 'bye' to exit.")

    while True:
        user_input = input("You: ").lower()

        # GREETINGS
        if user_input in ["hi", "hello", "hey"]:
            print("Chatbot: Hello there! How can I assist you today?")
        elif "good morning" in user_input:
            print("Chatbot: Good morning! Hope you have a productive day.")
        elif "good night" in user_input:
            print("Chatbot: Sweet dreams. Good night!")
        elif "good afternoon" in user_input:
            print("Chatbot: Good afternoon! What can I do for you?")
        
        # PERSONAL QUESTIONS
        elif "your name" in user_input:
            print("Chatbot: I'm ChatPy, your chatbot assistant.")
        elif "how are you" in user_input:
            print("Chatbot: I'm doing great! Thanks for asking.")
        elif "who created you" in user_input:
            print("Chatbot: I was created by a Python enthusiast.")
        elif "are you real" in user_input:
            print("Chatbot: I exist in the code world!")
        
        # CONVERSATIONAL
        elif "what can you do" in user_input:
            print("Chatbot: I can answer simple questions, chat with you, and help you learn programming!")
        elif "thank you" in user_input or "thanks" in user_input:
            print("Chatbot: You're welcome! 😊")
        elif "i love you" in user_input:
            print("Chatbot: Aww, that's sweet! I'm just code though 😅")
        elif "do you love me" in user_input:
            print("Chatbot: I like chatting with you, if that counts!")
        elif "joke" in user_input:
            print("Chatbot: Why don’t scientists trust atoms? Because they make up everything!")
        
        # TIME & DATE
        elif "time" in user_input:
            from datetime import datetime
            print("Chatbot: Current time is", datetime.now().strftime("%H:%M:%S"))
        elif "date" in user_input:
            from datetime import datetime
            print("Chatbot: Today's date is", datetime.now().strftime("%Y-%m-%d"))

        # GENERAL KNOWLEDGE
        elif "capital of india" in user_input:
            print("Chatbot: The capital of India is New Delhi.")
        elif "largest ocean" in user_input:
            print("Chatbot: The Pacific Ocean is the largest.")
        elif "tallest mountain" in user_input:
            print("Chatbot: Mount Everest is the tallest mountain on Earth.")
        elif "who is prime minister of india" in user_input:
            print("Chatbot: As of 2025, it's Narendra Modi.")
        elif "who is president of usa" in user_input:
            print("Chatbot: As of 2025, it's Joe Biden.")
        elif "full form of ai" in user_input:
            print("Chatbot: AI stands for Artificial Intelligence.")
        elif "full form of cpu" in user_input:
            print("Chatbot: CPU stands for Central Processing Unit.")
        
        # EDUCATIONAL
        elif "what is python" in user_input:
            print("Chatbot: Python is a powerful, high-level programming language.")
        elif "what is ai" in user_input:
            print("Chatbot: AI is a field of computer science that enables machines to mimic human intelligence.")
        elif "what is machine learning" in user_input:
            print("Chatbot: Machine learning is a subset of AI that allows systems to learn from data.")
        elif "what is chatbot" in user_input:
            print("Chatbot: A chatbot is a program designed to simulate conversation with humans.")
        elif "what is debugging" in user_input:
            print("Chatbot: Debugging is the process of finding and fixing errors in code.")
        
        # ENTERTAINMENT
        elif "movie" in user_input:
            print("Chatbot: I like sci-fi movies like Interstellar. What about you?")
        elif "game" in user_input:
            print("Chatbot: Chess, Ludo, and Minecraft are some cool games!")
        elif "music" in user_input:
            print("Chatbot: Music is soothing! Do you have a favorite song?")
        
        # EXIT
        elif user_input == "bye":
            print("Chatbot: Goodbye! Have a nice day.")
            break
        
        # DEFAULT RESPONSE
        else:
            print("Chatbot: Sorry, I don't understand that. Try asking something else.")

# Run the chatbot
chatbot()