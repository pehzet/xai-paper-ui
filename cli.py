from chatbot import XAIChatbot

bot = XAIChatbot()

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    bot_response = bot.chat(user_input)
    print(f"Bot: {bot_response}")

