from bot import DocumentQABot  # Import the class from bot.py
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Initialize the bot
    bot = DocumentQABot()  # Instantiate the DocumentQABot class

    print("\nChatbot initialized. Type 'quit' to exit.")
    print("\nAvailable commands:")
    print("- Schedule a call: 'schedule a call' or 'book appointment'")
    print("- View appointments: 'show appointments' or 'view scheduled calls'")
    print("- Cancel appointment: 'cancel appointment' or 'cancel call'")
    print("- Reload documents: 'reload documents' or 'refresh documents'")
    print("- Ask questions about documents in TEST_CASE directory")

    print("\nChatbot initialized. Type 'quit' to exit.")
    while True:
        query = input("\nYou: ").strip()  # Get input from the user
        if query.lower() == 'quit':
            break

        # Process the query and get the response from the bot
        response = bot.process_query(query)  # Use the process_query method of DocumentQABot
        print(f"Bot: {response}")  # Display the bot's response

if __name__ == "__main__":
    main()
