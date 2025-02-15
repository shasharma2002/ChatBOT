import json
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from datetime import datetime
import os
import re
import parsedatetime
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()



class DocumentQABot:
    def __init__(self, test_case_dir: str = "TEST_CASE", chroma_dir: str = "chroma_index"):
        """
        Initialize the chatbot with document QA and appointment scheduling capabilities.

        Args:
            test_case_dir (str): Directory containing documents to process
            chroma_dir (str): Directory for storing the vector database
        """
        self.test_case_dir = test_case_dir
        self.chroma_dir = chroma_dir
        self.appointments_file = "appointments.json"

        # Ensure directories exist
        os.makedirs(test_case_dir, exist_ok=True)
        os.makedirs(chroma_dir, exist_ok=True)

        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.processed_files = set()

        # Load existing data
        self.load_appointments()
        self.setup_vector_store()
        self.setup_tools()

    def load_appointments(self) -> None:
        """Load existing appointments from JSON file"""
        try:
            if os.path.exists(self.appointments_file):
                with open(self.appointments_file, 'r') as f:
                    self.appointments = json.load(f)
            else:
                self.appointments = []
                with open(self.appointments_file, 'w') as f:
                    json.dump(self.appointments, f)
        except Exception as e:
            print(f"Error loading appointments: {str(e)}")
            self.appointments = []

    def save_appointments(self) -> None:
        """Save appointments to JSON file"""
        try:
            with open(self.appointments_file, 'w') as f:
                json.dump(self.appointments, f, indent=2)
        except Exception as e:
            print(f"Error saving appointments: {str(e)}")

    def setup_vector_store(self) -> None:
        """Initialize or load the vector store"""
        try:
            if not os.path.exists(self.test_case_dir) or not os.listdir(self.test_case_dir):
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.chroma_dir
                )
            else:
                self.processed_files = set()
                self.load_new_documents()
        except Exception as e:
            print(f"Error setting up vector store: {str(e)}")
            raise

    def load_new_documents(self) -> bool:
        """
        Check for and load any new documents in the TEST_CASE directory

        Returns:
            bool: True if new documents were loaded, False otherwise
        """
        new_documents = []

        try:
            for file in os.listdir(self.test_case_dir):
                file_path = os.path.join(self.test_case_dir, file)

                if file_path in self.processed_files:
                    continue

                try:
                    if file.endswith('.txt'):
                        loader = TextLoader(file_path)
                        new_documents.extend(loader.load())
                    elif file.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                        new_documents.extend(loader.load())

                    self.processed_files.add(file_path)
                    print(f"Added new document: {file}")

                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")

            if new_documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(new_documents)

                if not hasattr(self, 'vectorstore'):
                    self.vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=self.embeddings,
                        persist_directory=self.chroma_dir
                    )
                else:
                    self.vectorstore.add_documents(splits)

                print(f"Processed {len(splits)} new document chunks")
                return True

        except Exception as e:
            print(f"Error in load_new_documents: {str(e)}")

        return False

    def setup_tools(self) -> None:
        """Set up the available tools for the chatbot"""
        self.tools = [
            Tool(
                name="Schedule Appointment",
                func=self.handle_scheduling_flow,
                description="Schedule an appointment and collect user information"
            ),
            Tool(
                name="View Appointments",
                func=self.view_appointments,
                description="View all scheduled appointments"
            ),
            Tool(
                name="Cancel Appointment",
                func=self.cancel_appointment,
                description="Cancel an existing appointment"
            ),
            Tool(
                name="Reload Documents",
                func=self.load_new_documents,
                description="Check for and load any new documents"
            )
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory
        )

    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(pattern, email))

    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        pattern = r'^\+?1?\d{9,15}$'
        return bool(re.match(pattern, phone))

    def parse_date(self, date_string: str) -> Optional[str]:
        """Parse date string to YYYY-MM-DD format"""
        cal = parsedatetime.Calendar()
        time_struct, parse_status = cal.parse(date_string)
        if parse_status:
            return datetime(*time_struct[:3]).strftime('%Y-%m-%d')
        return None

    def collect_contact_info(self) -> Dict[str, str]:
        """Collect and validate user contact information"""
        contact_info = {}

        name = input("Please enter your name: ").strip()
        while not name:
            print("Name cannot be empty.")
            name = input("Please enter your name: ").strip()
        contact_info['name'] = name

        while True:
            email = input("Please enter your email: ").strip()
            if self.validate_email(email):
                contact_info['email'] = email
                break
            print("Invalid email format. Please try again.")

        while True:
            phone = input("Please enter your phone number: ").strip()
            if self.validate_phone(phone):
                contact_info['phone'] = phone
                break
            print("Invalid phone format. Please try again (9-15 digits).")

        return contact_info

    def view_appointments(self, query: Optional[str] = None) -> str:
        """View all scheduled appointments"""
        if not self.appointments:
            return "No appointments scheduled."

        response = "\nScheduled Appointments:\n"
        for idx, appt in enumerate(self.appointments, 1):
            response += f"\n{idx}. Date: {appt['date']}\n"
            response += f"   Name: {appt['name']}\n"
            response += f"   Email: {appt['email']}\n"
            response += f"   Phone: {appt['phone']}\n"

        return response

    def cancel_appointment(self, query: str) -> str:
        """Cancel an existing appointment"""
        if not self.appointments:
            return "No appointments to cancel."

        print("\nCurrent appointments:")
        for idx, appt in enumerate(self.appointments, 1):
            print(f"{idx}. Date: {appt['date']} - {appt['name']}")

        while True:
            try:
                choice = int(input("\nEnter the number of the appointment to cancel (0 to abort): "))
                if choice == 0:
                    return "Cancellation aborted."
                if 1 <= choice <= len(self.appointments):
                    cancelled = self.appointments.pop(choice - 1)
                    self.save_appointments()
                    return f"Cancelled appointment for {cancelled['name']} on {cancelled['date']}"
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def handle_scheduling_flow(self, query: str) -> str:
        """Handle the complete appointment scheduling flow"""
        print("\nLet's collect your contact information first.")
        contact_info = self.collect_contact_info()

        while True:
            date_input = input(
                "\nWhen would you like to schedule the call? (e.g., 'next Monday', 'tomorrow'): ").strip()
            date = self.parse_date(date_input)
            if date:
                # Check for existing appointments on the same date
                for appt in self.appointments:
                    if appt['date'] == date:
                        print(f"Warning: There is already an appointment scheduled for {date}")
                        if input("Would you like to choose a different date? (y/n): ").lower() == 'y':
                            continue
                break
            print("Could not understand the date. Please try again.")

        appointment = {
            "date": date,
            "name": contact_info['name'],
            "email": contact_info['email'],
            "phone": contact_info['phone']
        }

        self.appointments.append(appointment)
        self.save_appointments()

        response = (
            f"\nAppointment scheduled successfully!\n"
            f"Name: {appointment['name']}\n"
            f"Email: {appointment['email']}\n"
            f"Phone: {appointment['phone']}\n"
            f"Date: {appointment['date']}"
        )

        return response

    def process_query(self, query: str) -> str:
        """Process user queries and route to appropriate handler"""
        try:
            # Check for document reload request
            if any(keyword in query.lower() for keyword in
                   ['reload documents', 'refresh documents', 'update documents']):
                if self.load_new_documents():
                    return "Successfully loaded new documents."
                return "No new documents found."

            # Check for appointment cancellation
            if any(keyword in query.lower() for keyword in ['cancel appointment', 'cancel call']):
                return self.cancel_appointment(query)

            # Check for viewing appointments
            if any(keyword in query.lower() for keyword in
                   ['show appointments', 'view appointments', 'list appointments', 'scheduled calls', 'view calls']):
                return self.view_appointments()

            # Check for scheduling
            if any(keyword in query.lower() for keyword in
                   ['schedule', 'appointment', 'book', 'call me', 'contact me']):
                return self.agent.run(query)

            # Document QA
            return self.qa_chain({"question": query})["answer"]

        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return f"An error occurred: {str(e)}"


def setup_environment():
    """Create necessary directories and files"""
    os.makedirs("TEST_CASE", exist_ok=True)
    os.makedirs("chroma_index", exist_ok=True)



def main():
    # Setup environment
    setup_environment()

    # Load environment variables
    import dotenv
    dotenv.load_dotenv()

    if not os.getenv('OPENAI_API_KEY'):
        print("Please set your OPENAI_API_KEY in the .env file")
        return

    # Initialize the bot
    bot = DocumentQABot()

    print("\nAvailable commands:")
    print("- Schedule a call: 'schedule a call' or 'book appointment'")
    print("- View appointments: 'show appointments' or 'view scheduled calls'")
    print("- Cancel appointment: 'cancel appointment' or 'cancel call'")
    print("- Reload documents: 'reload documents' or 'refresh documents'")
    print("- Ask questions about documents in TEST_CASE directory")

    while True:
        query = input("\nYou: ").strip()
        if query.lower() == 'quit':
            break

        response = bot.process_query(query)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()