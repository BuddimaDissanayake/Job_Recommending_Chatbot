from rag.rag_chain import create_rag_chain

def main():
    rag_chain = create_rag_chain()

    print("👩‍💻 Welcome to the Job Recommender Chatbot!")
    print("Ask about job roles, traits, skills, etc.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("🧑 You: ")
        if user_input.lower() == "exit":
            break
        response = rag_chain.run(user_input)
        print("\n🤖 Bot:", response, "\n")

if __name__ == "__main__":
    main()
