import requests

def synthesize_answer(question, context, model_name="gemma:2b"):
    """
    Synthesizes an answer using a locally running Ollama model.
    
    Parameters:
        question (str): The user's question.
        context (str): The document chunk or context to answer from.
        model_name (str): The Ollama model tag (default: 'gemma:2b').

    Returns:
        str: The generated answer or an error message.
    """
    try:
        prompt = f"Q: {question}\nContext: {context}\nA:"
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False}
        )
        answer = response.json().get("response", "").strip()
        if not answer:
            print("⚠️ Gemma returned an empty response.")
            return "No answer generated."

         # 🔍 Debug logging
        print("🧠 Prompt sent to Gemma:\n", prompt)
        print("🧠 Response from Gemma:\n", answer)
        return answer

    except Exception as e:
        print(f"❌ Error in synthesis: {str(e)}")
        return "Error"