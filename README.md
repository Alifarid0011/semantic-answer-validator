Semantic Answer Validator

The Semantic Answer Validator is a lightweight API that checks whether a student's answer is semantically aligned with the expected answers.
It uses modern semantic similarity models to go beyond exact keyword matching, supporting both English and Persian inputs.

🚀 Features

✅ Semantic validation instead of strict keyword matching.

✅ Multi-language support (English + Persian).

✅ Handles both positive and negative statements correctly.

✅ Supports optional keywords for stricter validation.

✅ RESTful API with FastAPI.

✅ Ready-to-use test cases for evaluation.

📦 Installation
# Clone repository
git clone https://github.com/AliFarid0011/semantic-answer-validator.git
cd semantic-answer-validator

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Linux / Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

▶️ Run the API
uvicorn app:app --reload --port 8020

🔗 Example Request

```
POST http://127.0.0.1:8020/check_answer
Content-Type: application/json
{
  "student_answer": "Artificial intelligence improves efficiency in many industries.",
  "accepted_answers": [
    "AI enhances productivity across different sectors.",
    "Artificial intelligence helps automate tasks and increase efficiency."
  ]
}
---------
Example Response
{
  "is_correct": true,
  "similarity_score": 0.86
}
```


📖 Use Cases
🏫 Education → Evaluate free-text answers in quizzes/exams.
🌍 Multilingual applications → Supports semantic similarity in Persian & English.
📊 Surveys / Feedback analysis → Classify open-ended responses by meaning.
🛠 Tech Stack
Python 3.10+
FastAPI (API framework)
Sentence-Transformers (semantic embeddings)
Uvicorn (ASGI server)
📌 Roadmap
 Add support for more languages.
 Improve contradiction detection.
 Add optional scoring (0–100).
 Dockerize for production use.
📜 License

This project is licensed under the MIT License.