# Pharmacist Assistant AI

An intelligent system that reads prescription text, extracts medicine names, matches them with available stock, and suggests alternatives — built using a hybrid approach of rule-based logic and LLMs.

---

## Problem Statement

Pharmacists often face challenges such as:
- Unclear or abbreviated prescriptions (e.g., *PCM*, *IBU*)
- Difficulty matching prescribed medicines with available stock
- Lack of quick alternative suggestions

These issues can lead to delays and potential errors in dispensing medicines.

---

## Solution

Pharmacist Assistant AI automates the workflow:

1. **Extracts medicine names** from prescription text  
2. **Matches medicines** with available inventory  
3. **Suggests alternatives** if the medicine is unavailable  
4. Uses a **hybrid approach**:
   - Rule-based logic for speed and reliability  
   - LLM-based extraction for handling complex cases  

---

## Key Features

-  Intelligent medicine extraction  
-  Hybrid AI system (Rule-based + LLM fallback)  
-  Alternative medicine suggestion  
-  Robust fallback (works even without API quota)  
-  Environment-based simulation (easy / medium / hard)  
-  Built-in testing for validation  

---

##  Project Structure
.
├── env/
│ ├── environment.py
│ ├── models.py
│ └── graders.py
├── tests/
│ ├── test_env.py
│ └── test_grader.py
├── inference.py
├── README.md


---

##  How It Works

### 1. Extraction
- Extracts medicine names from prescription text  
- Uses rule-based logic for common cases  
- Falls back to LLM (`gpt-4o-mini`) for complex inputs  

### 2. Matching
- Matches extracted medicine with available medicines  

### 3. Alternative Suggestion
- Suggests substitutes when the required medicine is unavailable  

### 4. Evaluation
- Scores system performance using predefined grading logic  

---

##  Example

### Input
Prescription: "Tab PCM 500mg"
Available Medicines:
- Paracetamol 500mg
- Crocin 500mg

### Output
Extracted: PCM
Matched: Paracetamol 500mg
Score: 1.0

##  Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/vani-max/PharmacistAssistantAI.git
cd PharmacistAssistantAI
```

### Install Dependencies
pip install openai

### Set API key(optional)
export OPENAI_API_KEY="your_api_key_here"

### Run the Project
python inference.py

### Run Tests
python tests/test_env.py
python tests/test_grader.py

### Sample Output
easy score: 1.0
medium score: 1.0
hard score: 0.7

### Highlights
- Hybrid AI architecture (efficient + scalable)
- Handles real-world prescription ambiguity
- Fault-tolerant design with fallback mechanisms
- Optimized for performance and minimal API usage
  
### Future Improvements
- OCR for handwritten prescriptions
- Integration with real pharmacy databases
- Streamlit/Web UI for pharmacists
- Multi-language prescription support

### FINAL STEP

Run:
```bash
git add README.md
git commit -m "Final README for hackathon"
git push
```
