
import hashlib
import os
import re
import json
import tempfile
import uuid


import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title='QuizRAG Test')
st.title('QuizRAG')

if 'question_data' not in st.session_state:
    st.session_state.question_data = None

if 'asked_questions' not in st.session_state:
    st.session_state.asked_questions = []

if 'questions_attempted' not in st.session_state:
    st.session_state.questions_attempted = 0

if 'correct_answers' not in st.session_state:
    st.session_state.correct_answers = 0

if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []

if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if 'submitted_answer' not in st.session_state:
    st.session_state.submitted_answer = None

if 'quiz_attempted' not in st.session_state:
    st.session_state.quiz_attempted = 0

if 'quiz_correct' not in st.session_state:
    st.session_state.quiz_correct = 0

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    st.error('GEMINI_API_KEY was not found in your .env file.')
    st.stop()

llm = ChatGoogleGenerativeAI(
    model='gemini-3.6-flash',
    google_api_key=api_key
)

uploaded_file = st.file_uploader('Upload a PDF', type=['pdf'])

num_questions = st.selectbox(
    'Number of questions',
    options=[5, 10, 20],
    index=1
)


if uploaded_file is not None:
    try:
        @st.cache_data
        def process_pdf(pdf_bytes):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix='.pdf'
            ) as temp_file:
                temp_file.write(pdf_bytes)
                temp_path = temp_file.name

            try:
                documents = PyPDFLoader(temp_path).load()
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            cleaned_documents = []

            for doc in documents:
                cleaned_text = re.sub(
                    r'(?<=[a-z])(?=[A-Z])',
                    ' ',
                    doc.page_content
                )
                cleaned_text = re.sub(
                    r'\s+', ' ', cleaned_text
                ).strip()                

                if cleaned_text:
                    cleaned_documents.append(
                        Document(
                            page_content=cleaned_text,
                            metadata=doc.metadata.copy()
                        )
                    )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(cleaned_documents)

            return [
                {
                    'page_content': chunk.page_content,
                    'metadata': chunk.metadata
                }
                for chunk in chunks
            ]

        cached_chunks = process_pdf(uploaded_file.getvalue())

        chunks = [
            Document(
                page_content=chunk['page_content'],
                metadata=chunk['metadata']
            )
            for chunk in cached_chunks
        ]

        st.success(f'Created {len(chunks)} chunks.')

        @st.cache_resource
        def get_embeddings():
            return HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2'
            )

        embeddings = get_embeddings()

        @st.cache_resource
        def create_vector_store(_chunks, _embeddings, file_id):
            return Chroma.from_documents(
                documents=_chunks,
                embedding=_embeddings,
                collection_name=f'quizrag_{file_id}',
                persist_directory=None
            )

        vector_store = create_vector_store(
            chunks,
            embeddings,
            hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        )

        if st.button('Generate Question'):
            with st.spinner(
                f'Generating {num_questions} questions from your notes...'
            ):
                retrieved_docs = vector_store.similarity_search(
                    'Important concepts, definitions, explanations, '
                    'and key topics in the study material',
                    k=min(num_questions * 2, 20)
                )

                context = '\n\n'.join(
                    doc.page_content for doc in retrieved_docs
                )

                prompt = ChatPromptTemplate.from_template("""
Generate exactly {num_questions} multiple-choice questions
based strictly on the study material.

Return only a valid JSON object with this structure:
{{
    "questions": [
        {{
            "question": "Question text",
            "options": {{
                "A": "First option",
                "B": "Second option",
                "C": "Third option",
                "D": "Fourth option"
            }},
            "correct_answer": "A",
            "topic": "Topic",
            "explanation": "Explanation",
            "supporting_evidence": "Exact quotation from the material"
        }}
    ]
}}

Requirements:
- Generate exactly {num_questions} distinct questions.
- Provide four distinct options per question.
- Each question must have exactly one correct answer.
- The correct answer must be supported by the study material.
- Incorrect options must be plausible but incorrect according to the material.
- Supporting evidence must be an exact quotation from the material.
- The explanation must agree with the correct answer.
- Cover different concepts and topics where possible.
- Return valid JSON only, without Markdown fences.

Study material:
{context}
""")

                response = llm.invoke(
                    prompt.format_messages(
                        context=context,
                        num_questions=num_questions
                    )
                )

                content = response.content

                if isinstance(content, list):
                    raw_output = '\n'.join(
                        item['text']
                        for item in content
                        if isinstance(item, dict)
                        and item.get('type') == 'text'
                    )
                else:
                    raw_output = content

                raw_output = raw_output.strip()

                if raw_output.startswith('```json'):
                    raw_output = raw_output[7:]
                elif raw_output.startswith('```'):
                    raw_output = raw_output[3:]

                if raw_output.endswith('```'):
                    raw_output = raw_output[:-3]

                result = json.loads(raw_output.strip())

                if not isinstance(result, dict):
                    raise ValueError(
                        'Gemini returned an invalid quiz format.'
                    )

                generated_questions = result.get('questions')

                if not isinstance(generated_questions, list):
                    raise ValueError(
                        'Gemini did not return a questions list.'
                    )

                valid_questions = []
                seen_questions = set(st.session_state.asked_questions)

                required_fields = {
                    'question',
                    'options',
                    'correct_answer',
                    'topic',
                    'explanation',
                    'supporting_evidence'
                }

                for question in generated_questions:
                    if not isinstance(question, dict):
                        continue

                    if not required_fields.issubset(question.keys()):
                        continue

                    if not all(
                        isinstance(question[key], str)
                        and question[key].strip()
                        for key in [
                            'question',
                            'topic',
                            'explanation',
                            'supporting_evidence'
                        ]
                    ):
                        continue

                    options = question['options']

                    if not isinstance(options, dict):
                        continue

                    if set(options.keys()) != {'A', 'B', 'C', 'D'}:
                        continue

                    if not all(
                        isinstance(options[key], str)
                        and options[key].strip()
                        for key in ['A', 'B', 'C', 'D']
                    ):
                        continue

                    if len({
                        options[key].strip().casefold()
                        for key in ['A', 'B', 'C', 'D']
                    }) != 4:
                        continue

                    if question['correct_answer'] not in {
                        'A', 'B', 'C', 'D'
                    }:
                        continue

                    normalized_question = ' '.join(
                        question['question'].lower().split()
                    )

                    if normalized_question in seen_questions:
                        continue

                    seen_questions.add(normalized_question)
                    valid_questions.append(question)

                    if len(valid_questions) == num_questions:
                        break

                if not valid_questions:
                    raise ValueError(
                        'No valid, non-duplicate questions were generated. '
                        'Try generating the quiz again.'
                    )

                st.session_state.quiz_questions = valid_questions
                st.session_state.current_question_index = 0
                st.session_state.quiz_attempted = 0
                st.session_state.quiz_correct = 0
                st.session_state.submitted = False
                st.session_state.submitted_answer = None

                st.session_state.asked_questions.extend(
                    ' '.join(q['question'].lower().split())
                    for q in valid_questions
                )

                if 'selected_answer' in st.session_state:
                    del st.session_state['selected_answer']

                st.session_state.question_data = None

                if len(valid_questions) < num_questions:
                    st.warning(
                        f'You requested {num_questions} questions, '
                        f'but only {len(valid_questions)} valid, '
                        'non-duplicate questions were generated.'
                    )

                st.success(
                    f'Generated {len(valid_questions)} questions.'
                )

        if st.session_state.quiz_questions:
            total_questions = len(st.session_state.quiz_questions)
            question_index = st.session_state.current_question_index

            if question_index < total_questions:
                question_data = st.session_state.quiz_questions[
                    question_index
                ]

                st.progress(
                    (question_index + 1) / total_questions
                )

                st.caption(
                    f'Question {question_index + 1} of {total_questions}'
                )

                st.subheader(question_data['question'])

                selected_answer = st.radio(
                    'Choose your answer:',
                    options=['A', 'B', 'C', 'D'],
                    format_func=lambda key: (
                        f'{key}: {question_data["options"][key]}'
                    ),
                    key='selected_answer',
                    disabled=st.session_state.submitted
                )

                if (
                    st.button('Submit Answer')
                    and not st.session_state.submitted
                ):
                    st.session_state.submitted = True
                    st.session_state.submitted_answer = selected_answer
                    st.session_state.quiz_attempted += 1
                    st.session_state.questions_attempted += 1

                    if selected_answer == question_data['correct_answer']:
                        st.session_state.quiz_correct += 1
                        st.session_state.correct_answers += 1

                    st.rerun()

                if st.session_state.submitted:
                    submitted_answer = st.session_state.submitted_answer

                    if submitted_answer == question_data['correct_answer']:
                        st.success('Correct answer!')
                    else:
                        correct_answer = question_data['correct_answer']

                        st.error(
                            f'Incorrect. The correct answer is '
                            f'{correct_answer}: '
                            f'{question_data["options"][correct_answer]}'
                        )

                    st.write(
                        '**Explanation:**',
                        question_data['explanation']
                    )
                    st.write('**Topic:**', question_data['topic'])
                    st.write(
                        '**Supporting evidence:**',
                        question_data['supporting_evidence']
                    )

                    if st.button(
                        'Next Question'
                        if question_index < total_questions - 1
                        else 'Finish Quiz'
                    ):
                        st.session_state.current_question_index += 1
                        st.session_state.submitted = False
                        st.session_state.submitted_answer = None

                        if 'selected_answer' in st.session_state:
                            del st.session_state['selected_answer']

                        st.rerun()

            else:
                st.subheader('Quiz Results')

                st.metric(
                    'Score',
                    f'{st.session_state.quiz_correct}/'
                    f'{total_questions}'
                )

                accuracy = (
                    st.session_state.quiz_correct
                    / total_questions
                ) * 100

                st.metric('Accuracy', f'{accuracy:.1f}%')

                preparation_rating = (
                    st.session_state.quiz_correct
                    / total_questions
                ) * 10

                st.metric(
                    'Quiz Rating',
                    f'{preparation_rating:.1f}/10'
                )

                st.write(
                    f'Questions attempted: '
                    f'{st.session_state.quiz_attempted}'
                )

                if st.button('Generate Another Quiz'):
                    st.session_state.quiz_questions = []
                    st.session_state.current_question_index = 0
                    st.session_state.submitted = False
                    st.session_state.submitted_answer = None

                    if 'selected_answer' in st.session_state:
                        del st.session_state['selected_answer']

                    st.rerun()

            st.divider()

            st.write(
                f'Total questions attempted across quizzes: '
                f'{st.session_state.questions_attempted} '
                f'| Total correct answers: '
                f'{st.session_state.correct_answers}'
            )

            if st.session_state.questions_attempted > 0:
                overall_rating = (
                    st.session_state.correct_answers
                    / st.session_state.questions_attempted
                ) * 10

                st.metric(
                    'Overall Preparation Rating',
                    f'{overall_rating:.1f}/10'
                )       

    except Exception as error:
        st.error(f'An error occurred: {error}')
