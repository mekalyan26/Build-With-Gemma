import os
import tempfile
from pathlib import Path
from typing import List
import psycopg2
from psycopg2 import pool
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from psycopg2.extras import Json

load_dotenv('database_connect.env')
db_url = os.getenv("DATABASE_URL")

connection_pool = pool.SimpleConnectionPool(
    1,
    10,
    dsn=db_url,
)
load_dotenv('gemma.env')
api_key = os.getenv('GEMMA_API_KEY')
model_name = 'spur-gemma4'

if not api_key:
    st.error('GEMMA_API_KEY is not set in gemma.env. Please add it and reload the app.')
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://ai.spuric.com/v1"
)


def load_text_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f'Patient history file not found: {path}')
    return file_path.read_text(encoding='utf-8')


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or '.txt'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def build_summary_prompt(image_ref: str, history_text: str, context: str, prompt: str) -> str:
    return (
        'You are a medical assistant. Review the patient history and image details, then provide a concise summary, '
        'a detailed description in simple statistics, and a clear explanation for the doctor. '
        'If the doctor might not understand the response, include a follow-up question to request additional information.\n\n'
        f'Image reference: {image_ref}\n\n'
        f'Patient history:\n{history_text}\n\n'
        f'Context:\n{context}\n\n'
        f'Prompt:\n{prompt}\n\n'
        'Output format:\n'
        '1. Summary\n'
        '2. Detailed description\n'
        '3. Simple statistics (short bullet list)\n'
        '4. Follow-up question for the doctor if clarification is needed\n'
    )
def insert_patient(
    name,
    age,
    additional_info=None
):
    connection = connection_pool.getconn()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO patients
            (
                name,
                age,
                additional_info
            )
            VALUES (%s, %s, %s)
            RETURNING patient_id;
            """,
            (
                name,
                age,
                Json(additional_info)
            )
        )

        patient_id = cursor.fetchone()[0]

        if patient_id is None:
            raise RuntimeError('Failed to create patient record.')

        connection.commit()

        return patient_id

    finally:
        cursor.close()
        connection_pool.putconn(connection)
def insert_ai_response(
    patient_id,
    prompt,
    response,
    response_remarks: str = 'Not rated',
) -> int:

    connection = connection_pool.getconn()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO ai_responses
            (
                patient_id,
                prompt,
                response,
                response_remarks
            )
            VALUES (%s,%s,%s,%s)
            RETURNING response_id
            """,
            (
                patient_id,
                prompt,
                response,
                response_remarks,
            )
        )

        ai_response_id = cursor.fetchone()[0]
        connection.commit()

        return ai_response_id

    finally:
        cursor.close()
        connection_pool.putconn(connection)

def update_ai_response_feedback(ai_response_id: int, feedback: str) -> None:
    connection = connection_pool.getconn()
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            UPDATE ai_responses
            SET response_remarks = %s
            WHERE response_id = %s
            """,
            (feedback, ai_response_id),
        )
        connection.commit()
    finally:
        cursor.close()
        connection_pool.putconn(connection)


def fetch_patient_response_stats() -> list:
    connection = connection_pool.getconn()
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            select p.patient_id,p.additional_info->>'context' as additional_context, p.name,p.age,a.response_remarks from patients p  join ai_responses a on p.patient_id =a.patient_id 
and a.response_remarks is not NULL
            """
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        connection_pool.putconn(connection)


def summarize_patient_information(
    patient_id: str,
    image_path: str,
    patient_history_text: str,
    context: str,
    prompt: str,
    summary_feedback: str = 'Not rated',
) -> str:

    combined_prompt = build_summary_prompt(
        image_path,
        patient_history_text,
        context,
        prompt
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": combined_prompt
                }
            ],
            max_tokens=800,
        )

    except OpenAIError as exc:
        raise RuntimeError(f'Spur API error: {exc}') from exc


    summarized_response = response.choices[0].message.content

    # Save AI response with optional user review feedback
    ai_response_id = insert_ai_response(
        patient_id=patient_id,
        prompt=prompt,
        response=summarized_response,
        response_remarks=summary_feedback,
    )

    st.session_state.latest_ai_response_id = ai_response_id
    st.session_state.summary_feedback = summary_feedback

    return summarized_response


def ask_follow_up(conversation: List[dict], question: str) -> str:
    messages = [
        {'role': 'system', 'content': 'You are a conversational medical assistant. Answer follow-up questions clearly and politely.'}
    ]
    messages.extend(conversation)
    messages.append({'role': 'user', 'content': question})
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=400,
        )
    except OpenAIError as exc:
        raise RuntimeError(f'Spur API error: {exc}') from exc

    return response.choices[0].message.content


def initialize_session_state() -> None:
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'summary' not in st.session_state:
        st.session_state.summary = ''
    if 'summary_feedback' not in st.session_state:
        st.session_state.summary_feedback = ''
    if 'follow_up_history' not in st.session_state:
        st.session_state.follow_up_history = []


def main() -> None:
    st.set_page_config(page_title='CareFlow 360 Medical Assistant', layout='wide')
    initialize_session_state()

    st.markdown(
        """
        <style>
            .main-header {
                background: linear-gradient(135deg, #0f4c81 0%, #2a799d 100%);
                color: white;
                padding: 30px 32px;
                border-radius: 22px;
                box-shadow: 0 24px 60px rgba(15, 76, 129, 0.18);
                margin-bottom: 20px;
            }
            .main-header h1 {
                margin-bottom: 8px;
            }
            .section-card {
                background: white;
                border-radius: 22px;
                padding: 24px;
                box-shadow: 0 18px 40px rgba(39, 92, 137, 0.12);
                margin-bottom: 24px;
            }
            .section-card h2 {
                margin-bottom: 14px;
                color: #0f4c81;
            }
            .small-note {
                color: #4a6b86;
                font-size: 0.95rem;
                line-height: 1.6;
            }
            .stButton>button {
                border-radius: 14px;
                background-color: #0f4c81;
                color: white;
                padding: 0.85rem 1.4rem;
            }
            .stButton>button:hover {
                background-color: #165988;
            }
            .stTextInput>div>div>input,
            .stTextArea>div>div>textarea,
            .stNumberInput>div>div>input {
                border-radius: 14px;
            }
            .stFileUploader>div>div {
                border-radius: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='main-header'><h1>CareFlow 360 Medical Assistant</h1><p class='small-note'>Generate clinical summaries, store AI responses, and manage follow-up questions in a polished workspace.</p></div>",
        unsafe_allow_html=True,
    )

    try:
        stats = fetch_patient_response_stats()
    except Exception as exc:
        stats = []
        st.error(f"Unable to load patient stats: {exc}")

    if stats:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader('Patient response dashboard')
        st.markdown('Quick view of patients, age, and AI response status.')
        st.table([
            {
                'Patient ID': row[0],
                'Addition Remarks':row[1],
                'Name': row[2],
                'Age': row[3],
                'Response remarks': row[4],
            }
            for row in stats
        ])
        st.markdown('</div>', unsafe_allow_html=True)

    with st.sidebar:
        #st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown('**Add Patient Details**')
        patient_name = st.text_input('Patient name')
        patient_age = st.number_input('Patient age', min_value=0, max_value=130, step=1)
        st.markdown('---')
        st.subheader('Uploads')
        image_file = st.file_uploader('Upload patient image', type=['png', 'jpg', 'jpeg', 'bmp'])
        image_path_input = st.text_input('Or enter local image path')
        history_file = st.file_uploader('Upload patient history text file', type=['txt', 'md'])
        history_path_input = st.text_input('Or enter local patient history path')
        st.markdown('---')
        st.subheader('Prompt settings')
        context = st.text_area('Additional context', value='', height=120)
        prompt = st.text_area(
            'Prompt',
            value='Summarize the patient case and describe the findings in simple statistics.',
            height=140,
        )
        st.markdown('<p class="small-note">Use the prompt field to guide the AI response tone, detail, and structure.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        summary_button = st.button('Generate summary')

    summary_column, chat_column = st.columns([2, 1])

    with summary_column:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader('Summary workspace')
        st.markdown('<p class="small-note">Review generated summaries and patient context here. Summary results are saved automatically.</p>', unsafe_allow_html=True)

        if summary_button:
            if not image_file and not image_path_input:
                st.error('Please provide an image file or an image path.')
            elif not history_file and not history_path_input:
                st.error('Please provide a patient history file or a history path.')
            else:
                try:
                    if image_file:
                        image_path = save_uploaded_file(image_file)
                    else:
                        image_path = image_path_input
                        if not Path(image_path).is_file():
                            raise FileNotFoundError(f'Image file not found: {image_path}')

                    if history_file:
                        patient_history_text = history_file.getvalue().decode('utf-8')
                    else:
                        patient_history_text = load_text_file(history_path_input)

                    patient_id = insert_patient(
                        name=patient_name or 'Unknown',
                        age=patient_age,
                        additional_info={'context': context},
                    )

                    summary = summarize_patient_information(
                        patient_id,
                        image_path,
                        patient_history_text,
                        context,
                        prompt,
                        'Not rated',
                    )
                    st.session_state.summary_feedback = 'Not rated'
                    st.session_state.summary = summary
                    st.session_state.conversation = [
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': summary},
                    ]

                    st.success('Summary generated successfully.')
                    st.markdown(f"<div class='section-card'><strong>Patient:</strong> {patient_name or 'Unknown'}<br><strong>Age:</strong> {patient_age}</div>", unsafe_allow_html=True)
                    st.write(summary)
                    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
                    if feedback_col1.button('Accept suggestion', key='feedback_accept'):
                        st.session_state.summary_feedback = 'Accepted'
                        if st.session_state.latest_ai_response_id:
                            update_ai_response_feedback(st.session_state.latest_ai_response_id, 'Accepted')
                    if feedback_col2.button('Needs update', key='feedback_needs_update'):
                        st.session_state.summary_feedback = 'Needs update'
                        if st.session_state.latest_ai_response_id:
                            update_ai_response_feedback(st.session_state.latest_ai_response_id, 'Needs update')
                    if feedback_col3.button('Deny suggestion', key='feedback_deny'):
                        st.session_state.summary_feedback = 'Denied'
                        if st.session_state.latest_ai_response_id:
                            update_ai_response_feedback(st.session_state.latest_ai_response_id, 'Denied')
                    if st.session_state.summary_feedback != 'Not rated':
                        st.markdown(f"**Feedback:** {st.session_state.summary_feedback}")

                    if image_file:
                        st.image(image_file, caption='Uploaded patient image', use_column_width=True)
                    elif Path(image_path_input).is_file():
                        st.image(image_path_input, caption='Selected patient image', use_column_width=True)
                except (FileNotFoundError, RuntimeError) as exc:
                    st.error(str(exc))
        elif st.session_state.summary:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader('Last generated summary')
            st.write(st.session_state.summary)

            feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
            if feedback_col1.button('Accept suggestion', key='feedback_accept_stored'):
                st.session_state.summary_feedback = 'Accepted'
                if st.session_state.latest_ai_response_id:
                    update_ai_response_feedback(st.session_state.latest_ai_response_id, 'Accepted')
            if feedback_col2.button('Needs update', key='feedback_needs_update_stored'):
                st.session_state.summary_feedback = 'Needs update'
                if st.session_state.latest_ai_response_id:
                    update_ai_response_feedback(st.session_state.latest_ai_response_id, 'Needs update')
            if feedback_col3.button('Deny suggestion', key='feedback_deny_stored'):
                st.session_state.summary_feedback = 'Denied'
                if st.session_state.latest_ai_response_id:
                    update_ai_response_feedback(st.session_state.latest_ai_response_id, 'Denied')

            if st.session_state.summary_feedback != 'Not rated':
                st.markdown(f"**Feedback:** {st.session_state.summary_feedback}")

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with chat_column:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader('Follow-up chat')
        st.markdown('<p class="small-note">Ask follow-up questions after a summary has been generated.</p>', unsafe_allow_html=True)

        follow_up_question = st.text_input('Ask a follow-up question for the doctor')
        follow_up_button = st.button('Ask follow-up')

        if follow_up_button:
            if not follow_up_question:
                st.error('Please enter a follow-up question.')
            elif not st.session_state.conversation:
                st.error('Generate a summary first so the chat has context.')
            else:
                try:
                    answer = ask_follow_up(st.session_state.conversation, follow_up_question)
                    st.session_state.follow_up_history.append({'question': follow_up_question, 'answer': answer})
                    st.session_state.conversation.append({'role': 'user', 'content': follow_up_question})
                    st.session_state.conversation.append({'role': 'assistant', 'content': answer})
                    st.success('Follow-up answer generated.')
                except RuntimeError as exc:
                    st.error(str(exc))

        if st.session_state.follow_up_history:
            for item in st.session_state.follow_up_history[::-1]:
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer']}")
                st.markdown('---')

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()