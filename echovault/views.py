# views_cbv.py (Class-Based Views rendering HTML templates)
from django.shortcuts import render, redirect
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from django.contrib.auth import authenticate, login as auth_login
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import UserSerializer, VideoSerializer
from .models import CustomUser, Video
import tempfile, subprocess, whisper, os
from pydub import AudioSegment
from django.conf import settings
import pytesseract, cv2
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from django.contrib.auth.mixins import LoginRequiredMixin
from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from django.views import View

from dotenv import load_dotenv
load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").lower()
openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

def sanitize_email(email):
    return email.replace('@', '_at_').replace('.', '_dot_')
    

class SignupView(APIView):
    def get(self, request):
        return render(request, 'echovault/signup.html')

    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return redirect('login_cbv')

        return render(
            request,
            'echovault/signup.html',
            {
                'errors': serializer.errors,
                'data': request.data  # Pass user input back to form
            }
        )


class LoginView(APIView):
    def get(self, request):
        return render(request, 'echovault/login.html')

    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")
        user = authenticate(email=email, password=password)
        if user:
            auth_login(request, user)
            return redirect('upload_video_cbv')
        return render(request, 'echovault/login.html', {'error': 'Invalid credentials'})

class UploadVideoView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request):
        user_videos = Video.objects.filter(user=request.user).order_by('-id')
        return render(request, 'echovault/upload_video.html', {
            'user': request.user,
            'all_videos': user_videos
        })


    def post(self, request):
        username = request.user.email
        safe_username = username.replace("@", "_at_").replace(".", "_")
        print("username -> ", safe_username)

        video_file = request.FILES.get('video')
        if not video_file:
            return render(request, 'echovault/upload_video.html', {'error': 'No video uploaded'})

        video_instance = Video.objects.create(user=request.user, video_file=video_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            for chunk in video_file.chunks():
                temp_video.write(chunk)
            video_path = temp_video.name

        audio_path = video_path.replace('.mp4', '.wav')
        subprocess.run(['ffmpeg', '-i', video_path, audio_path], check=True)
        audio = AudioSegment.from_wav(audio_path)
        audio.export(audio_path, format='wav')

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)
        model = whisper.load_model("base", device=device)

        res = model.transcribe(audio_path)
        transcript_text = res["text"]
        print("transcript text -> ", transcript_text)

        # Save transcript to Video model
        video_instance.transcript = transcript_text
        

        transcript_path = audio_path.replace('.wav', '.txt')
        print(' transcript path ->',transcript_path)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)

        # Save embeddings in persistent Chroma vectorstore
        if MODEL_PROVIDER == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)
        else:
            # from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(api_key=openai_key)
        persist_path = os.path.join(settings.BASE_DIR, "vectorstore", safe_username)
        os.makedirs(persist_path, exist_ok=True)

        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(transcript_path)

        if loader:
            documents = loader.load()

            # Add filename as metadata to each document
            from pathlib import Path
            filename = Path(video_instance.video_file.name).name
            print(" metadata filename -> ",filename)
            for doc in documents:
                doc.metadata['filename'] = filename
        
        # summary code
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])

        if MODEL_PROVIDER == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=gemini_key)
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(api_key=openai_key)

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document

        # Step 1: Split full text into smaller chunks
        summary_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        summary_chunks = summary_splitter.create_documents([full_text])

        chunk_summaries = []

        for i, chunk in enumerate(summary_chunks):
            prompt = f"Summarize the following transcript chunk:\n\n{chunk.page_content}"
            try:
                response = llm.invoke(prompt)
                chunk_summaries.append(response.content if hasattr(response, "content") else response)
            except Exception as e:
                print(f"Error summarizing chunk {i}: {e}")
                continue

        # Step 2: Combine individual chunk summaries
        final_summary = "\n".join(chunk_summaries)
        print("Final Summary ->", final_summary)

        # Save to the video instance
        video_instance.summary = final_summary
        video_instance.save()

        # --- New chunking code starts here ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(chunks, embedding=embeddings, collection_name=safe_username,persist_directory=persist_path)
        vectorstore.persist()

        context = {
    'video': video_instance,
    'transcript': transcript_text,
    'all_videos': Video.objects.filter(user=request.user).order_by('-id')  # latest first
}
        return render(request, 'echovault/upload_result.html', context)


class QueryVectorstoreView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        query_text = request.POST.get('query')
        filename = request.POST.get('filename')  # Get filename from POST data
        username = request.user.email
        safe_username = username.replace("@", "_at_").replace(".", "_")

        answer = None
        if query_text:
            persist_path = os.path.join(settings.BASE_DIR, "vectorstore", safe_username)
            if MODEL_PROVIDER == "gemini":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)
            else:
                # from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(api_key=openai_key)
                
            vectorstore = Chroma(
                collection_name=safe_username,
                embedding_function=embeddings,
                persist_directory=persist_path
            )

            # Use metadata filter on filename
            search_kwargs = {"filter": {"filename": filename}} if filename else {}
            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

            if MODEL_PROVIDER == "gemini":
                # llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_key)
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=gemini_key)
            else:
                llm = OpenAI(api_key=openai_key)

            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            answer = qa.run(query_text)

        # Get the correct video by matching the filename
        all_videos = Video.objects.filter(user=request.user).order_by('-id')
        print("all_videos -> ",all_videos)
        selected_video = next((vid for vid in all_videos if os.path.basename(vid.video_file.name) == filename), None)

        print("selected video -> ",selected_video)

        context = {
            'query': query_text,
            'answer': answer,
            'filename': filename,
            'selected_filename': filename,
            'video': selected_video,
            'all_videos': all_videos,
            'transcript': selected_video.transcript if selected_video and selected_video.transcript else '',
        }

        return render(request, 'echovault/upload_result.html', context)


class UploadResultView(LoginRequiredMixin, View):
    def get(self, request):
        user = request.user
        videos = Video.objects.filter(user=user).order_by('-id')
        context = {
            'video': videos.first() if videos else None,
            'transcript': videos.first().transcript if videos and videos.first().transcript else '',
            'all_videos': videos
        }
        return render(request, 'echovault/upload_result.html', context)


class OCRSearchView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return render(request, 'echovault/ocr_search.html')

    def post(self, request):
        keyword = request.data.get('keyword')
        video_path = request.POST.get('video_path')
        found = False
        if video_path:
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                text = pytesseract.image_to_string(frame)
                if keyword.lower() in text.lower():
                    found = True
                    break
            cap.release()
        context = {'found': found, 'keyword': keyword, 'video_path': video_path}
        return render(request, 'echovault/ocr_result.html', context)


