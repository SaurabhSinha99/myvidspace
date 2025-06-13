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
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from django.contrib.auth.mixins import LoginRequiredMixin
from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.utils.http import urlsafe_base64_decode
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.shortcuts import redirect
from django.http import HttpResponse
from django.views import View
from pathlib import Path
import glob
import subprocess
import numpy as np
from PIL import Image, ImageChops
from django.shortcuts import render, get_object_or_404
from echovault.utils import extract_slides
from django.utils.encoding import force_bytes, force_str
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").lower()
openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

def sanitize_email(email):
    return email.replace('@', '_at_').replace('.', '_dot_')

User = get_user_model()

def activate_account(request, uidb64, token):
    try:
        uid = urlsafe_base64_decode(uidb64).decode()
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        return redirect('login_cbv')  # or render a success page
    else:
        return HttpResponse('Activation link is invalid or expired!')
    
class SignupView(APIView):
    def get(self, request):
        return render(request, 'echovault/signup.html')

    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            user.is_active = False  # deactivate until email verification
            user.save()

            # Create token and uid
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)

            # Build activation link
            current_site = get_current_site(request)
            activation_link = request.build_absolute_uri(
                reverse('activate', kwargs={'uidb64': uid, 'token': token})
            )

            # Send email
            subject = 'Activate your EchoVault account'
            message = f'Hi {user.first_name},\n\nPlease click the link below to verify your email and activate your account:\n{activation_link}\n\nThanks,\nEchoVault Team'
            send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [user.email])

            return render(request, 'echovault/email_sent.html')  # show a message to check inbox

        return render(request, 'echovault/signup.html', {
            'errors': serializer.errors,
            'data': request.data
        })
# class SignupView(APIView):
#     def get(self, request):
#         return render(request, 'echovault/signup.html')

#     def post(self, request):
#         serializer = UserSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return redirect('login_cbv')

#         return render(
#             request,
#             'echovault/signup.html',
#             {
#                 'errors': serializer.errors,
#                 'data': request.data  # Pass user input back to form
#             }
#         )


class LoginView(APIView):
    def get(self, request):
        return render(request, 'echovault/login.html')

    def post(self, request):
        email = request.POST.get("email")
        password = request.POST.get("password")
        user = authenticate(email=email, password=password)

        if user:
            auth_login(request, user)
            return redirect('upload_video_cbv')
        
        # If authentication fails, return error to template
        return render(request, 'echovault/login.html', {
            'error': 'Invalid email or password.',
            'email': email  # Optional: pre-fill the email field
        })
    
def password_reset_request(request):
    if request.method == "POST":
        email = request.POST.get("email")
        user = CustomUser.objects.filter(email=email).first()
        if user:
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)

            reset_link = request.build_absolute_uri(
                reverse('password_reset_confirm', kwargs={'uidb64': uid, 'token': token})
            )

            subject = 'Reset your EchoVault password'
            message = f'Hi {user.first_name},\n\nClick the link below to reset your password:\n{reset_link}\n\nIf you did not request this, please ignore this email.\n\n- EchoVault Team'
            send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [user.email])
            print(" email sent")
            return render(request, 'echovault/email_sent.html')

        return render(request, 'echovault/password_reset_request.html', {'error': 'Email not found'})

    return render(request, 'echovault/password_reset_request.html')

def password_reset_confirm(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = CustomUser.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, CustomUser.DoesNotExist):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        if request.method == "POST":
            new_password = request.POST.get("password")
            user.set_password(new_password)
            user.save()
            return redirect('login_cbv')
        return render(request, 'echovault/password_reset_confirm.html', {'validlink': True})
    else:
        return render(request, 'echovault/password_reset_confirm.html', {'validlink': False})

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

        print("Starting splitting frames")
        # Path()

        # frames_dict = extract_slides(vid_obj=video_instance,video_path=video_path)
        
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
    'summary': final_summary,
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
            print("search kwargs -> ",search_kwargs)
            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

            if MODEL_PROVIDER == "gemini":
                # llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_key)
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=gemini_key)
            else:
                llm = OpenAI(api_key=openai_key)

            # qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            # answer = qa.run(query_text)

            # Custom prompt: Only answer if it's in the context
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    "You are an expert assistant. Use only the provided context to answer the question. "
                    "If the answer is not in the context, say: 'The answer is not available in the provided information.'\n\n"
                    "Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
                )
            )

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt_template}
            )
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
            'summary': selected_video.summary if selected_video and selected_video.summary else '',

        }

        return render(request, 'echovault/upload_result.html', context)


class UploadResultView(LoginRequiredMixin, View):
    def get(self, request):
        user = request.user
        videos = Video.objects.filter(user=user).order_by('-id')
        context = {
            'video': videos.first() if videos else None,
            'transcript': videos.first().transcript if videos and videos.first().transcript else '',
            'summary': videos.first().summary if videos and videos.first().summary else '',
            'all_videos': videos
        }
        return render(request, 'echovault/upload_result.html', context)
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
@csrf_exempt
def ocr_keyword_search(request):
    if request.method == "GET":
        return JsonResponse("success",safe=False)
    if request.method == "POST":
        body = json.loads(request.body)
        print("body -> ",body)
        video_name = body.get("video_name")
        video_name = f'videos/{video_name}'
        print(" video name -> ",video_name)
        # video_name = "videos/_Set_up_a_recovery_phone_number_or_e_-_Notepad_2025-05-30_15-45-08_soYzdNK.mp4"
        keyword = body.get("keyword", "").lower()
        print("working")
        video = get_object_or_404(Video, video_file=video_name)
        frames = video.frames_data or []
        print("working 2")

        matches = []
        for frame in frames:
            ocr_text = frame.get("ocr_text", "")
            if keyword in ocr_text.lower():
                matches.append({
                    "time_beg": frame.get("time_beg", 0),
                    "text": ocr_text.strip(),
                })

        return JsonResponse({"matches": matches})


# class OCRKeywordSearchView(APIView):
#     permission_classes = [IsAuthenticated]  # Optional: customize as needed

#     def post(self, request, *args, **kwargs):
#         video_name = request.data.get("video_name")
#         keyword = request.data.get("keyword", "").lower()

#         video = get_object_or_404(Video, video_file=video_name)
#         frames = video.frames_data or []

#         matches = []
#         for frame in frames:
#             ocr_text = frame.get("ocr_text", "")
#             if keyword in ocr_text.lower():
#                 matches.append({
#                     "time_beg": frame.get("time_beg", 0),
#                     "text": ocr_text.strip(),
#                 })

#         return Response({"matches": matches})
    


