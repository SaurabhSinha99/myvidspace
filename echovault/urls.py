# urls_cbv.py
from django.urls import path
from .views import SignupView, LoginView, UploadVideoView, OCRSearchView, QueryVectorstoreView, UploadResultView
from echovault import views

urlpatterns = [
    path('signup/', SignupView.as_view(), name='signup_cbv'),
    path('login/', LoginView.as_view(), name='login_cbv'),
    path('upload/', UploadVideoView.as_view(), name='upload_video_cbv'),
    path('ocr-search/', OCRSearchView.as_view(), name='ocr_search_cbv'),
    path('query-vectorstore/', QueryVectorstoreView.as_view(), name='query_vectorstore'),
    path('upload-result/', UploadResultView.as_view(), name='upload_result_direct'),
]
