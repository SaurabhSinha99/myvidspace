# urls_cbv.py
from django.urls import path
from .views import SignupView, LoginView, UploadVideoView, QueryVectorstoreView, UploadResultView,activate_account
from echovault import views
from .views import password_reset_request,password_reset_confirm
urlpatterns = [
    path('ocr_search/', views.ocr_keyword_search, name='ocr_keyword_search'),
    path('signup/', SignupView.as_view(), name='signup_cbv'),
    path('login/', LoginView.as_view(), name='login_cbv'),
    path('upload/', UploadVideoView.as_view(), name='upload_video_cbv'),
    path('query-vectorstore/', QueryVectorstoreView.as_view(), name='query_vectorstore'),
    path('upload-result/', UploadResultView.as_view(), name='upload_result_direct'),
    path('activate/<uidb64>/<token>/', activate_account, name='activate'),
    path('password-reset/', password_reset_request, name='password_reset_request'),
    path('reset/<uidb64>/<token>/', password_reset_confirm, name='password_reset_confirm'),

]
