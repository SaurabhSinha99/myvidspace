from rest_framework import serializers
from .models import Video
import re

class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Video
        fields = ['id', 'video_file', 'uploaded_at']
# serializers.py
from rest_framework import serializers
from .models import CustomUser

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['id', 'first_name', 'last_name', 'email', 'mobile_number', 'password']
        extra_kwargs = {'password': {'write_only': True}}

    def validate_password(self, value):
        # Minimum 8 characters, at least 1 letter, 1 digit, and 1 special character
        if len(value) < 8:
            raise serializers.ValidationError("Password must be at least 8 characters long.")
        if not re.search(r"[A-Za-z]", value):
            raise serializers.ValidationError("Password must contain at least one letter.")
        if not re.search(r"\d", value):
            raise serializers.ValidationError("Password must contain at least one digit.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", value):
            raise serializers.ValidationError("Password must contain at least one special character.")
        return value

    def create(self, validated_data):
        return CustomUser.objects.create_user(**validated_data)
