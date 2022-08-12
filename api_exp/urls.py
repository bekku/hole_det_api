from django.contrib import admin
from django.urls import path, include

from rest_framework import routers
from .api_view import TemplateViewSet
from . import views
from . import predict_image_from_base64 as pred

router = routers.DefaultRouter()
router.register('template', TemplateViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
    path('api/v1/image',
        pred.PredictImageFromBase64.as_view(), name='pred-v'),
]
